import os
import dill
from copy import deepcopy
import json
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple
from tqdm import tqdm
from functools import lru_cache

import numpy as np
import torch
from torch_geometric.data import Data
# from collections import namedtuple

from examples.graph_diffusion.data.preprocessing import QuantileDiscretizer
from examples.graph_diffusion.data.seq_dataset import SequenceDataset
from examples.graph_diffusion.utils.utils import to_torch
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
import algos

# ---------------- Graph Dataset ---------------- #
@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

def preprocess_item(items):
    xs, attn_biases, attn_edge_types, spatial_poses, adjs, edge_inputs = [],[],[],[],[],[]
    if 'edge_attr' not in items.keys:
        items.edge_attr = torch.arange(items.edge_index.size(1))
        
    for x in items.x:
        edge_attr, edge_index = items.edge_attr, items.edge_index
        N = x.size(0)
        x = convert_to_single_emb(x)

        # node adj matrix [N, N] bool
        adj = torch.zeros([N, N], dtype=torch.bool)
        adj[edge_index[0, :], edge_index[1, :]] = True

        # edge feature here
        if len(edge_attr.size()) == 1:
            edge_attr = edge_attr[:, None]
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
        attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
            convert_to_single_emb(edge_attr) + 1
        )

        shortest_path_result, path = algos.floyd_warshall(adj.numpy())
        max_dist = np.amax(shortest_path_result)
        edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
        spatial_pos = torch.from_numpy((shortest_path_result)).long()
        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token
        
        xs.append(x)
        attn_biases.append(attn_bias)
        attn_edge_types.append(attn_edge_type)
        spatial_poses.append(spatial_pos)
        adjs.append(adj.long().sum(dim=1).view(-1))
        edge_inputs.append(edge_input)
        
    x = torch.stack(xs).to(xs[0])
    attn_bias = torch.stack(attn_biases).to(attn_biases[0])
    attn_edge_type = torch.stack(attn_edge_types).to(attn_edge_types[0])
    spatial_pos = torch.stack(spatial_poses).to(spatial_poses[0])
    adj = torch.stack(adjs).to(adjs[0])
    edge_input = np.stack(edge_inputs)
    
    # combine
    items.x = x
    items.attn_bias = attn_bias
    items.attn_edge_type = attn_edge_type
    items.spatial_pos = spatial_pos
    items.in_degree = adj
    items.out_degree = items.in_degree  # for undirected graph
    items.edge_input = torch.from_numpy(edge_input).long()

    return items


class MujocoGraphDataset(SequenceDataset):
    def __init__(self, N=100, root_path=None, discretizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # paths
        if discretizer is not None:
            self.discretizer = QuantileDiscretizer(self.joined_raw, N)
        else:
            self.discretizer = None
        self.env_name = self.dataset.split('-')[0]
        asserts = root_path / 'asserts'
        self.cache_path = asserts / f'{self.dataset}_graph_dataset.dill'
        xml_path = asserts / f'{self.env_name}.xml'
        state_space_path = open(asserts / 'state_space.json')
        
        self.vocab_size = N # vocab size
        self.N = N # vocab size
        self.num_virtual_tokens = 1
        
        state_space = json.load(state_space_path)
        self.state_space = state_space[self.dataset]

        tree = ET.parse(xml_path)
        root = tree.getroot()
        self.world_body = root.find('worldbody')
        self.actuator = root.find('actuator')

        self.graph_struc = self._get_graph_structure()
        """
            joined_segmneted E x T x D
                - E : num of episodes
                - T : num of transitions
                - D : 1 transiton Dimension,
                    s_dim + a_dim + r
        """
    def get_discretizer(self):
        return self.discretizer
    
    def _get_graph_structure(self):
        # graph data structure
        node_dict = defaultdict(dict)
        i = 0
        for child in self.world_body.iter():
            if child.tag == 'body':
                name = child.attrib['name']
                node_dict[name] = {
                    'index': i, 
                    'joint': [joint.attrib['name'] for joint in child.findall('joint')], 
                    'direction': [joint.attrib['name'] for joint in child.findall('body')],
                    'node_features': []
                    }
                i += 1
        self.num_node = len(node_dict.keys())
        num_joint = max([len(v["joint"]) for v in node_dict.values()]) * 2
        self.max_feature_dim = max([num_joint, self.a_dim])
        return node_dict
    
    def _get_data(self, idx):
        path_ind, start_ind, end_ind = self.indices[idx]

        joined = self.joined_segmented[path_ind, start_ind:end_ind:self.step]
        terminations = self.termination_flags[path_ind, start_ind:end_ind:self.step]
        if self.discretizer is not None:
            joined = self.discretizer.discretize(joined)

        ## replace with termination token if the sequence has ended
        # if not (joined[terminations] == 0).all():
        #     print()
        # assert (joined[terminations] == 0).all(), \
        #         f'Everything after termination should be 0: {path_ind} | {start_ind} | {end_ind}'
        joined[terminations] = self.N

        ## [ (sequence_length / skip) x observation_dim]
        if self.discretizer is not None:
            joined = to_torch(joined, device='cpu', dtype=torch.long).contiguous()
        else:
            joined = to_torch(joined, device='cpu', dtype=torch.float).contiguous()
        ## don't compute loss for parts of the prediction that extend
        ## beyond the max path length
        traj_inds = torch.arange(start_ind, end_ind, self.step)
        mask = torch.ones(joined.shape, dtype=torch.bool)
        mask[traj_inds > self.max_path_length - self.step] = 0

        ## flatten everything
        # joined = joined.view(-1)
        # mask = mask.view(-1)

        # X = joined[:-1]
        # Y = joined[1:]
        # mask = mask[:-1]
        return joined, mask # T x D
    
    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        joined, mask = self._get_data(idx)
        seq_X = joined[:-1]
        seq_Y = joined[1:]
        mask = mask[:-1]
        
        t, d = joined.size()
        s, a, r = self.s_dim, self.a_dim, d - (self.s_dim+self.a_dim)
        
        obs, acts, rs = joined[:, :s], joined[:, s:s+a], joined[:, -r:]

        node_dict = deepcopy(self.graph_struc)
        # all_observations (num_epi * len_seq, obs_dim)
        for state_name, each_axis_obs in zip(self.state_space, obs.permute(1,0)):
            for body, node_values in self.graph_struc.items():
                if state_name in node_values['joint']:
                    node = body
            # (node_feature_dim, num_epi * len_seq)
            node_dict[node]['node_features'].append(each_axis_obs.tolist())
        
        node_features = torch.zeros((t, self.num_node + 2, self.max_feature_dim))
        if self.discretizer is not None:
            node_features = node_features.long()
        # num_node : body obs nodes, 1 : action node, 1: reward node

        e_start, e_end = [], []
        for body, node_values in node_dict.items():
            # edge sindex
            n_i = node_values['index']
            n_adj = len([node_values['direction']])
            if n_adj > 0:
                for adj in node_values['direction']:
                    e_start.append(n_i)
                    e_end.append(node_dict[adj]['index'])
            
            if self.discretizer is not None:
                features = torch.LongTensor(node_values['node_features']).permute(1,0)
            else:
                features = torch.FloatTensor(node_values['node_features']).permute(1,0)
            # features = torch.FloatTensor(node_values['node_features'])
            # node feature (total_transition, num_node, node_feature_dim)
            node_features[:, n_i, :features.size(1)] = features
            
        # action_rewards edge, only halfcheeta yet.
        # 0: root, 1: bthigh, 2:bshin, 3:bfoot, 4: fthigh, 5:fshin, 6:ffoot
        if self.discretizer is not None:
            node_features[:, -2, :a] = torch.LongTensor(acts)
            node_features[:, -1, :r] = torch.LongTensor(rs)
        else:
            node_features[:, -2, :a] = torch.FloatTensor(acts)
            node_features[:, -1, :r] = torch.FloatTensor(rs)
        
        act_edge_start, act_edge_end = [len(e_start)] * len(e_start), [i for i in range(len(e_start))]
        rew_edge_start, rew_edge_end = [len(e_start) + 1], [len(e_start)]
        
        obs_edge_attr = [i for i in range(len(e_start))]

        e_start = e_start + act_edge_start + rew_edge_start
        e_end = e_end + act_edge_end + rew_edge_end
        edge_index = torch.LongTensor([e_start, e_end])
        
        act_edge_attr = [obs_edge_attr[-1] + 1] * len(act_edge_start)
        rew_edge_attr = [act_edge_attr[-1] + 1]
        
        padding_mask = (node_features == 0)
        
        edge_attr = torch.LongTensor(obs_edge_attr + act_edge_attr + rew_edge_attr)
        item = preprocess_item(Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr))
        
        item.mask = mask
        item.idx = idx
        item.padding_mask = padding_mask
        item.seq_X = seq_X
        item.seq_Y = seq_Y
        
        return item
    
    def reconstruct(self, graph):
        halfcheeta = [
        (0,0), (0,1), 
        (1,0), (2,0), (3,0), (4,0), (5,0), (6,0), 
        (0,2), (0,3), (0,4), 
        (1,1), (2,1), (3,1), (4,1), (5,1), (6,1), # obs
        (7,0), (7,1), (7,2), (7,3), (7,4), (7,5), # action
        (8,0)] # reward