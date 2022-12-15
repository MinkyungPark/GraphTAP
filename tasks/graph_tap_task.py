# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import contextlib
from dataclasses import dataclass, field
from omegaconf import II, open_dict, OmegaConf
from pathlib import Path

import numpy as np
from fairseq.data import (
    NestedDictionaryDataset,
    NumSamplesDataset,
)
from fairseq.tasks import FairseqDataclass, FairseqTask, register_task

from examples.graph_diffusion.data.graph_dataset import MujocoGraphDataset
from examples.graph_diffusion.data.dataset import BatchedDataDataset
from examples.graph_diffusion.utils.utils import DotConfig, set_seed

logger = logging.getLogger(__name__)

@dataclass
class GraphMujocoConfig(FairseqDataclass):
    max_nodes: int = field(
        default=16,
        metadata={"help": "max nodes per graph"},
    )
    num_atoms: int = field(
        default=512 * 9,
        metadata={"help": "number of atom types in the graph"},
    )
    num_edges: int = field(
        default=512 * 3,
        metadata={"help": "number of edge types in the graph"},
    )
    num_in_degree: int = field(
        default=512,
        metadata={"help": "number of in degree types in the graph"},
    )
    num_out_degree: int = field(
        default=512,
        metadata={"help": "number of out degree types in the graph"},
    )
    num_spatial: int = field(
        default=512,
        metadata={"help": "number of spatial types in the graph"},
    )
    num_edge_dis: int = field(
        default=128,
        metadata={"help": "number of edge dis types in the graph"},
    )
    multi_hop_max_dist: int = field(
        default=5,
        metadata={"help": "max distance of multi-hop edges"},
    )
    spatial_pos_max: int = field(
        default=50,
        metadata={"help": "max distance of multi-hop edges"},
    )
    edge_type: str = field(
        default="multi_hop",
        metadata={"help": "edge type in the graph"},
    )
    train_epoch_shuffle: bool = field(
        default=False,
        metadata={"help": "whether to shuffle the dataset at each epoch"},
    )
    config: str = field(
        default="",
        metadata={"help": "config yaml path"},
    )
    root_path: str = field(
        default="",
        metadata={"help": "save root path"},
    )


@register_task("graph_tap", dataclass=GraphMujocoConfig)
class GraphTapTask(FairseqTask):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.mcfg = DotConfig(cfg.config)
        self.root_path = Path(cfg.root_path)
        set_seed(self.mcfg.seed)
        self.cached_data = self.root_path / 'asserts' / f'{self.mcfg.dataset}_graph_dataset.dill'
        
        seq_length = self.mcfg.subsampled_sequence_length * self.mcfg.step
        self.dm = MujocoGraphDataset(
            env=self.mcfg.dataset,
            N=self.mcfg.N,
            penalty=self.mcfg.termination_penalty,
            sequence_length=seq_length,
            step=self.mcfg.step,
            discount=self.mcfg.discount,
            max_path_length=self.mcfg.max_path_length,
            root_path=self.root_path,
            discretizer=self.mcfg.discretizer
        )
        self.num_node = self.dm.num_node + 2
        s_dim, a_dim, t_dim = self.dm.s_dim, self.dm.a_dim, self.dm.joined_dim
        self.mcfg.observation_dim = s_dim
        self.mcfg.action_dim = a_dim
        # self.mcfg.transition_dim = t_dim + 1 # terminal mask
        # block_size = self.mcfg.subsampled_sequence_length * t_dim + 1
        self.mcfg.transition_dim = t_dim
        block_size = self.mcfg.subsampled_sequence_length * t_dim
        self.mcfg.block_size = block_size
        

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        return cls(cfg)

    def load_dataset(self, split, combine=False, **kwargs):
        if split != 'train': return None
        
        batched_data = BatchedDataDataset(
            self.dm,
            max_node=self.max_nodes(),
            multi_hop_max_dist=self.cfg.multi_hop_max_dist,
            spatial_pos_max=self.cfg.spatial_pos_max,
        )
        data_sizes = np.array([self.max_nodes()] * len(batched_data))
        # target = TargetDataset(batched_data)
        dataset = NestedDictionaryDataset(
            {
                "nsamples": NumSamplesDataset(),
                "net_input": {"batched_data": batched_data},
                "target": None,
            },
            sizes=data_sizes,
        )
        # if split == "train" and self.cfg.train_epoch_shuffle:
        #     dataset = EpochShuffleDataset(
        #         dataset, num_samples=len(dataset), seed=self.mcfg.seed
        #     )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        # self.datasets[split] = dataset
        # return self.datasets[split]
        self.datasets["train"] = dataset
        return self.datasets["train"]

    def build_model(self, cfg):
        from fairseq import models

        with open_dict(cfg) if OmegaConf.is_config(cfg) else contextlib.ExitStack():
            cfg.max_nodes = self.cfg.max_nodes

        model = models.build_model(cfg, self)
        return model

    def max_nodes(self):
        return self.cfg.max_nodes

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None

    @property
    def label_dictionary(self):
        return None
