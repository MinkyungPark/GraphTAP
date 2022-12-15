# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch


def pad_1d_unsqueeze(x, padlen):
    b = x.size(0)
    x = x + 1  # pad id = 0
    xlen = x.size(1)
    if xlen < padlen:
        new_x = x.new_zeros([b, padlen], dtype=x.dtype)
        new_x[:, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen, mask=True):
    b = x.size(0)
    if not mask:
        x = x + 1  # pad id = 0
    xlen, xdim = x.size()[1:]
    if xlen < padlen:
        new_x = x.new_zeros([b, padlen, xdim], dtype=x.dtype)
        new_x[:, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    b = x.size(0)
    xlen = x.size(1)
    if xlen < padlen:
        new_x = x.new_zeros([b, padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:, :xlen, :xlen] = x
        new_x[:, xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    b = x.size(0)
    xlen = x.size(1)
    if xlen < padlen:
        new_x = x.new_zeros([b, padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:, :xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    b = x.size(0)
    x = x + 1
    xlen = x.size(1)
    if xlen < padlen:
        new_x = x.new_zeros([b, padlen, padlen], dtype=x.dtype)
        new_x[:, :xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    b, xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([b, padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:, :xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)

def cat_size(x, B, T):
    s = [B*T]
    for d in [s for s in x.size()[2:]]:
        s.append(d)
    x = x.view(s)
    return x

def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    B, T = len(items), items[0].x.size(0)
    items = [item for item in items if item is not None and item.x.size(1) <= max_node]
    items = [
        (
            item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.y,
            item.padding_mask,
            item.seq_X,
            item.seq_Y,
            item.mask
        )
        for item in items
    ]
    (
        idxs,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        ys,
        padding_masks,
        seq_xs,
        seq_ys,
        masks
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][:, 1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
    max_node_num = max(i.size(1) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    
    if ys[0] is not None:
        y = torch.cat(ys)
    
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    padding_mask = torch.cat(
        [pad_2d_unsqueeze(i, max_node_num) for i in padding_masks]
    ).bool()
    x[padding_mask] = 0
    
    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    )
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])
    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=cat_size(attn_bias, B, T),
        attn_edge_type=cat_size(attn_edge_type, B, T),
        spatial_pos=cat_size(spatial_pos, B, T),
        in_degree=cat_size(in_degree, B, T),
        out_degree=cat_size(in_degree, B, T), # undirect graph
        x=cat_size(x, B, T),
        edge_input=cat_size(edge_input, B, T),
        mask=torch.stack(masks),
        padding_mask=cat_size(padding_mask, B, T),
        # y=y,
        batch_size=B,
        timesteps=T,
        seq_X=torch.stack(seq_xs),
        seq_Y=torch.stack(seq_ys)
    )
