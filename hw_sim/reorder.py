import numpy as np

import dgl
import torch


def calc(graph, threshold=90):
    a0 = graph
    u_list = []
    v_list = []
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if not a0[i][j]:
                u_list.append(j)
                v_list.append(i)
    g = dgl.graph((u_list, v_list))
    g.ndata['in_deg'] = g.in_degrees()

    n_node = g.num_nodes()
    n_edge = g.num_edges()
    out_deg = g.out_degrees()
    high_density = out_deg[out_deg > threshold]
    high_density_idx = np.where(out_deg > threshold)[0]

    total = len(high_density_idx)
    tmp1 = 200
    orig_a, orig_b = g.edges()

    total_dense = 0
    for i in high_density_idx:
        total_dense += torch.sum(orig_a == i)

    for i in range(total):
        orig_a[orig_a == i] = tmp1
        orig_b[orig_b == i] = tmp1
        orig_a[orig_a == high_density_idx[i]] = i
        orig_b[orig_b == high_density_idx[i]] = i
        orig_a[orig_a == tmp1] = torch.tensor(high_density_idx[i])
        orig_b[orig_b == tmp1] = torch.tensor(high_density_idx[i])
    dense_cnt = total_dense

    new_graph = torch.ones(graph.shape[0], graph.shape[1])
    for i in range(len(orig_a)):
        try:
            new_graph[orig_b[i], orig_a[i]] = 0
        except:
            pass
    new_graph = new_graph.numpy()
    total_cnt = n_edge

    return dense_cnt, total_cnt, new_graph, total
