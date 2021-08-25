import os
import random
import networkx as nx
import copy
import logging
import pickle
import pandas as pd
from torch_geometric.utils import to_networkx, degree

import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import k_hop_subgraph, from_networkx
from torch_geometric.utils import train_test_split_edges

from FedML.fedml_core.non_iid_partition.noniid_partition import partition_class_samples_with_dirichlet_distribution

def split_graph(graph, train_ratio = 0.8, val_ratio = 0.1, test_ratio = 0.1):
    assert train_ratio + val_ratio + test_ratio == 1
    graph.edge_label = graph.edge_label.float()
    edge_size = graph.edge_label.size()[0]
    train_num = int(edge_size * train_ratio)
    val_num = max(1, int(edge_size * val_ratio))
    test_num = max(1, int(edge_size * test_ratio))
    train_num = edge_size - val_num - test_num
    
    [train_split, val_split, test_split] = torch.utils.data.random_split(range(edge_size), [train_num, val_num, test_num])
    train_split = torch.tensor(train_split)
    val_split = torch.tensor(val_split)
    test_split = torch.tensor(test_split)
    graph.edge_train = graph.edge_index[:, train_split]
    graph.label_train = graph.edge_label[train_split]
    graph.edge_val = graph.edge_index[:, val_split]
    graph.label_val = graph.edge_label[val_split]
    graph.edge_test = graph.edge_index[:, test_split]
    graph.label_test = graph.edge_label[test_split]
    
    return graph

def _convert_to_nodeDegreeFeatures(graphs):
    graph_infos = []
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree
        graph_infos.append((graph, g.degree, graph.num_nodes))    # (graph, node_degrees, num_nodes)

    new_graphs = []
    for i, tpl in enumerate(graph_infos):
        idx, x = tpl[0].edge_index[0], tpl[0].x
        deg = degree(idx, tpl[2], dtype=torch.long)
        deg = F.one_hot(deg, num_classes=maxdegree + 1).to(torch.float)

        new_graph = tpl[0].clone()
        new_graph.__setitem__('x', deg)
        new_graphs.append(new_graph)

    return new_graphs

def _subgraphing(g, partion, mapping_item2category):
    nodelist = [[] for i in set(mapping_item2category.keys())]
    for k, v in partion.items():
        for category in v:
            nodelist[category].append(k)

    graphs = []
    for nodes in nodelist:
        if len(nodes) < 2:
            continue
        graph = nx.subgraph(g, nodes)
        Gcc = sorted(nx.connected_components(graph), key=len, reverse=True)
        G0 = graph.subgraph(Gcc[0])
        graphs.append(from_networkx(G0))
    return graphs


def _read_mapping(path, data, filename):
    mapping = {}
    with open(os.path.join(path, data, filename)) as f:
        for line in f:
            s = line.strip().split()
            mapping[int(s[0])] = int(s[1])
    
    return mapping


def _build_nxGraph(path, data, filename, mapping_user, mapping_item):
    G = nx.Graph()
    with open(os.path.join(path, data, filename)) as f:
        for line in f:
            s = line.strip().split()
            s = [int(i) for i in s]
            G.add_edge(mapping_user[s[0]], mapping_item[s[1]], edge_label=s[2])
    return G


def get_data_category(path, data, algo):
    """ For link prediction. """

    logging.info("read mapping")
    mapping_user = _read_mapping(path, data, 'user.dict')
    mapping_item = _read_mapping(path, data, 'item.dict')
    mapping_item2category = _read_mapping(path, data, 'category.dict')
    logging.info('build networkx graph')

    graph = _build_nxGraph(path, data, 'graph.txt', mapping_user, mapping_item)
    logging.info('get partion')

    partion = partition_by_category(graph, mapping_item2category)
    logging.info('subgraphing')
    graphs = _subgraphing(graph, partion, mapping_item2category)
    logging.info('converting to node degree')
    graphs = _convert_to_nodeDegreeFeatures(graphs)
    graphs_split = []
    logging.info('spliting into trian val and test')
    for g in graphs:
        graphs_split.append(split_graph(g))
    return graphs_split

def partition_by_category(graph, mapping_item2category):
    partition = {}
    for key in mapping_item2category:
        partition[key] = [mapping_item2category[key]]
        for neighbor in graph.neighbors(key):
            if neighbor not in partition:
                partition[neighbor] = []
            partition[neighbor].append(mapping_item2category[key])
    return partition

def create_category_split(path, data, pred_task='link_prediction', algo='Louvain'):
    assert pred_task in ['link_prediction']
    logging.info("reading data")

    graphs_split = get_data_category(path, data, algo)

    return graphs_split

def partition_data_by_category(args, path, compact=True):
    graphs_split = create_category_split(path, args.dataset, args.pred_task)

    client_number = len(graphs_split)

    partition_dicts = [None] * client_number

    for client in range(client_number):

        partition_dict = {'graph': graphs_split[client]}
        partition_dicts[client] = partition_dict

    global_data_dict = {
        'graphs': graphs_split,
        }

    return global_data_dict, partition_dicts


# Single process sequential
def load_partition_data(args, path, client_number, uniform=True, global_test=True, normalize_features=False,
                        normalize_adj=False):
    global_data_dict, partition_dicts = partition_data_by_category(args, path)
    feature_dim = global_data_dict['graphs'][0].x.shape[1]
    client_number = len(partition_dicts)
    args.client_num_in_total = client_number

    data_local_num_dict = {}
    train_data_local_dict = {}

    # This is a PyG Dataloader
    train_data_global = DataLoader(global_data_dict['graphs'], batch_size=1, shuffle=True,
                                        pin_memory=True)

    train_data_num = len(global_data_dict['graphs'])

    for client in range(client_number):
        train_dataset_client = partition_dicts[client]['graph']

        data_local_num_dict[client] = 1
        train_data_local_dict[client] = DataLoader([train_dataset_client], batch_size=1, shuffle=True,
                                                        pin_memory=True)
        logging.info("Client idx = {}, local sample number = {}".format(client, len(train_dataset_client)))

    val_data_num = test_data_num  = train_data_num
    val_data_local_dict = test_data_local_dict = train_data_local_dict
    val_data_global = test_data_global = train_data_global
    return train_data_num, val_data_num, test_data_num, train_data_global, val_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, val_data_local_dict, test_data_local_dict, feature_dim