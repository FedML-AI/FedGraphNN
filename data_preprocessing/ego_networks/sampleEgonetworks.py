import os
import argparse
import pickle
import random
import time

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Coauthor, CitationFull, TUDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, degree, k_hop_subgraph


# citation networks (# nodes): e.g. DBLP (17716), Core (2708), CiteSeer (3327), PubMed (19717)
# collaboration networks (# nodes): e.g. CS (18333), Physics (34493)
# social networks (# ego-networks): e.g. COLLAB, IMDB, DEEZER_EGO_NETS (9629), TWITCH_EGOS (127094)


def _convert_to_nodeDegreeFeatures(graphs):
    graph_infos = []
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree
        graph_infos.append(
            (graph, g.degree, graph.num_nodes)
        )  # (graph, node_degrees, num_nodes)

    new_graphs = []
    for i, tpl in enumerate(graph_infos):
        idx, x = tpl[0].edge_index[0], tpl[0].x
        deg = degree(idx, tpl[2], dtype=torch.long)
        deg = F.one_hot(deg, num_classes=maxdegree + 1).to(torch.float)

        new_graph = tpl[0].clone()
        new_graph.__setitem__("x", deg)
        new_graphs.append(new_graph)

    return new_graphs


def _to_egoNet(g, ego, hop_number):
    # get ego-networks for sampled nodes
    sub_nodes, sub_edge_index, _, _ = k_hop_subgraph(ego, hop_number, g.edge_index)

    def re_index(source):
        mapping = dict(zip(sub_nodes.numpy(), range(sub_nodes.shape[0])))
        return mapping[source]

    edge_index_u = [*map(re_index, sub_edge_index[0][:].numpy())]
    edge_index_v = [*map(re_index, sub_edge_index[1][:].numpy())]

    egonet = Data(
        edge_index=torch.tensor([edge_index_u, edge_index_v]),
        x=g.x[sub_nodes],
        y=g.y[sub_nodes],
    )
    return egonet


def _get_egonetworks(g, ego_number, hop_number):
    ego_number = min(ego_number, g.num_nodes)

    num_features = g.num_node_features
    num_labels = len(g.y.unique())

    # sample central nodes
    # nodes are names as 0, 1, 2, ...
    egos = random.sample(range(g.num_nodes), ego_number)
    egonetworks = [_to_egoNet(g, ego, hop_number) for ego in egos]

    return egonetworks, ego_number, num_features, num_labels


def get_data(path, data, ego_number, hop_number):

    start = time.time()
    if data not in ["CS", "Physics", "cora", "citeseer", "DBLP", "PubMed"]:
        raise Exception("no such dataset!")
    elif data in ["CS", "Physics"]:
        pyg_dataset = Coauthor(path, data)
        subgraphs, num_graphs, num_features, num_labels = _get_egonetworks(
            pyg_dataset[0], ego_number, hop_number
        )
    else:
        pyg_dataset = CitationFull(path, data)
        subgraphs, num_graphs, num_features, num_labels = _get_egonetworks(
            pyg_dataset[0], ego_number, hop_number
        )

    # if type_network == 'social':
    #     pyg_dataset = TUDataset(os.path.join(path, "TUDataset"), data)
    #     ego_number = min(ego_number, len(pyg_dataset))
    #     subgraphs = random.sample(list(pyg_dataset), ego_number)
    print("Sampled ego-networks:", time.time() - start)

    start = time.time()
    if not subgraphs[0].__contains__("x"):
        subgraphs = _convert_to_nodeDegreeFeatures(subgraphs)
        print("Converted node features:", time.time() - start)

    return subgraphs, num_graphs, num_features, num_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", help="the path for loading dataset", type=str, default="./data"
    )
    parser.add_argument("--data", help="the name of dataset", type=str)
    # parser.add_argument('--type_network', help='the type of dataset: ["citation", "coauthor"]',
    #                     type=str)
    parser.add_argument(
        "--ego_number", help="the number of egos sampled", type=int, default=1000
    )
    parser.add_argument("--hop_number", help="the number of hops", type=int, default=2)

    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    subgraphs, num_graphs, num_features, num_labels = get_data(
        args.path, args.data, args.ego_number, args.hop_number
    )

    if args.data not in ["CS", "Physics", "cora", "citeseer", "DBLP", "PubMed"]:
        raise Exception("no such dataset!")
    elif args.data in ["CS", "Physics"]:
        outfile = os.path.join(args.path, args.data, "egonetworks.pkl")
    else:
        outfile = os.path.join(args.path, args.data, "egonetworks.pkl")
    # if args.type_network == 'social':
    #     outfile = os.path.join(args.path, "TUDataset", args.data, 'egonetworks.pkl')
    pickle.dump([subgraphs, num_graphs, num_features, num_labels], open(outfile, "wb"))
    print(f"Wrote to {outfile}.")
