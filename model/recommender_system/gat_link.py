import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F


class GATLinkPred(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim,  out_channels, dropout,  heads=2 ):
        super(GATLinkPred, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_dim , heads = heads)
        self.conv2 = GATConv(hidden_dim * heads, out_channels, heads = heads)
        self.dropout = dropout
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.conv2(x, edge_index)

    def decode(self, z, pos_edge_index):
        edge_index = pos_edge_index
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
