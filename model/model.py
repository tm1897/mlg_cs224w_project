import torch

import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn.conv import MessagePassing

class LightGCNStack(torch.nn.Module):
    def __init__(self, latent_dim, args):
        super(LightGCNStack, self).__init__()
        conv_model = LightGCN
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(latent_dim))
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers-1):
            self.convs.append(conv_model(latent_dim))

        self.dropout = args.dropout
        self.num_layers = args.num_layers

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)

        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class LightGCN(MessagePassing):
    def __init__(self, latent_dim, **kwargs):
        super(LightGCN, self).__init__(node_dim=0, **kwargs)
        self.latent_dim = latent_dim

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size=None):
        return self.propagate(edge_index=edge_index, x=(x,x), size=size)

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index, dim_size=None):
        return torch_scatter.scatter(src=inputs, index=index, dim=0, dim_size=dim_size, reduce='mean')

    
if __name__ == '__main__':
    print("Ok")