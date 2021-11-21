import torch

import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn.conv import MessagePassing

class LightGCNStack(torch.nn.Module):
    def __init__(self, latent_dim, dataset, args):
        super(LightGCNStack, self).__init__()
        conv_model = LightGCN
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(latent_dim))
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers-1):
            self.convs.append(conv_model(latent_dim))

        self.num_layers = args.num_layers
        self.dataset = dataset
        self.embedding_user = torch.nn.Embedding(
            num_embeddings= dataset.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings= dataset.num_items, embedding_dim=self.latent_dim)

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

    def forward(self, x, edge_index, size=None):
        return self.propagate(edge_index=edge_index, x=(x,x), size=size)

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index, dim_size=None):
        return torch_scatter.scatter(src=inputs, index=index, dim=0, dim_size=dim_size, reduce='mean')

