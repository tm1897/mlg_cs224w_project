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

        self.latent_dim = latent_dim
        self.num_layers = args.num_layers
        self.dataset = dataset
        self.embeddings = torch.nn.Embedding(
            num_embeddings= dataset.num_nodes, embedding_dim=self.latent_dim)
        print('ok')

    def reset_parameters(self):
        self.embeddings.reset_parameters()

    def forward(self):
        x, edge_index, batch = self.embeddings.weight, self.dataset.train_pos_edge_index, self.dataset.batch

        final_embeddings = torch.zeros(size=x.size(), device='cuda')
        final_embeddings = final_embeddings + x/(self.num_layers+1)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            final_embeddings = final_embeddings + x/(self.num_layers+1)

        return final_embeddings

    def decode(self, z, pos_edge_index, neg_edge_index):  # only pos and neg edges
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)  # concatenate pos and neg edges
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)  # dot product
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t()  # get adj NxN
        #return (prob_adj > 0).nonzero(as_tuple=False).t()  # get predicted edge_list
        return prob_adj


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

