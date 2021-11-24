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
        self.embeddings_users = torch.nn.Embedding(num_embeddings= dataset.num_users, embedding_dim=self.latent_dim)
        self.embeddings_artists = torch.nn.Embedding(num_embeddings= dataset.num_artists, embedding_dim=self.latent_dim)

    def reset_parameters(self):
        self.embeddings.reset_parameters()

    def forward(self):
        x_users, x_artists, batch = self.embeddings_users.weight, self.embeddings_artists.weight, \
                                                self.dataset.batch

        final_embeddings_users = torch.zeros(size=x_users.size(), device='cuda')
        final_embeddings_artists = torch.zeros(size=x_artists.size(), device='cuda')
        final_embeddings_users = final_embeddings_users + x_users/(self.num_layers + 1)
        final_embeddings_artists = final_embeddings_artists + x_artists/(self.num_layers+1)
        for i in range(self.num_layers):
            x_users = self.convs[i]((x_artists, x_users), self.dataset.edge_index_a2u, size=(self.dataset.num_artists, self.dataset.num_users))
            x_artists = self.convs[i]((x_users, x_artists), self.dataset.edge_index_u2a, size=(self.dataset.num_users, self.dataset.num_artists))
            final_embeddings_users = final_embeddings_users + x_users/(self.num_layers+1)
            final_embeddings_artists = final_embeddings_artists + x_artists/(self.num_layers + 1)

        return final_embeddings_users, final_embeddings_artists

    def decode(self, z1, z2, pos_edge_index, neg_edge_index):  # only pos and neg edges
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)  # concatenate pos and neg edges
        logits = (z1[edge_index[0]] * z2[edge_index[1]]).sum(dim=-1)  # dot product
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
        return self.propagate(edge_index=edge_index, x=(x[0], x[1]), size=size)

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index, dim_size=None):
        return torch_scatter.scatter(src=inputs, index=index, dim=0, dim_size=dim_size, reduce='mean')

