import torch
from cmfrec import CMF_implicit
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn.conv import MessagePassing
from scipy.sparse import coo_matrix

def to_scipy_sparse_matrix(edge_index, num_nodes):
    row, col = edge_index.cpu()
    edge_attr = torch.ones(row.size(0))
    out = coo_matrix(
        (edge_attr.numpy(), (row.numpy(), col.numpy())), (num_nodes[0], num_nodes[1]))
    return out



class LightGCNStack(torch.nn.Module):
    def __init__(self, args):
        super(LightGCNStack, self).__init__()
        self.latent_dim = args.latent_dim
        conv_model = LightGCN
        self.convs = nn.ModuleList()
        self.convs.append(conv_model())
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers-1):
            self.convs.append(conv_model())

        self.num_layers = args.num_layers
        self.dataset = None
        self.embeddings_users = None
        self.embeddings_artists = None
        self.lambda_reg = args.lambda_reg
        self.hot_start = args.hot_start

    def reset_parameters(self):
        self.embeddings.reset_parameters()

    def init_data(self, dataset):
        self.dataset = dataset
        if self.hot_start:
            model = CMF_implicit(
                k=self.latent_dim,
                nonneg=False,
                random_state=1,
                niter=100
            )
            model.fit(to_scipy_sparse_matrix(self.dataset.edge_index_u2a, num_nodes=(self.dataset.num_users, self.dataset.num_artists)))
            self.embeddings_users = nn.Embedding.from_pretrained(torch.FloatTensor(model.A_), freeze=False).to('cuda')
            self.embeddings_artists = nn.Embedding.from_pretrained(torch.FloatTensor(model.B_), freeze=False).to('cuda')
        else:
            self.embeddings_users = torch.nn.Embedding(num_embeddings=dataset.num_users,
                                                       embedding_dim=self.latent_dim).to('cuda')
            self.embeddings_artists = torch.nn.Embedding(num_embeddings=dataset.num_artists,
                                                         embedding_dim=self.latent_dim).to('cuda')

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

    def decode_all(self, z_users, z_artists):
        prob_adj = z_users @ z_artists.t()  # get adj NxN
        #return (prob_adj > 0).nonzero(as_tuple=False).t()  # get predicted edge_list
        return prob_adj

    def BPRLoss(self, prob_adj, real_adj, edge_index):
        loss = 0
        pos_scores = prob_adj[edge_index.cpu().numpy()]
        for pos_score, node_index in zip(pos_scores, edge_index[0]):
            neg_scores = prob_adj[node_index, real_adj[node_index] == 0]
            loss = loss - torch.sum(torch.log(torch.sigmoid(pos_score.repeat(neg_scores.size()[0]) - neg_scores))) / \
                   neg_scores.size()[0]

        #loss += self.lambda_reg*(torch.pow(torch.norm(self.embeddings_users.weight, dim=None), 2) +
                                 #torch.pow(torch.norm(self.embeddings_artists.weight), 2))

        return loss


    def topN(self, user_id, n):
        z_users, z_artists = self.forward()
        scores = torch.squeeze(z_users[user_id] @ z_artists.t())
        return torch.topk(scores, k=n)


class LightGCN(MessagePassing):
    def __init__(self, **kwargs):
        super(LightGCN, self).__init__(node_dim=0, **kwargs)


    def forward(self, x, edge_index, size=None):
        return self.propagate(edge_index=edge_index, x=(x[0], x[1]), size=size)

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index, dim_size=None):
        return torch_scatter.scatter(src=inputs, index=index, dim=0, dim_size=dim_size, reduce='mean')

