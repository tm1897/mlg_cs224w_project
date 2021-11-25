import torch
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def get_link_labels(pos_edge_index, neg_edge_index):
    # returns a tensor:
    # [1,1,1,1,...,0,0,0,0,0,..] with the number of ones is equel to the lenght of pos_edge_index
    # and the number of zeros is equal to the length of neg_edge_index
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device='cuda')
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def train(model, data, optimizer):
    model.train()
    data.neg_edge_index_u2a = negative_sampling(
        edge_index=data.edge_index_u2a,  # positive edges
        num_nodes=(data.num_users, data.num_artists),  # number of nodes
        num_neg_samples=data.edge_index_u2a.size(1),
        method='sparse').to('cuda')  # number of neg_sample equal to number of pos_edges

    data.neg_edge_index_a2u = negative_sampling(
        edge_index=data.edge_index_a2u,  # positive edges
        num_nodes=(data.num_artists, data.num_users),  # number of nodes
        num_neg_samples=data.edge_index_u2a.size(1),
        method='sparse').to('cuda')

    optimizer.zero_grad()

    z_users, z_artists = model.forward()  # encode
    link_logits_u2a = model.decode(z1=z_users, z2=z_artists, pos_edge_index=data.edge_index_u2a,
                                   neg_edge_index=data.neg_edge_index_u2a)  # decode
    link_logits_a2u = model.decode(z1=z_artists, z2=z_users, pos_edge_index=data.edge_index_a2u,
                                   neg_edge_index=data.neg_edge_index_a2u)  # decode

    link_labels_u2a = get_link_labels(data.edge_index_u2a, data.neg_edge_index_u2a)
    link_labels_a2u = get_link_labels(data.edge_index_a2u, data.neg_edge_index_u2a)
    loss = F.binary_cross_entropy_with_logits(link_logits_u2a, link_labels_u2a) + \
           F.binary_cross_entropy_with_logits(link_logits_a2u, link_labels_a2u)

    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def test(model, data_val_test):
    model.eval()
    perfs = []
    for data in data_val_test:
        z_users, z_artists = model.forward()  # encode train
        link_logits_u2a = model.decode(z1=z_users, z2=z_artists, pos_edge_index=data.edge_index_u2a,
                                   neg_edge_index=data.neg_edge_index_u2a)  # decode
        link_logits_a2u = model.decode(z1=z_artists, z2=z_users, pos_edge_index=data.edge_index_a2u,
                                   neg_edge_index=data.neg_edge_index_a2u)  # decode # decode test or val
        link_probs_u2a = link_logits_u2a.sigmoid()  # apply sigmoid
        link_probs_a2u = link_logits_a2u.sigmoid()

        link_labels_u2a = get_link_labels(data.edge_index_u2a, data.neg_edge_index_u2a)
        link_labels_a2u = get_link_labels(data.edge_index_a2u, data.neg_edge_index_a2u)# get link

        perfs.append(roc_auc_score(link_labels_u2a.cpu(), link_probs_u2a.cpu())
                                   + roc_auc_score(link_labels_a2u.cpu(), link_probs_a2u.cpu())/2)  # compute roc_auc score
    return perfs
