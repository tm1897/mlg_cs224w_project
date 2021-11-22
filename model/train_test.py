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
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,  # positive edges
        num_nodes=data.num_nodes,  # number of nodes
        num_neg_samples=data.train_pos_edge_index.size(1))  # number of neg_sample equal to number of pos_edges

    optimizer.zero_grad()

    z = model.forward()  # encode
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)  # decode

    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def test(model, data):
    model.eval()
    perfs = []
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']

        z = model.forward()  # encode train
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)  # decode test or val
        link_probs = link_logits.sigmoid()  # apply sigmoid

        link_labels = get_link_labels(pos_edge_index, neg_edge_index)  # get link

        perfs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))  # compute roc_auc score
    return perfs

