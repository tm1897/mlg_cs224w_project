import pandas as pd
from torch_geometric.data import Data
import torch

def load_feather():
    df_data = pd.read_feather('../data/user_tag_artist.feather')
    node_ids = 0
    node_id_dict = dict()
    edge_data = []
    for index, row in df_data.iterrows():
        if 'u'+str(row['userID']) not in node_id_dict.keys():
            node_id_dict['u'+str(row['userID'])] = node_ids
            node_ids = node_ids + 1
        if 'a'+str(row['artistID']) not in node_id_dict.keys():
            node_id_dict['a' + str(row['artistID'])] = node_ids
            node_ids = node_ids + 1
        edge_data.append([node_id_dict['u'+str(row['userID'])], node_id_dict['a' + str(row['artistID'])]])
        edge_data.append([node_id_dict['a' + str(row['artistID'])], node_id_dict['u' + str(row['userID'])]])

    return Data(edge_index=torch.LongTensor(edge_data).t().contiguous(), num_nodes=len(node_id_dict.keys()))


