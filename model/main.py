from functools import partial

import get_pyg_data
from model import LightGCNStack
import torch

from src.data_preprocessing import TrainTestGenerator
from src.evaluator import Evaluator
from train_test import train, test
from torch_geometric.utils import train_test_split_edges
import time

class objectview(object):
    def __init__(self, *args, **kwargs):
        d = dict(*args, **kwargs)
        self.__dict__ = d

if __name__=='__main__':
    # best_val_perf = test_perf = 0
    # data = get_pyg_data.load_data()
    #data = train_test_split_edges(data)

    args = {'model_type': 'LightGCN', 'num_layers': 3, 'batch_size': 32, 'hidden_dim': 64,
         'dropout': 0, 'epochs': 1000, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3,
         'lr': 0.1}

    evaluate(args)

    # args = objectview(args)
    # model, data = LightGCNStack(latent_dim=64, dataset=data, args=args).to('cuda'), data.to('cuda')
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    # for epoch in range(1, 1001):
    #     start = time.time()
    #     train_loss = train(model, data, optimizer)
    #     val_perf, tmp_test_perf = test(model, (data, data))
    #     if val_perf > best_val_perf:
    #         best_val_perf = val_perf
    #         test_perf = tmp_test_perf
    #     log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}, Elapsed time: {:.2f}'
    #     print(log.format(epoch, train_loss, best_val_perf, test_perf, time.time()-start))