from functools import partial

import get_pyg_data
from lightgcn_model import LightGCNStack
import torch

from src.data_preprocessing import TrainTestGenerator
from src.evaluator import Evaluator
from train_test import train, test
from torch_geometric.utils import train_test_split_edges
import time

import pandas as pd


class objectview(object):
    def __init__(self, *args, **kwargs):
        d = dict(*args, **kwargs)
        self.__dict__ = d


# Wrapper for evaluation
class LightGCN_recommender:
    def __init__(self, args):
        self.args = objectview(args)
        self.model = LightGCNStack(latent_dim=64, args=self.args).to('cuda')
        self.a_rev_dict = None
        self.u_rev_dict = None
        self.a_dict = None
        self.u_dict = None

    def fit(self, data: pd.DataFrame):
        # Default rankings when userID is not in training set
        self.default_recommendation = data["artistID"].value_counts().index.tolist()

        # LightGCN
        data, self.u_rev_dict, self.a_rev_dict, self.u_dict, self.a_dict = get_pyg_data.load_bipartitedata(data)
        data = data.to("cuda")
        self.model.init_data(data)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.001)

        best_val_perf = test_perf = 0

        for epoch in range(1, self.args.epochs+1):
            start = time.time()
            train_loss = train(self.model, data, self.optimizer)
            val_perf, tmp_test_perf = test(self.model, (data, data))
            if val_perf > best_val_perf:
                best_val_perf = val_perf
                test_perf = tmp_test_perf
            log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}, Elapsed time: {:.2f}'
            print(log.format(epoch, train_loss, best_val_perf, test_perf, time.time()-start))

    def recommend(self, user_id, n):
        try:
            recommendations = self.model.topN(self.u_dict[str(user_id)], n=n)
        except KeyError:

            recommendations = self.default_recommendation
        else:
            recommendations = recommendations.indices.cpu().tolist()
            recommendations = list(map(lambda x: self.a_rev_dict[x], recommendations))
        return recommendations


def evaluate(args):
    data_dir = "../data/"
    data_generator = TrainTestGenerator(data_dir)

    evaluator = Evaluator(partial(LightGCN_recommender, args), data_generator)
    evaluator.evaluate()

    evaluator.save_results('../results/lightgcn.csv', '../results/lightgcn_time.csv')
    print('Recall:')
    print(evaluator.get_recalls())
    print('MRR:')
    print(evaluator.get_mrr())


if __name__=='__main__':
    args = {'model_type': 'LightGCN', 'num_layers': 3, 'batch_size': 32, 'hidden_dim': 32,
         'dropout': 0, 'epochs': 1000, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3,
         'lr': 0.1, 'lambda_reg': 1e-4}

    evaluate(args)
