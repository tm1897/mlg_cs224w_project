import logging

import pandas as pd

from src.data_preprocessing import TrainTestGenerator

logger = logging.getLogger(__name__)


def rank_items(train, test, recommended):
    train = train.copy()
    test = test.copy()
    recommended = recommended.copy()

    # Remove train items from recommended
    # Items seen in train will not be recommended to users
    for item in train:
        recommended.remove(item)

    ranks = []

    # Iterate through all test items
    for item in test:
        try:
            rank = recommended.index(item) + 1  # Indices start with 0, ranks with 1
        except ValueError:
            # Item was not seen in train set
            rank = None
        else:
            recommended.remove(item)
        finally:
            ranks.append(rank)
    return ranks


def hit_rate_at_k(ranks, k):
    ranks = pd.Series(ranks)
    hits_at_k = ranks <= k
    hits_at_k = hits_at_k.sum()
    hit_rate = hits_at_k / len(ranks)

    return hit_rate


def mean_reciprocal_rank(ranks):
    # TODO How to deal with nans - items not in train set
    ranks = pd.Series(ranks)
    mrr = (1 / ranks).mean()

    return mrr


class Evaluator:
    def __init__(self, model_init, train_test_generator: TrainTestGenerator):
        self.model_init = model_init
        self.train_test_generator = train_test_generator

        self.results = pd.DataFrame()

    def evaluate(self):
        results = []
        for test_year, train, test in self.train_test_generator.forward_chaining():
            logging.info(f"Test year: {test_year}")
            model = self.model_init()
            model.fit(train)
            n_items = len(train["artistID"].unique())

            for user_id in test["userID"].unique():
                user_train = list(train.loc[train["userID"] == user_id, "artistID"])
                user_test = list(test[test["userID"] == user_id].sort_values("timestamp")["artistID"])
                recommended = list(model.recommend(user_id, n_items))
                ranks = rank_items(user_train, user_test, recommended)
                results_user = pd.DataFrame({
                    "user": user_id,
                    "item": user_test,
                    "rank": ranks,
                    "test_year": test_year
                })
                results.append(results_user)
        self.results = pd.concat(results).reset_index(drop=True)

    def get_hit_rates(self, Ks: list = None):
        if Ks is None:
            Ks = [5, 10, 25, 50, 500]

        results = self.results
        results_df = []
        years = sorted(results["test_year"].unique())
        for year in years:
            results_year = results[results["test_year"] == year]
            cases = len(results_year)
            row = [cases]
            for k in Ks:
                hr = hit_rate_at_k(results_year["rank"], k)
                row.append(hr)
            results_df.append(row)
        results_df = pd.DataFrame(results_df, columns=["cases"] + Ks, index=years)

        return results_df

    def get_mrr(self):
        results = self.results
        results_df = []
        years = sorted(results["test_year"].unique())
        for year in years:
            results_year = results[results["test_year"] == year]
            ranks = results_year["rank"].dropna()
            cases = len(ranks)
            mrr = mean_reciprocal_rank(ranks)
            row = [cases, mrr]
            results_df.append(row)
        results_df = pd.DataFrame(results_df, columns=["cases", "mrr"], index=years)

        return results_df

    def save_results(self, file_path):
        """

        :param file_path: CSV file
        :return:
        """
        self.results.to_csv(file_path, index=False)
