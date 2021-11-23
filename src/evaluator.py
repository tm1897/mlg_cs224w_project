import logging
from collections import defaultdict

import pandas as pd

from src.data_preprocessing import TrainTestGenerator

logger = logging.getLogger(__name__)


class HitRate:
    def __init__(self, n=None, ignore_cold=False):
        self.hits = defaultdict(int)
        self.cases = 0
        self.n = n if n is not None else [5, 10, 25, 50, 500]
        self.ignore_cold = ignore_cold

    def add_case(self, train, test, recommended):
        # TODO Instead of computing HR directly, compute "normalized" rank and use it for HR, MRR, ...
        train = train.copy()
        test = test.copy()
        recommended = recommended.copy()

        # Remove train items from recommended
        # Items seen in train will not be recommended to users
        for item in train:
            recommended.remove(item)

        # Iterate through all test items
        for item in test:
            for n in self.n:
                # Increment hits[n] by one, if item is in the first n items
                self.hits[n] += item in recommended[:n]
            try:
                # Remove item from recommended list
                # Item will not be recommended to users again, if they already interacted with it
                recommended.remove(item)
            except ValueError:
                # Item was not seen in train set
                if self.ignore_cold:
                    # If ignore cold, reduce total number of cases
                    self.cases -= 1
        self.cases += len(test)

    def get_hit_rate(self, n: int):
        if n not in self.n:
            raise KeyError(f"{n} was not specified in constructor")
        return self.hits[n] / self.cases

    def __str__(self):
        return "\n".join([f"{n}: {self.get_hit_rate(n)}" for n in self.n])


class Evaluator:
    def __init__(self, model_init, train_test_generator: TrainTestGenerator):
        self.model_init = model_init
        self.train_test_generator = train_test_generator

        self.hit_rate = {}

    def evaluate(self):
        for test_year, train, test in self.train_test_generator.forward_chaining():
            logging.info(f"Test year: {test_year}")
            model = self.model_init()
            model.fit(train)
            hit_rate = HitRate()
            n_items = len(train["artistID"].unique())

            for user_id in test["userID"].unique():
                user_train = list(train.loc[train["userID"] == user_id, "artistID"])
                user_test = list(test[test["userID"] == user_id].sort_values("timestamp")["artistID"])
                recommended = list(model.recommend(user_id, n_items))

                hit_rate.add_case(user_train, user_test, recommended)

            self.hit_rate[test_year] = hit_rate

    def get_results(self):
        hit_rates = {}

        for year, hit_rate in self.hit_rate.items():
            hit_rates[year] = [hit_rate.cases] + [hit / hit_rate.cases for hit in hit_rate.hits.values()]

        results = pd.DataFrame(hit_rates, index=["cases"] + hit_rate.n).T
        results["cases"] = results["cases"].astype(int)

        return results
