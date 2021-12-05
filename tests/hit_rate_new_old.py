import unittest
from collections import defaultdict

import numpy as np
import pandas as pd

from src.evaluator import rank_items, hit_rate_at_k, mean_reciprocal_rank


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


class CompareTest(unittest.TestCase):
    def test_random(self):
        np.random.seed(1)
        for N in [500, 1000, 5000, 10000]:
            items = np.arange(N)
            np.random.shuffle(items)

            hit_rate_old = HitRate()

            train_test = np.random.choice(items, N//50)
            train = train_test[:len(train_test)//2]
            test = train_test[len(train_test)//2:]
            test_unknown_items = np.arange(-1, -11, -1)

            train = list(train)
            test = list(test) + list(test_unknown_items)
            items = list(items)

            hit_rate_old.add_case(train, test, items)

            ranks = rank_items(train, test, items)
            print(mean_reciprocal_rank(ranks))

            for n in hit_rate_old.n:
                self.assertEqual(
                    hit_rate_old.get_hit_rate(n),
                    hit_rate_at_k(ranks, n)
                )


if __name__ == '__main__':
    unittest.main()
