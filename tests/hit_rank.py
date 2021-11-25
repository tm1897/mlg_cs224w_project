import logging
import random
import sys
import unittest

import pandas as pd

from src.evaluator import HitRate

logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s - %(message)s'
)


def get_auto_recommendations(N, seed, train_size, test_size):
    random.seed(seed)
    items = list(range(N))

    recommended = items.copy()
    random.shuffle(recommended)
    # recommended = set(recommended)

    train = random.sample(items.copy(), train_size)
    train = set(train)

    test = random.sample(list(set(items) - train).copy(), test_size)
    test = set(test)

    return recommended, train, test


class HitRateTests(unittest.TestCase):
    def test_1(self):
        hr = HitRate(n=[1, 2, 3, 4])

        train = pd.Series([1, 3, 8, 9])
        test = pd.DataFrame({
            "artistID": [2, 4, 5],
            "timestamp": [1, 5, 3]
        })
        recommended = pd.Series([1, 2, 3, 6, 7, 8, 9, 4, 5])
        train, test, recommended = list(train), list(test.sort_values("timestamp")["artistID"]), list(recommended)
        hr.add_case(
            train=train,
            test=test,
            recommended=recommended,
        )

        """
        2, 5, 4
        [2, 6, 7, 4, 5]

        2  [2, 6, 7, 4, 5]
        5  [6, 7, 4, 5]
        4  [6, 7, 4]
        """
        self.assertEqual(1/3, hr.get_hit_rate(1), f"hr@{1}")
        self.assertEqual(1/3, hr.get_hit_rate(2), f"hr@{2}")
        self.assertEqual(2/3, hr.get_hit_rate(3), f"hr@{3}")
        self.assertEqual(1, hr.get_hit_rate(4), f"hr@{4}")

    def test_unknown(self):
        train = [4, 5]
        test = [1, 2, 6]
        recommended = [1, 3, 2, 4, 5]
        """
        [1, 3, 2]
        1 [1, 3, 2]
        2 [3, 2]
        6 [3]
        """

        hr = HitRate([1, 2, 3, 4, 5])
        hr.add_case(train, test, recommended)
        self.assertEqual(1/3, hr.get_hit_rate(1), f"hr@{1}")
        self.assertEqual(2/3, hr.get_hit_rate(2), f"hr@{2}")
        self.assertEqual(2/3, hr.get_hit_rate(3), f"hr@{3}")
        self.assertEqual(2/3, hr.get_hit_rate(4), f"hr@{4}")

    def test_auto(self):
        seed = 1

        n = [1, 2, 3, 4, 5]

        hr = HitRate(n=n)
        recommended, train, test = get_auto_recommendations(8, seed, 3, 2)

        # train, test, recommended = list(train), list(test), list(recommended)

        hr.add_case(
            train,
            test,
            recommended
        )

        for n in n:
            logging.debug(n)
            recommended, train, test = get_auto_recommendations(8, seed, 3, 2)
            train, test, recommended = list(train), list(test), list(recommended)

            # recommended - train
            for item in train:
                recommended.remove(item)
            logging.debug(recommended)
            logging.debug(test)

            # Hit rate
            hits = 0
            cases = 0
            for item in test:
                hits += item in recommended[:n]
                cases += 1
                recommended.remove(item)

            self.assertEqual(hits/cases, hr.get_hit_rate(n), msg=f"HR@{n}")
            logging.debug(hits/cases)
            logging.debug("")


if __name__ == '__main__':
    unittest.main()
