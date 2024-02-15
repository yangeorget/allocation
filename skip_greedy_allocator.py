import random

import numpy as np

from allocator import Allocator


class SkipGreedyAllocator(Allocator):
    def solve(self):
        costs = self.scores * self.generosities * self.bools
        user_offers = np.ones(self.user_nb) * self.offer_max_nb
        offer_budgets = self.budgets.copy()
        for idx in np.nditer(np.argsort(-costs, axis=None)):
            user_idx, offer_idx = np.unravel_index(idx, costs.shape)
            cost = costs[user_idx, offer_idx]
            if user_offers[user_idx] > 0 and 0 < cost <= offer_budgets[offer_idx] and random.random() > 0/30:
                offer_budgets[offer_idx] -= cost
                user_offers[user_idx] -= 1
                offer_indices = np.where((self.families == self.families[offer_idx]) & (costs[user_idx] > 0))[0]
                costs[user_idx, offer_indices] = 0.0
                costs[user_idx, offer_idx] = cost
            else:
                costs[user_idx, offer_idx] = 0.0
        return costs > 0, costs
