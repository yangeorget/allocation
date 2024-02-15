import random

import numpy as np

from allocator import Allocator


class VeryGreedyAllocator(Allocator):
    def solve(self):
        costs = self.scores * self.generosities * self.bools
        user_offers = np.ones(self.user_nb) * self.offer_max_nb
        offer_budgets = self.budgets.copy()
        allocations = np.zeros((self.user_nb, self.offer_nb), dtype=bool)
        indices = np.argsort(-costs, axis=None)
        for idx in np.nditer(indices):
            user_idx, offer_idx = np.unravel_index(idx, costs.shape)
            cost = costs[user_idx, offer_idx]
            if 0 < cost <= offer_budgets[offer_idx] and user_offers[user_idx] > 0:
                offer_indices = np.where((self.families == self.families[offer_idx]) & (costs[user_idx] > 0))[0]
                costs[user_idx, offer_indices] = 0.0
                offer_budgets[offer_idx] -= cost
                user_offers[user_idx] -= 1
                allocations[user_idx, offer_idx] = True
            else:
                costs[user_idx, offer_idx] = 0.0
        return allocations, self.scores * self.generosities * allocations
