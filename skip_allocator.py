import random

import numpy as np

from allocator import Allocator


class SkipAllocator(Allocator):
    def solve(self, best_evaluation):
        costs = self.scores * self.generosities * self.bools
        user_offers = np.ones(self.user_nb) * self.offer_max_nb
        offer_budgets = self.budgets.copy()
        for idx in np.nditer(np.argsort(-costs, axis=None)):
            user_idx, offer_idx = np.unravel_index(idx, costs.shape)
            cost = costs[user_idx, offer_idx]
            if cost > 0:
                if cost > offer_budgets[offer_idx]:
                    costs[user_idx, offer_idx] = 0.0
                elif user_offers[user_idx] == 0:
                    costs[user_idx, :] = 0.0
                elif random.random() < 1 / 1000:
                    costs[user_idx, offer_idx] = 0.0
                else:
                    offer_budgets[offer_idx] -= cost
                    user_offers[user_idx] -= 1
                    if user_offers[user_idx] == 0:
                        costs[user_idx, :] = 0.0
                    else:
                        offer_indices = np.where(self.families == self.families[offer_idx])[0]
                        costs[user_idx, offer_indices] = 0.0
                    costs[user_idx, offer_idx] = cost
        return costs > 0, costs
