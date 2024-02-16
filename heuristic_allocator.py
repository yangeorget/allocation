import random

import numpy as np

from allocator import Allocator


class HeuristicAllocator(Allocator):
    def solve(self, init_costs, best_result):
        costs = init_costs.copy()
        user_offers = np.ones(self.user_nb) * self.offer_max_nb
        offer_budgets = self.budgets.copy()
        # offer_family_size = [np.count_nonzero(self.families == self.families[idx]) for idx in range(self.offer_nb)]
        # offer_family_weight = (offer_family_size - np.min(offer_family_size) + 1) ** 1 / 4
        for idx in np.nditer(
            np.argsort(
                -costs * (1 + np.random.rand(self.user_nb, self.offer_nb) / 3),
                axis=None,
            )
        ):
            user_idx, offer_idx = np.unravel_index(idx, costs.shape)
            cost = costs[user_idx, offer_idx]
            if cost > 0:
                if cost > offer_budgets[offer_idx]:
                    costs[user_idx, offer_idx] = 0.0
                elif user_offers[user_idx] == 0:
                    costs[user_idx, :] = 0.0
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
