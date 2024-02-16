import random

import numpy as np

from allocator import Allocator


class HeuristicAllocator(Allocator):
    def solve(self, init_costs, best_result):
        allocations = init_costs > 0
        user_offers = np.ones((self.user_nb, 1)) * self.offer_max_nb
        offer_budgets = self.budgets.copy()
        # offer_family_size = [np.count_nonzero(self.families == self.families[idx]) for idx in range(self.offer_nb)]
        # offer_family_weight = (offer_family_size - np.min(offer_family_size) + 1) ** 1 / 4
        for idx in np.nditer(
            np.argsort(-init_costs * (1 + np.random.rand(self.user_nb, self.offer_nb) / 3), axis=None)
        ):
            user_idx, offer_idx = np.unravel_index(idx, init_costs.shape)
            if allocations[user_idx, offer_idx]:
                if init_costs[user_idx, offer_idx] > offer_budgets[offer_idx]:
                    allocations[user_idx, offer_idx] = False
                elif user_offers[user_idx] == 0:
                    allocations[user_idx] = False
                else:
                    offer_budgets[offer_idx] -= init_costs[user_idx, offer_idx]
                    user_offers[user_idx] -= 1
                    if user_offers[user_idx] == 0:
                        allocations[user_idx] = False
                    else:
                        offer_indices = np.where(self.families == self.families[offer_idx])[0]
                        allocations[user_idx, offer_indices] = False
                    allocations[user_idx, offer_idx] = True
        return allocations, init_costs * allocations
