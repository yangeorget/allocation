import random

import numpy as np

from allocator import Allocator
from hidden.allocation.heuristic_allocator import HeuristicAllocator


class RepairAllocator(HeuristicAllocator):
    def solve(self, init_costs, best_result):
        best_allocations = best_result["allocations"]
        best_costs = best_result["costs"]
        if best_allocations is None:
            return super().solve(init_costs, best_result)
        for user_idx in range(self.user_nb):
            offers = np.where(best_allocations[user_idx])[0]
            if len(offers):
                for offer_idx in np.nditer(offers):
                    candidates = np.where(
                        (
                            self.bools[user_idx]
                            & np.logical_not(best_allocations[user_idx])
                            & (init_costs[user_idx] > best_costs[user_idx][offer_idx])
                            & (np.sum(best_costs, axis=0) + init_costs[user_idx] <= self.budgets)
                            & (self.families == self.families[offer_idx])
                        )
                    )[0]
                    if len(candidates):
                        new_offer_idx = np.random.choice(candidates, 1)[0]
                        best_allocations[user_idx, new_offer_idx] = True
                        best_costs[user_idx, new_offer_idx] = init_costs[user_idx][new_offer_idx]
                        best_allocations[user_idx, offer_idx] = False
                        best_costs[user_idx, offer_idx] = 0
        return best_allocations, best_costs
