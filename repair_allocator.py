import random

import numpy as np

from allocator import Allocator
from hidden.allocation.heuristic_allocator import HeuristicAllocator


class RepairAllocator(HeuristicAllocator):
    # TODO : fix this, this is broken, exception for small pb
    # TODO: we probably just want to swap two random cells
    def solve(self, init_costs, best_result):
        best_allocations = best_result["allocations"]
        best_costs = best_result["costs"]
        if best_allocations is None:
            return super().solve(init_costs, best_result)
        offer_budgets = self.budgets - np.sum(best_costs, axis=0)
        for offer_idx in np.nditer(np.where(offer_budgets > 0)):
            for user_idx1 in range(self.user_nb - 1):
                if best_allocations[user_idx1, offer_idx]:
                    for user_idx2 in range(user_idx1 + 1, self.user_nb):
                        if self.bools[user_idx2, offer_idx] and np.logical_not(best_allocations[user_idx2, offer_idx]):
                            if (
                                init_costs[user_idx2, offer_idx] - init_costs[user_idx1, offer_idx]
                                > offer_budgets[offer_idx]
                            ):
                                best_allocations[user_idx2, offer_idx] = True
                                best_allocations[user_idx1, offer_idx] = False
                if np.logical_not(best_allocations[user_idx1, offer_idx]):
                    for user_idx2 in range(user_idx1 + 1, self.user_nb):
                        if self.bools[user_idx2, offer_idx] and best_allocations[user_idx2, offer_idx]:
                            if (
                                init_costs[user_idx1, offer_idx] - init_costs[user_idx2, offer_idx]
                                > offer_budgets[offer_idx]
                            ):
                                best_allocations[user_idx2, offer_idx] = False
                                best_allocations[user_idx1, offer_idx] = True
        return best_allocations, best_allocations * init_costs
