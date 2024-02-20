import numpy as np

from hidden.allocation.heuristic_allocator import HeuristicAllocator


class RepairAllocator(HeuristicAllocator):
    def solve(self, init_costs, best_result):
        best_allocations = np.loadtxt(f"allocations/stephkyl/allocations.csv", delimiter=",", dtype=bool)
        best_costs = init_costs * best_allocations
        offer_budgets = self.budgets - np.sum(best_costs, axis=0)
        for idx in np.where(best_costs > 0)[0]:
            user_idx, offer_idx = np.unravel_index(idx, init_costs.shape)
            # a change in a family is easy to make thus we want to make the best choice in each family
            candidate_costs = init_costs[user_idx][
                (init_costs[user_idx] > 0.0)
                & (best_costs[user_idx] == 0)
                & (offer_budgets >= init_costs[user_idx])
                & (init_costs[user_idx] > best_costs[user_idx][offer_idx])
                & (self.families == self.families[offer_idx])
            ]
            if len(candidate_costs) > 0:
                best_offer_idx = np.argmax(candidate_costs)
                best_costs[user_idx][offer_idx] = 0
                best_costs[user_idx][best_offer_idx] = init_costs[user_idx][best_offer_idx]
                offer_budgets[best_offer_idx] -= init_costs[user_idx][best_offer_idx]
        return None, best_costs
