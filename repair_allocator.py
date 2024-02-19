import numpy as np

from hidden.allocation.heuristic_allocator import HeuristicAllocator


class RepairAllocator(HeuristicAllocator):
    def solve(self, init_costs, best_result):
        _, best_costs = super().solve(init_costs, best_result)
        if best_costs is None:
            return None, None  # cannot repair
        offer_budgets = self.budgets - np.sum(best_costs, axis=0)
        for user_idx in range(self.user_nb):
            for offer_idx in range(self.offer_nb):
                if best_costs[user_idx][offer_idx] > 0:
                    best_offer_idx = None
                    best_cost = 0.0
                    # a change in a family is easy to make thus we want to make the best choice in each family
                    for other_offer_idx in np.where(
                        (self.families == self.families[offer_idx])
                        & (init_costs[user_idx] > 0)
                        & (best_costs[user_idx] == 0)
                        & (offer_budgets >= init_costs[user_idx])
                        & (init_costs[user_idx] > best_costs[user_idx][offer_idx])
                    )[0]:
                        if init_costs[user_idx][other_offer_idx] > best_cost:
                            best_cost = init_costs[user_idx][other_offer_idx]
                            best_offer_idx = other_offer_idx
                    if best_offer_idx is not None:
                        best_costs[user_idx][offer_idx] = 0
                        best_costs[user_idx][best_offer_idx] = init_costs[user_idx][best_offer_idx]
                        offer_budgets[best_offer_idx] -= init_costs[user_idx][best_offer_idx]
        return None, best_costs
