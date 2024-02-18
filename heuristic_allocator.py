import numpy as np

from allocator import Allocator


class HeuristicAllocator(Allocator):
    def solve(self, init_costs, best_result):
        costs = init_costs.copy()
        best_cost = np.sum(costs)
        user_offers = np.ones((self.user_nb, 1)) * self.offer_max_nb
        offer_budgets = self.budgets.copy()
        for idx in np.nditer(
            np.argsort(-init_costs * (1 + np.random.rand(self.user_nb, self.offer_nb) / 4), axis=None)
        ):
            user_idx, offer_idx = np.unravel_index(idx, init_costs.shape)
            if costs[user_idx, offer_idx] > 0.0:
                if costs[user_idx, offer_idx] > offer_budgets[offer_idx]:
                    best_cost -= costs[user_idx, offer_idx]
                    if best_cost <= best_result["evaluation"]:
                        return None, None
                    costs[user_idx, offer_idx] = 0.0
                else:  # offer_budgets[offer_idx] > 0 otherwise costs[user_idx, offer_idx] == 0
                    offer_budgets[offer_idx] -= costs[user_idx, offer_idx]
                    user_offers[user_idx] -= 1
                    if user_offers[user_idx] == 0:
                        best_cost -= np.sum(costs[user_idx]) - costs[user_idx, offer_idx]
                        if best_cost <= best_result["evaluation"]:
                            return None, None
                        cost = costs[user_idx, offer_idx]
                        costs[user_idx] = 0
                        costs[user_idx, offer_idx] = cost
                    else:
                        offer_indices = np.where(self.families == self.families[offer_idx])[0]
                        best_cost -= np.sum(costs[user_idx][offer_indices]) - costs[user_idx, offer_idx]
                        if best_cost <= best_result["evaluation"]:
                            return None, None
                        cost = costs[user_idx, offer_idx]
                        costs[user_idx, offer_indices] = 0
                        costs[user_idx, offer_idx] = cost
        return costs > 0, costs
