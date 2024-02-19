import numpy as np

from allocator import Allocator


class HeuristicAllocator(Allocator):
    def solve(self, init_costs, best_result):
        max_cost = np.sum(init_costs)
        user_offers = np.ones((self.user_nb, 1)) * self.offer_max_nb
        offer_budgets = self.budgets.copy()
        index_weights = -init_costs * (1 + np.random.rand(self.user_nb, self.offer_nb) / offer_budgets)
        costs = init_costs.copy()
        for idx in np.nditer(np.argsort(index_weights, axis=None)[: np.count_nonzero(index_weights)]):
            user_idx, offer_idx = np.unravel_index(idx, init_costs.shape)
            if costs[user_idx, offer_idx] > 0.0:
                if costs[user_idx, offer_idx] > offer_budgets[offer_idx]:
                    max_cost -= costs[user_idx, offer_idx]
                    if max_cost <= best_result["evaluation"]:
                        return None, None
                    costs[user_idx, offer_idx] = 0.0
                else:  # offer_budgets[offer_idx] > 0 otherwise costs[user_idx, offer_idx] == 0
                    offer_budgets[offer_idx] -= costs[user_idx, offer_idx]
                    user_offers[user_idx] -= 1
                    if user_offers[user_idx] == 0:
                        max_cost -= np.sum(costs[user_idx]) - costs[user_idx, offer_idx]
                        if max_cost <= best_result["evaluation"]:
                            return None, None
                        cost = costs[user_idx, offer_idx]
                        costs[user_idx] = 0
                        costs[user_idx, offer_idx] = cost
                    else:
                        family_mask = self.families == self.families[offer_idx]
                        max_cost -= np.sum(costs[user_idx][family_mask]) - costs[user_idx, offer_idx]
                        if max_cost <= best_result["evaluation"]:
                            return None, None
                        cost = costs[user_idx, offer_idx]
                        costs[user_idx, family_mask] = 0
                        costs[user_idx, offer_idx] = cost
        return None, costs
