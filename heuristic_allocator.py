import numpy as np

from allocator import Allocator


class HeuristicAllocator(Allocator):
    def solve(self, init_costs, best_result):
        costs = init_costs.copy()
        # max_cost = np.sum(init_costs)  # we keep track of the best achievable cost
        allocations = np.zeros((self.user_nb, self.offer_nb), dtype=bool)
        user_offers = np.ones((self.user_nb, 1)) * self.offer_max_nb
        offer_budgets = self.budgets.copy()
        index_weights = -init_costs * (
            1 + np.random.rand(self.user_nb, self.offer_nb) / 3
        )  # introducing some randomness
        for idx in np.nditer(np.argsort(index_weights, axis=None)[: np.count_nonzero(index_weights)]):
            user_idx, offer_idx = np.unravel_index(idx, init_costs.shape)
            if costs[user_idx, offer_idx] > 0.0:  # most are == 0.0
                allocations[user_idx, offer_idx] = True  # we keep track of the allocations
                # handle line
                offer_budgets[offer_idx] -= costs[user_idx, offer_idx]
                mask = np.logical_not(allocations[:, offer_idx]) & (costs[:, offer_idx] > offer_budgets[offer_idx])
                # max_cost -= np.sum(costs[mask, offer_idx])
                # if max_cost <= best_result["evaluation"]:
                #    return None, None
                costs[mask, offer_idx] = 0.0
                # handle column
                user_offers[user_idx] -= 1
                mask = np.logical_not(allocations[user_idx])  # will be used when user_offers[user_idx] == 0
                if user_offers[user_idx] > 0:
                    mask &= self.families == self.families[offer_idx]
                # max_cost -= np.sum(costs[user_idx][mask])
                # if max_cost <= best_result["evaluation"]:
                #    return None, None
                costs[user_idx][mask] = 0.0
        return None, costs
