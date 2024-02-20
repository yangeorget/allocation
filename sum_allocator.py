import numpy as np

from allocator import Allocator


class SumAllocator(Allocator):
    def solve(self, init_costs, family_costs, best_result):
        costs = init_costs.copy()
        max_costs = np.sum(np.partition(family_costs, -self.offer_max_nb, axis=1)[:, -self.offer_max_nb :], axis=1)
        max_cost = np.sum(max_costs)
        allocations = np.zeros((self.user_nb, self.offer_nb), dtype=bool)
        user_offers = np.ones((self.user_nb, 1)) * self.offer_max_nb
        offer_budgets = self.budgets.copy()
        index_weights = -init_costs * (
            1 + np.random.rand(self.user_nb, self.offer_nb) / 3
        )  # introducing some randomness
        for idx in np.nditer(np.argsort(index_weights, axis=None)[: np.count_nonzero(index_weights)]):
            user_idx, offer_idx = np.unravel_index(idx, init_costs.shape)
            if costs[user_idx, offer_idx] > 0.0:  # most are == 0.0
                family_idx = self.families[offer_idx]
                allocations[user_idx, offer_idx] = True  # we keep track of the allocations
                cost = costs[user_idx, offer_idx]
                family_costs[user_idx, family_idx] = cost
                # handle column
                offer_budgets[offer_idx] -= cost
                user_mask = np.logical_not(allocations[:, offer_idx]) & (costs[:, offer_idx] > offer_budgets[offer_idx])
                costs[user_mask, offer_idx] = 0.0
                for uidx in np.nditer(np.where(user_mask)[0], ["zerosize_ok"]):
                    max_cost -= max_costs[uidx]
                    family_costs[uidx, family_idx] = np.max(costs[uidx, self.families == family_idx])
                # handle line
                user_offers[user_idx] -= 1
                max_cost -= max_costs[user_idx]
                if user_offers[user_idx] == 0:
                    offer_mask = np.logical_not(allocations[user_idx])
                    costs[user_idx, offer_mask] = 0.0
                    for oidx in np.nditer(np.where(offer_mask)[0], ["zerosize_ok"]):
                        family_costs[user_idx, oidx] = np.max(costs[user_idx, self.families == self.families[oidx]])
                else:
                    costs[user_idx][self.families == family_idx] = 0.0
                    costs[user_idx, offer_idx] = cost
                max_cost += np.sum(np.partition(family_costs[user_idx], -self.offer_max_nb)[-self.offer_max_nb :])
                for uidx in np.nditer(np.where(user_mask)[0], ["zerosize_ok"]):
                    max_cost += np.sum(np.partition(family_costs[uidx], -self.offer_max_nb)[-self.offer_max_nb :])
                if max_cost <= best_result["evaluation"]:
                    return None, None
        return None, costs
