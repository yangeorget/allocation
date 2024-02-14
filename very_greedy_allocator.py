import numpy as np

from allocator import Allocator


class VeryGreedyAllocator(Allocator):
    def solve(self):
        costs = self.scores * self.generosities * self.bools
        user_offers = np.ones(self.user_nb) * self.offer_max_nb
        offer_budgets = self.budgets.copy()
        allocations = np.zeros((self.user_nb, self.offer_nb), dtype=bool)
        nonzero_nb = np.count_nonzero(costs)
        while nonzero_nb:
            idx = np.argmax(costs)  # no need to reshape
            user_idx = idx // self.offer_nb
            offer_idx = idx % self.offer_nb
            cost = costs[user_idx, offer_idx]
            if user_offers[user_idx] > 0 and offer_budgets[offer_idx] >= cost:
                offer_indices = np.where((self.families == self.families[offer_idx]) & (costs[user_idx] > 0))[0]
                costs[user_idx, offer_indices] = 0.0
                nonzero_nb -= offer_indices.shape[0]
                offer_budgets[offer_idx] -= cost
                user_offers[user_idx] -= 1
                allocations[user_idx, offer_idx] = True
            else:
                costs[user_idx, offer_idx] = 0.0
                nonzero_nb -= 1
        return allocations, self.scores * self.generosities * allocations
