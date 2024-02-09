import numpy as np

from allocator import Allocator


class GreedyAllocator(Allocator):
    def solve(self):
        costs = self.scores * self.generosities * self.bools
        user_allocations = self.compute_user_allocations(costs)
        self.update_costs(costs, user_allocations)
        offer_allocations = self.compute_offer_allocations(costs)
        self.update_costs(costs, offer_allocations)
        families_allocations = self.compute_family_allocations(costs)
        self.update_costs(costs, families_allocations)
        return costs > 0, costs

    def compute_user_allocations(self, costs):
        top = 1
        bottom = self.offer_max_nb + top - 1
        partition = np.partition(costs, [-bottom, -top], axis=1)
        return np.logical_and(costs >= partition[:, [-bottom]], costs <= partition[:, [-top]])

    def compute_offer_allocations(self, costs):
        return np.cumsum(costs, axis=0) <= self.budgets

    def compute_family_allocations(self, costs):
        family_allocations = np.ones((self.user_nb, self.offer_nb), dtype=bool)
        for family_idx in range(self.family_nb):
            family = self.families == family_idx
            allocations = np.cumsum(np.logical_and(costs > 0, family), axis=1) <= 1
            np.logical_and(family_allocations, allocations, out=family_allocations)
        return family_allocations
