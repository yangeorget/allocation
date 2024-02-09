import numpy as np

from greedy_allocator import GreedyAllocator


class RandomAllocator(GreedyAllocator):
    def solve(self):
        costs = self.scores * self.generosities * self.bools
        user_allocations = self.compute_user_allocations(costs)
        self.update_costs(costs, user_allocations)
        families_allocations = self.compute_family_allocations(costs)
        self.update_costs(costs, families_allocations)
        offer_allocations = self.compute_offer_allocations(costs)
        self.update_costs(costs, offer_allocations)
        return costs > 0, costs

    def compute_user_allocations(self, costs):
        offer_indices = np.arange(self.offer_nb)
        allocations = []
        for user_idx in range(self.user_nb):
            user_costs = costs[user_idx]
            user_costs_sum = np.sum(user_costs)
            if user_costs_sum > 0:
                probabilities = user_costs / np.sum(user_costs)
                choice_nb = min(self.offer_max_nb, np.count_nonzero(probabilities))
                choices = np.isin(
                    offer_indices, np.random.choice(offer_indices, choice_nb, replace=False, p=probabilities)
                )
            else:
                choices = np.zeros(self.offer_nb, dtype=bool)
            allocations.append(choices)
        return np.array(allocations)
