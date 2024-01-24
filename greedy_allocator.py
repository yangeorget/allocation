import numpy as np

from allocator import Allocator


class GreedyAllocator(Allocator):
    def __init__(self, max_offer_nb, offer_max_user_nb, ok, scores, exploration_factor=0):
        self.exploration_factor = exploration_factor
        super().__init__(max_offer_nb, offer_max_user_nb, ok, scores)

    def name(self):
        return f"GreedyAllocator(exploration_factor={self.exploration_factor})"

    def solve(self):
        permutation = np.random.permutation(self.user_nb)
        shuffled_allocations, shuffled_scores = self.find_solution(permutation)
        inv_permutation = GreedyAllocator.inverse_permutation(permutation)
        return shuffled_allocations[inv_permutation], shuffled_scores[inv_permutation]

    @staticmethod
    def inverse_permutation(a):
        b = np.arange(a.shape[0])
        b[a] = b.copy()
        return b

    def find_solution(self, permutation):
        allocations = self.init_allocations[permutation]
        scores = self.init_scores[permutation]
        Allocator.update_scores(scores, allocations)
        user_allocations = Allocator.compute_user_allocations(scores, self.max_offer_nb, self.exploration_factor)
        Allocator.update_allocations(allocations, user_allocations)
        Allocator.update_scores(scores, allocations)
        offer_allocations = Allocator.compute_offer_allocations(scores, self.offer_max_user_nb)
        Allocator.update_allocations(allocations, offer_allocations)
        Allocator.update_scores(scores, allocations)
        return allocations, scores
