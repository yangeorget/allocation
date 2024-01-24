import numpy as np

from greedy_allocator import GreedyAllocator


class RepairAllocator(GreedyAllocator):
    def find_solution(self, permutation):
        shuffled_allocations, shuffled_scores = super().find_solution(permutation)
        self.improve_solution(shuffled_allocations, shuffled_scores, permutation)
        return shuffled_allocations, shuffled_scores

    def improve_solution(self, allocations, scores, permutation):
        old_scores = self.init_scores[permutation]
        old_allocations = self.init_allocations[permutation]
        offer_score_sum = np.sum(scores, axis=1)
        for user_index in range(self.user_nb):
            user_true_nb = np.count_nonzero(allocations[user_index])
            for offer_index in range(self.offer_nb):
                if user_true_nb == self.max_offer_nb:
                    break
                if (
                    not allocations[user_index, offer_index]
                    and old_allocations[user_index, offer_index]
                    and offer_score_sum[offer_index] + old_scores[user_index, offer_index]
                    < self.offer_max_user_nb[offer_index]
                ):
                    allocations[user_index, offer_index] = True
                    scores[user_index, offer_index] = old_scores[user_index, offer_index]
                    user_true_nb += 1
                    offer_score_sum[offer_index] += scores[user_index, offer_index]

