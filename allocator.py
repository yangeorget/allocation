import random

import numpy as np


class Allocator:
    def __init__(self, max_offer_nb, offer_max_user_nb, init_allocations, init_scores):
        self.user_nb = init_allocations.shape[0]
        self.offer_nb = init_allocations.shape[1]
        self.max_offer_nb = max_offer_nb
        self.offer_max_user_nb = offer_max_user_nb
        self.init_allocations = init_allocations
        self.init_scores = init_scores

    def optimize(self, nb_iter):
        print(self.name())
        best_score_sum = 0
        for iteration in range(nb_iter):
            # print(f"iteration {iteration}")
            allocations, scores = self.solve()
            assert Allocator.verify_allocation_ok(allocations, self.init_allocations)
            assert Allocator.verify_allocation_user_max_offer_nb(allocations, self.max_offer_nb)
            assert Allocator.verify_allocation_offer_max_user_nb(allocations, scores, self.offer_max_user_nb)
            score_sum = np.sum(scores)
            if score_sum > best_score_sum:
                best_score_sum = score_sum
                best_allocations = allocations
                print(f"iteration {iteration}: {best_score_sum}")
        return best_allocations, best_score_sum

    def solve(self):
        pass

    def name(self):
        return None

    @staticmethod
    def compute_user_allocations(scores, max_offer_nb, exploration_factor=0):
        top = 1 + random.randint(0, exploration_factor)
        bottom = max_offer_nb + top - 1
        partition = np.partition(scores, [-bottom, -top], axis=1)
        return np.logical_and(scores >= partition[:, [-bottom]], scores <= partition[:, [-top]])

    @staticmethod
    def compute_offer_allocations(scores, offer_max_user_nb):
        return np.cumsum(scores, axis=0) <= offer_max_user_nb

    @staticmethod
    def update_allocations(allocations, new_allocations):
        np.logical_and(allocations, new_allocations, out=allocations)

    @staticmethod
    def update_scores(scores, allocations):
        np.minimum(scores, 1.0 * allocations, out=scores)

    @staticmethod
    def verify_allocation_ok(allocation, ok):
        constraints = np.logical_or(np.logical_not(allocation), ok)
        return np.all(constraints)

    @staticmethod
    def verify_allocation_user_max_offer_nb(allocation, max_offer_nb):
        offer_nb_by_user = np.sum(allocation, axis=1)
        return np.all(offer_nb_by_user <= max_offer_nb)

    @staticmethod
    def verify_allocation_offer_max_user_nb(allocation, scores, offer_max_user_nb):
        user_nb_by_offer = np.sum(np.multiply(allocation, scores), axis=0)
        return np.all(user_nb_by_offer <= offer_max_user_nb)
