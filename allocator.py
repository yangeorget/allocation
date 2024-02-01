import random

import numpy as np

from verifier import Verifier


class Allocator(Verifier):
    def optimize(self, nb_iter):
        print(self.name())
        best_evaluation = 0
        for iteration in range(nb_iter):
            # print(f"iteration {iteration}")
            allocations, scores = self.solve()
            self.assert_allocation_constraints(allocations)
            evaluation = self.evaluate(scores)
            if evaluation > best_evaluation:
                best_evaluation = evaluation
                best_allocations = allocations
                print(f"iteration {iteration}: {best_evaluation}")
        return best_allocations, best_evaluation

    def save(self, name, allocations):
        np.savetxt(name, allocations, delimiter=",", fmt="%d")

    def solve(self):
        pass

    def evaluate(self, scores):
        return np.sum(scores)

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
