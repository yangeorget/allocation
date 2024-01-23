import random
from matplotlib import pyplot
import numpy as np


def main():
    exploration_factor = 0
    user_nb = 100
    offer_nb = 20
    max_offer_nb = 10
    offer_max_user_nb = np.random.randint(1, user_nb, size=offer_nb)
    # print(f"offer_max_user_nb={offer_max_user_nb}")
    ok = np.random.rand(user_nb, offer_nb) > (1.0 / offer_nb)
    scores = np.random.random((user_nb, offer_nb))
    Allocator(max_offer_nb, offer_max_user_nb, ok, scores, exploration_factor).optimize()


class Allocator:
    def __init__(self, max_offer_nb, offer_max_user_nb, ok, scores, exploration_factor=0):
        self.exploration_factor = exploration_factor
        self.max_offer_nb = max_offer_nb
        self.offer_max_user_nb = offer_max_user_nb
        self.ok = ok
        self.scores = scores

    def optimize(self):
        best_score_sum = 0
        for iteration in range(100):
            # print(f"iteration {iteration}")
            permutation, allocations, score_sum = self.solve()
            if score_sum > best_score_sum:
                best_score_sum = score_sum
                best_permutation = permutation
                best_allocations = allocations
                print(f"iteration {iteration}: {best_score_sum}")
        return best_permutation, best_allocations, best_score_sum

    def display(self, allocations):
        pyplot.matshow(allocations)
        pyplot.show()

    def solve(self):
        user_nb = self.ok.shape[0]
        permutation = np.random.permutation(user_nb)  # np.arange(user_nb)
        return self.solve_permutation(permutation)

    def solve_permutation(self, permutation):
        # print(f"solve_permutation({permutation})")
        shuffled_scores = self.scores[permutation]
        shuffled_ok = self.ok[permutation]
        allocations, sum_score = self.solve_constraints(shuffled_ok, shuffled_scores)
        # inverse_permutation = self.inverse_permutation(permutation)
        # print(f"inverse_permutation={inverse_permutation}")
        # self.display(allocations[inverse_permutation])
        return permutation, allocations, sum_score

    def inverse_permutation(self, a):
        b = np.arange(a.shape[0])
        b[a] = b.copy()
        return b

    def solve_constraints(self, shuffled_ok, shuffled_scores):
        # print(f"solve_constraints({shuffled_ok}, {shuffled_scores})")
        allocations = shuffled_ok
        # self.display(allocations)
        scores = shuffled_scores
        update_scores(scores, allocations)
        user_allocations = compute_user_allocations(scores, self.max_offer_nb, self.exploration_factor)
        # self.display(user_allocations)
        update_allocations(allocations, user_allocations)
        update_scores(scores, allocations)
        offer_allocations = compute_offer_allocations(scores, self.offer_max_user_nb)
        # self.display(offer_allocations)
        update_allocations(allocations, offer_allocations)
        # self.display(allocations)
        update_scores(scores, allocations)
        assert verify_allocation_ok(allocations, shuffled_ok)
        assert verify_allocation_user_max_offer_nb(allocations, self.max_offer_nb)
        assert verify_allocation_offer_max_user_nb(allocations, shuffled_scores, self.offer_max_user_nb)
        return allocations, np.sum(scores)


def compute_user_allocations(scores, max_offer_nb, exploration_factor=0):
    top = 1 + random.randint(0, exploration_factor)
    bottom = max_offer_nb + top - 1
    partition = np.partition(scores, [-bottom, -top], axis=1)
    return np.logical_and(scores >= partition[:, [-bottom]], scores <= partition[:, [-top]])


def compute_offer_allocations(scores, offer_max_user_nb):
    return np.cumsum(scores, axis=0) <= offer_max_user_nb


def update_allocations(allocations, new_allocations):
    np.logical_and(allocations, new_allocations, out=allocations)


def update_scores(scores, allocations):
    np.minimum(scores, allocations, out=scores)


def verify_allocation_ok(allocation, ok):
    constraints = np.logical_or(np.logical_not(allocation), ok)
    return np.all(constraints)


def verify_allocation_user_max_offer_nb(allocation, max_offer_nb):
    offer_nb_by_user = np.sum(allocation, axis=1)
    return np.all(offer_nb_by_user <= max_offer_nb)


def verify_allocation_offer_max_user_nb(allocation, scores, offer_max_user_nb):
    user_nb_by_offer = np.sum(np.multiply(allocation, scores), axis=0)
    return np.all(user_nb_by_offer <= offer_max_user_nb)


if __name__ == "__main__":
    main()
