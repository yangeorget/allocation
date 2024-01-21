import numpy as np


def main():
    user_nb = 10000
    offer_nb = 100
    max_offer_nb = 20
    offer_max_user_nb = np.random.randint(1, user_nb, size=offer_nb)
    ok = np.random.rand(user_nb, offer_nb) > (1.0 / offer_nb)
    scores = np.random.random((user_nb, offer_nb))
    Allocator(max_offer_nb, offer_max_user_nb, ok, scores).optimize()


def main2():
    max_offer_nb = 2
    offer_max_user_nb = 3
    ok = np.array([[True, True, True], [True, True, True], [True, True, True]])
    scores = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    Allocator(max_offer_nb, offer_max_user_nb, ok, scores).solve()


class Allocator:
    def __init__(self, max_offer_nb, offer_max_user_nb, ok, scores):
        self.max_offer_nb = max_offer_nb
        self.offer_max_user_nb = offer_max_user_nb
        self.ok = ok
        self.scores = scores

    def optimize(self):
        best_score_sum = 0
        for iteration in range(10000):
            permutation, allocations, score_sum = self.solve()
            if score_sum > best_score_sum:
                best_score_sum = score_sum
                best_permutation = permutation
                best_allocations = allocations
                print(f"iteration {iteration}: {best_score_sum}")
        return best_permutation, best_allocations, best_score_sum

    def solve(self):
        user_nb = self.ok.shape[0]
        permutation = np.random.permutation(user_nb)
        return self.solve_permutation(permutation)

    def solve_permutation(self, permutation):
        shuffled_scores = self.scores[permutation]
        print(shuffled_scores)
        shuffled_ok = self.ok[permutation]
        print(shuffled_ok)
        allocations, sum_score = self.solve_constraints(shuffled_ok, shuffled_scores)
        return permutation, allocations, sum_score

    def solve_constraints(self, shuffled_ok, shuffled_scores):
        allocations = shuffled_ok
        scores = shuffled_scores
        update_scores(scores, allocations)
        update_allocations(
            allocations, scores >= compute_user_scores(scores, self.max_offer_nb)
        )
        update_scores(scores, allocations)
        update_allocations(
            allocations, compute_offer_allocations(scores, self.offer_max_user_nb)
        )
        update_scores(scores, allocations)
        assert verify_allocation_ok(allocations, shuffled_ok)
        assert verify_allocation_user_max_offer_nb(allocations, self.max_offer_nb)
        assert verify_allocation_offer_max_user_nb(
            allocations, shuffled_scores, self.offer_max_user_nb
        )
        return allocations, np.sum(scores)


@staticmethod
def update_allocations(allocations, new_allocations):
    np.logical_and(allocations, new_allocations, out=allocations)
    print(allocations)


@staticmethod
def update_scores(scores, allocations):
    np.minimum(scores, allocations, out=scores)
    print(scores)


@staticmethod
def compute_offer_allocations(scores, offer_max_user_nb):
    return np.cumsum(scores, axis=0) <= offer_max_user_nb


@staticmethod
def compute_user_scores(scores, max_offer_nb):
    return np.partition(scores, -max_offer_nb, axis=1)[:, -max_offer_nb].reshape(
        scores.shape[0], 1
    )


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


if __name__ == "__main__":
    main2()
