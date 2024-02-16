import numpy as np

from verifier import Verifier


class Allocator(Verifier):
    def __init__(
        self,
        offer_max_nb=0,
        families=None,
        budgets=None,
        generosities=None,
        bools=None,
        scores=None,
    ):
        super().__init__(offer_max_nb, families, budgets, generosities, bools, scores)
        offer_family_size = [
            np.count_nonzero(self.families == self.families[idx]) for idx in range(self.offer_nb)
        ]

    def optimize(self, nb_iter):
        best_evaluation = 0
        for iteration in range(nb_iter):
            allocations, costs = self.solve(best_evaluation)
            if allocations is not None:
                self.assert_allocation_constraints(allocations)
                evaluation = self.evaluate(costs)
                if evaluation > best_evaluation:
                    best_evaluation = evaluation
                    best_allocations = allocations
                    print(f"iteration {iteration}: {best_evaluation}")
        return best_allocations, best_evaluation

    def save(self, name, allocations):
        np.savetxt(name, allocations, delimiter=",", fmt="%d")

    def solve(self, best_evaluation):
        pass

    def evaluate(self, costs):
        return np.sum(costs)

    def update_costs(self, costs, allocations):
        np.multiply(costs, allocations, out=costs)

    def inverse_permutation(self, a):
        b = np.arange(a.shape[0])
        b[a] = b.copy()
        return b
