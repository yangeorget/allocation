import numpy as np

from verifier import Verifier


class Allocator(Verifier):
    def optimize(self, nb_iter):
        best_evaluation = 0
        for iteration in range(nb_iter):
            allocations, costs = self.solve()
            self.assert_allocation_constraints(allocations)
            evaluation = self.evaluate(costs)
            if evaluation > best_evaluation:
                best_evaluation = evaluation
                best_allocations = allocations
                print(f"iteration {iteration}: {best_evaluation}")
        return best_allocations, best_evaluation

    def save(self, name, allocations):
        np.savetxt(name, allocations, delimiter=",", fmt="%d")

    def solve(self):
        pass

    def evaluate(self, costs):
        return np.sum(costs)

    def update_costs(self, costs, allocations):
        np.multiply(costs, allocations, out=costs)
