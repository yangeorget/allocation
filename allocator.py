import numpy as np

from verifier import Verifier


class Allocator(Verifier):
    def optimize(self, iteration_nb):
        init_costs = self.scores * self.generosities * self.bools
        best_result = {"evaluation": 0, "allocations": None, "costs": None}
        early_stop_nb = 0
        for iteration in range(iteration_nb):
            allocations, costs = self.solve(init_costs, best_result)
            if allocations is not None:
                # self.assert_allocation_constraints(allocations)
                evaluation = self.evaluate(costs)
                if evaluation > best_result["evaluation"]:
                    best_result["evaluation"] = evaluation
                    best_result["allocations"] = allocations
                    best_result["costs"] = costs
                    print(f"iteration {iteration}: {evaluation}")
            else:
                early_stop_nb += 1
        return best_result, {"iteration_nb": iteration_nb, "early_stop_nb": early_stop_nb}

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
