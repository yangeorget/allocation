import numpy as np


class Verifier:
    def __init__(self, max_offer_nb, offer_max_user_nb, init_allocations, init_scores):
        if init_allocations is not None:
            self.user_nb = init_allocations.shape[0]
            self.offer_nb = init_allocations.shape[1]
        self.max_offer_nb = max_offer_nb
        self.offer_max_user_nb = offer_max_user_nb
        self.init_allocations = init_allocations
        self.init_scores = init_scores

    def verify_constraint_init_allocations(self, allocations):
        constraints = np.logical_or(np.logical_not(allocations), self.init_allocations)
        return np.all(constraints)

    def verify_constraint_user_max_offer_nb(self, allocations):
        offer_nb_by_user = np.sum(allocations, axis=1)
        return np.all(offer_nb_by_user <= self.max_offer_nb)

    def verify_constraint_offer_max_user_nb(self, allocations):
        user_nb_by_offer = np.sum(np.multiply(allocations, self.init_scores), axis=0)
        return np.all(user_nb_by_offer <= self.offer_max_user_nb)

    def assert_allocation_constraints(self, allocations):
        assert self.verify_constraint_offer_max_user_nb(allocations)
        assert self.verify_constraint_user_max_offer_nb(allocations)
        assert self.verify_constraint_init_allocations(allocations)

    def assert_file(self, name):
        allocations = np.loadtxt(name, delimiter=",", dtype=bool)
        self.assert_allocation_constraints(allocations)
