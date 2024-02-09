import numpy as np
from greedy_allocator import GreedyAllocator


def test_compute_user_allocations():
    costs = np.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]])
    user_allocations = GreedyAllocator(offer_max_nb=2).compute_user_allocations(costs)
    assert np.all(user_allocations == np.array([[True, True, False, False], [False, False, True, True]]))


def test_compute_offer_allocations():
    costs = np.array([[0.4, 0.2, 0.1], [0.1, 0.2, 0.2], [0.2, 0.3, 0.1]])
    offer_allocations = GreedyAllocator(budgets=np.array([0.4, 0.4, 0.6])).compute_offer_allocations(costs)
    assert np.all(offer_allocations == np.array([[True, True, True], [False, True, True], [False, False, True]]))
