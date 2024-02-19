import numpy as np

from allocator import Allocator


def test_update_costs_with_allocations():
    a = np.array([0.1, 0.2, 0.3])
    b = np.array([False, True, True])
    Allocator().update_costs_with_allocations(a, b)
    assert np.all(a == np.array([0.0, 0.2, 0.3]))
    assert np.all(b == np.array([False, True, True]))
