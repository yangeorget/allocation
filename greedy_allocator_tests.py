import numpy as np
import pytest

from allocator import Allocator
from greedy_allocator import GreedyAllocator


def test_inverse_permutation():
    assert np.all(np.array([3, 2, 0, 1]) == GreedyAllocator.inverse_permutation(np.array([2, 3, 1, 0])))
