import numpy as np

from allocator import Verifier


def test_verify_constraint_init_allocations():
    ok = np.array([[True, False, False], [False, True, True]])
    verifier = Verifier(None, None, ok, None)
    good_allocations = np.array([[True, False, False], [False, True, False]])
    verifier.verify_constraint_init_allocations(good_allocations)
    bad_allocations = np.array([[True, True, False], [False, True, False]])
    assert not verifier.verify_constraint_init_allocations(bad_allocations)


def test_verify_allocations_user_max_offer_nb():
    good_allocations = np.array([[True, False, False], [False, True, False]])
    verifier = Verifier(1, None, None, None)
    assert verifier.verify_constraint_user_max_offer_nb(good_allocations)
    bad_allocations = np.array([[True, True, False], [False, True, False]])
    assert not verifier.verify_constraint_user_max_offer_nb(bad_allocations)


def test_verify_allocation_offer_max_user_nb():
    verifier = Verifier(None, np.array([[0.3, 0.3, 0.3]]), None, np.array([[0.3, 0.2, 0.1], [0.1, 0.2, 0.3]]))
    good_allocations = np.array([[True, False, False], [False, True, False]])
    assert verifier.verify_constraint_offer_max_user_nb(good_allocations)
    bad_allocations = np.array([[True, True, False], [False, True, False]])
    assert not verifier.verify_constraint_offer_max_user_nb(bad_allocations)
