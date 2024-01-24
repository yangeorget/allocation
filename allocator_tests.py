import numpy as np

from allocator import Allocator


def test_verify_allocation_ok():
    ok = np.array([[True, False, False], [False, True, True]])
    good_allocation = np.array([[True, False, False], [False, True, False]])
    assert Allocator.verify_allocation_ok(good_allocation, ok)
    bad_allocation = np.array([[True, True, False], [False, True, False]])
    assert not Allocator.verify_allocation_ok(bad_allocation, ok)


def test_verify_allocation_user_max_offer_nb():
    good_allocation = np.array([[True, False, False], [False, True, False]])
    assert Allocator.verify_allocation_user_max_offer_nb(good_allocation, 1)
    bad_allocation = np.array([[True, True, False], [False, True, False]])
    assert not Allocator.verify_allocation_user_max_offer_nb(bad_allocation, 1)


def test_verify_allocation_offer_max_user_nb():
    scores = np.array([[0.3, 0.2, 0.1], [0.1, 0.2, 0.3]])
    offer_max_user_nb = np.array([[0.3, 0.3, 0.3]])
    good_allocation = np.array([[True, False, False], [False, True, False]])
    assert Allocator.verify_allocation_offer_max_user_nb(good_allocation, scores, offer_max_user_nb)
    bad_allocation = np.array([[True, True, False], [False, True, False]])
    assert not Allocator.verify_allocation_offer_max_user_nb(bad_allocation, scores, offer_max_user_nb)


def test_compute_user_allocations():
    scores = np.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]])
    allocations = Allocator.compute_user_allocations(scores, 2)
    assert np.all(allocations == np.array([[True, True, False, False], [False, False, True, True]]))


def test_compute_offer_allocations():
    scores = np.array([[0.4, 0.2, 0.1], [0.1, 0.2, 0.2], [0.2, 0.3, 0.1]])
    allocations = Allocator.compute_offer_allocations(scores, np.array([0.4, 0.4, 0.6]))
    assert np.all(allocations == np.array([[True, True, True], [False, True, True], [False, False, True]]))


def test_update_allocations():
    a = np.array([True, True, False])
    b = np.array([False, True, True])
    Allocator.update_allocations(a, b)
    assert np.all(a == np.array([False, True, False]))
    assert np.all(b == np.array([False, True, True]))


def test_update_scores():
    a = np.array([0.1, 0.2, 0.3])
    b = np.array([False, True, True])
    Allocator.update_scores(a, b)
    assert np.all(a == np.array([0.0, 0.2, 0.3]))
    assert np.all(b == np.array([False, True, True]))
