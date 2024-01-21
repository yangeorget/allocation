import numpy as np

from main import (
    verify_allocation_ok,
    verify_allocation_user_max_offer_nb,
    verify_allocation_offer_max_user_nb,
    compute_user_scores,
    compute_offer_allocations,
    Allocator,
)


def test_verify_allocation_ok():
    ok = np.array([[True, False, False], [False, True, True]])
    good_allocation = np.array([[True, False, False], [False, True, False]])
    assert verify_allocation_ok(good_allocation, ok)
    bad_allocation = np.array([[True, True, False], [False, True, False]])
    assert not verify_allocation_ok(bad_allocation, ok)


def test_verify_allocation_user_max_offer_nb():
    good_allocation = np.array([[True, False, False], [False, True, False]])
    assert verify_allocation_user_max_offer_nb(good_allocation, 1)
    bad_allocation = np.array([[True, True, False], [False, True, False]])
    assert not verify_allocation_user_max_offer_nb(bad_allocation, 1)


def test_verify_allocation_offer_max_user_nb():
    scores = np.array([[0.3, 0.2, 0.1], [0.1, 0.2, 0.3]])
    offer_max_user_nb = np.array([[0.3, 0.3, 0.3]])
    good_allocation = np.array([[True, False, False], [False, True, False]])
    assert verify_allocation_offer_max_user_nb(
        good_allocation, scores, offer_max_user_nb
    )
    bad_allocation = np.array([[True, True, False], [False, True, False]])
    assert not verify_allocation_offer_max_user_nb(
        bad_allocation, scores, offer_max_user_nb
    )


def test_compute_user_scores():
    scores = np.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]])
    scores = compute_user_scores(scores, 2)
    assert np.all(scores == np.array([0.3, 0.3]))


def test_compute_offer_allocations():
    scores = np.array([[0.4, 0.2, 0.1], [0.1, 0.2, 0.2], [0.2, 0.3, 0.1]])
    allocations = compute_offer_allocations(scores, np.array([0.4, 0.4, 0.6]))
    assert np.all(
        allocations
        == np.array([[True, True, True], [False, True, True], [False, False, True]])
    )


def test_solve_constraints():
    ok = np.array([[True, True, True], [True, True, True], [True, True, True]])
    scores = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    assert np.all(
        Allocator(2, 3, ok, scores).solve()
        == np.array(
            [[False, False, False], [False, False, False], [False, False, False]]
        )
    )
