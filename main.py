import numpy as np

from greedy_allocator import GreedyAllocator
from repair_allocator import RepairAllocator


def main():
    user_nb = 100
    offer_nb = 100
    max_offer_nb = 20
    offer_max_user_nb = np.random.randint(1, user_nb, size=offer_nb)
    init_allocations = np.random.rand(user_nb, offer_nb) > (1.0 / offer_nb)
    init_scores = np.random.random((user_nb, offer_nb))
    print("GreedyAllocator")
    GreedyAllocator(max_offer_nb, offer_max_user_nb, init_allocations, init_scores).optimize()
    print("RepairAllocator")
    RepairAllocator(max_offer_nb, offer_max_user_nb, init_allocations, init_scores).optimize()


if __name__ == "__main__":
    main()
