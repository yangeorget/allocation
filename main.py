import numpy as np

from greedy_allocator import GreedyAllocator
from repair_allocator import RepairAllocator


def main():
    user_nb = 10000
    offer_nb = 100
    max_offer_nb = 20
    offer_max_user_nb = np.random.randint(1, user_nb, size=offer_nb)
    init_allocations = np.random.rand(user_nb, offer_nb) > (1.0 / offer_nb)
    init_scores = np.random.random((user_nb, offer_nb))
    print(f"VMax={np.sum(offer_max_user_nb)}, HMax={user_nb*max_offer_nb}")
    allocator = GreedyAllocator(max_offer_nb, offer_max_user_nb, init_allocations, init_scores)
    allocations, _ = allocator.optimize(10)
    file_name = f"{allocator.name()}.csv"
    allocator.save(file_name, allocations)
    allocator.assert_file(file_name)


if __name__ == "__main__":
    main()
