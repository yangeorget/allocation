import argparse
import importlib

from rich.pretty import pprint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Allocator", description="Generates an allocation file")
    parser.add_argument(
        "-d",
        "--data",
        help="the path to the directory containing the constraint files",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        help="the number of iterations",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="the path of the output file",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        help="the algorithm full name",
    )
    args = parser.parse_args()
    module_name, class_name = args.algorithm.rsplit(".", 1)
    allocator_class = getattr(importlib.import_module(module_name), class_name)
    allocator = allocator_class()
    allocator.init_from_files(args.data)
    # allocator.compute_problem_stats()
    best_result = allocator.optimize(args.iterations)
    allocations = best_result["allocations"]
    pprint(allocator.compute_solution_stats(allocations), expand_all=True)
    allocator.save(args.output, allocations)
