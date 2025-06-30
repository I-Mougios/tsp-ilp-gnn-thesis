# src/tsp/utils.py
import csv
import gzip
import math
from functools import singledispatch
from math import factorial
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
from networkx import Graph, connected_components

__all__ = [
    "find_subtours",
    "number_of_pairwise_distances",
    "bool_",
    "default",
    "generate_clustered_tsp_data",
    "open_file",
    "open_dataset",
    "open_optimal_tour",
    "match_dataset_with_optimal_solution",
    "match_tsplib_dataset_with_solution",
]


def find_subtours(solution_array: np.ndarray, distances_df: pd.DataFrame) -> Tuple[Set[int]]:
    """
    Finds the sub-tours of the solutions array of a TSP problem.

    Args:
        solution_array (np.ndarray): A 1D array where each element represents whether a particular
                                      edge in the distance DataFrame is part of the solution (1 for yes, 0 for no).

        distances_df (pd.DataFrame): A DataFrame where each row represents the distances between two points

    Returns:
        tuple: A tuple of sets, where each set contains the nodes of a connected component in the graph
               formed by the solution array.
    """
    distances_df = distances_df.copy()
    # Filter the edges included in the optimal solution
    selected_edges = distances_df[  # selected edges is a list of tuples
        [bool_(is_selected) for is_selected in solution_array]
    ].index

    # Create a graph using the selected edges
    optimal_graph = Graph(list(selected_edges))

    # Find and return the connected components in the graph
    return tuple(connected_components(optimal_graph))


def number_of_pairwise_distances(number_of_points) -> int:
    return int(factorial(number_of_points) / (factorial(2) * factorial(number_of_points - 2)))


def bool_(value) -> bool:
    return False if math.isclose(value, 0, rel_tol=1e-06, abs_tol=1e-06) else True


@singledispatch
def default(obj):
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@default.register
def _(obj: np.ndarray):
    return obj.tolist()


@default.register
def _(obj: np.int64):
    return int(obj)


# Function to generate clustered city coordinates
def generate_clustered_tsp_data(num_cities=100, num_clusters=5, cluster_std=5, seed=None):
    """
    Generate TSP data with cities distributed in clusters.

    Parameters:
    - num_cities (int): Total number of cities.
    - num_clusters (int): Number of clusters.
    - cluster_std (float): Standard deviation of clusters.
    - seed (int): Random seed for reproducibility.

    Returns:
    - pd.DataFrame: A DataFrame containing city coordinates.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate cluster centers
    cluster_centers = np.random.uniform(0, 100, size=(num_clusters, 2))

    # Assign cities to clusters and generate coordinates
    cities = []
    for i in range(num_clusters):
        num_cities_in_cluster = num_cities // num_clusters
        if i < num_cities % num_clusters:
            num_cities_in_cluster += 1

        cluster_cities = np.random.normal(loc=cluster_centers[i], scale=cluster_std, size=(num_cities_in_cluster, 2))
        cities.extend(cluster_cities)

    cities = np.array(cities)

    # Create DataFrame
    tsp_dataset = pd.DataFrame({"x": cities[:, 0], "y": cities[:, 1]})

    return tsp_dataset


def remove_null_values(row: List):
    for _ in range(row.count("")):
        row.remove("")


def clean_row(file):
    for row in file:
        row = row.strip()
        try:
            if row and row[0].isdigit():
                row = row.split(" ")
                remove_null_values(row)
                yield [float(value) for value in row]
        except Exception as e:
            print(row)
            raise e


def open_file(filename: str | Path):
    with gzip.open(filename, "rb") as file:
        file = file.read().decode("utf-8")
        file = file.split("\n")
        yield from clean_row(file)


def open_optimal_tour(filename):
    with open(filename) as f:
        file = f.read().split("\n")
        minimum_distance = float(file.pop(0).split(":")[1])
        file.pop(0)
        file.remove("")
        optimal_tour = [int(town) for town in file]
        return optimal_tour, minimum_distance


def open_dataset(filename):
    rows = []
    with open(filename, mode="r", newline="\n") as f:
        file = csv.reader(f, delimiter=" ")
        for row in file:
            rows.append((float(row[0]), float(row[1])))

    return rows


def match_dataset_with_optimal_solution(datasets_dir, optimal_solutions_dir):
    optimal_solutions_filenames = filter(
        lambda filename: filename.name.startswith("results"), optimal_solutions_dir.iterdir()
    )

    datasets_filenames = filter(lambda filename: filename.name.startswith("dataset"), datasets_dir.iterdir())

    optimal_solutions_filenames = sorted(optimal_solutions_filenames, key=lambda filename: filename.stem.split("_")[1:])
    datasets_filenames = sorted(datasets_filenames, key=lambda filename: filename.stem.split("_")[1:])

    datasets_solutions = []
    for solution_path, dataset_path in zip(optimal_solutions_filenames, datasets_filenames):
        assert (
            dataset_path.stem.split("_")[1:] == solution_path.stem.split("_")[1:]
        ), f"Dataset and solution stem mismatch: {dataset_path.stem} and {solution_path.stem}"

        optimal_tour, minimum_distance = open_optimal_tour(solution_path)
        dataset = open_dataset(dataset_path)
        datasets_solutions.append((dataset, optimal_tour, minimum_distance))

    return datasets_solutions


def find_tsplib_dataset_based_on_solution_file(optimal_solutions_dir, datasets_dir):
    tsplib_datasets_solutions_files = []
    for solution_filename in optimal_solutions_dir.iterdir():
        identifier = solution_filename.stem.split("_")[1]
        for dataset_filename in datasets_dir.iterdir():
            if dataset_filename.stem.startswith(identifier):
                tsplib_datasets_solutions_files.append((dataset_filename, solution_filename))
                break
    return tsplib_datasets_solutions_files


def match_tsplib_dataset_with_solution(optimal_solutions_dir, datasets_dir):
    tsplib_datasets_solutions = []
    tsplib_datasets_solutions_paths = find_tsplib_dataset_based_on_solution_file(
        optimal_solutions_dir=optimal_solutions_dir, datasets_dir=datasets_dir
    )

    for dataset_path, solution_path in tsplib_datasets_solutions_paths:
        data = list(open_file(dataset_path))
        # The first element within the rows of data variable is the index and I want to drop it
        for row in data:
            row.pop(0)
        optimal_tour, minimum_distance = open_optimal_tour(solution_path)
        tsplib_datasets_solutions.append((data, optimal_tour, minimum_distance))

    return tsplib_datasets_solutions
