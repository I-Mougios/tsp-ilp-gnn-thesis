# src/tsp/solver.py
import json
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, dok_matrix, vstack
from scipy.spatial.distance import pdist

from tsp.utils import bool_, default, find_subtours

__all__ = ["TSP"]


class TSP:
    all = []

    def __init__(self, coordinates: np.ndarray | pd.DataFrame, distance_metric: str = "euclidean"):

        if isinstance(coordinates, pd.DataFrame):
            coordinates = coordinates.to_numpy()

        self._coordinates = coordinates
        self.distance_metric = distance_metric
        self.n_points = self.coordinates.shape[0]
        self._distances = None
        self._equality_constraints_matrix = None
        self._inequality_constraints_matrix = None
        self._inequality_constraints_rhs = None
        self._result = None
        self._optimal_tour = None
        self._minimum_distance = None
        self._decision_variables = None
        self._while_loop_iterations = 0
        self._elapsed_time = None

    @property
    def coordinates(self) -> np.ndarray:
        return self._coordinates

    @property
    def distances(self) -> pd.DataFrame:
        if self._distances is None:
            self._distances = self._calculate_distance_df()

        return self._distances

    @property
    def number_of_points(self) -> int:
        return self.coordinates.shape[0]

    @property
    def dimensions(self) -> int:
        return self.coordinates.shape[1]

    def _calculate_distance_df(self) -> np.ndarray:
        return pdist(self.coordinates, metric=self.distance_metric)

    @property
    def equality_constraints_matrix(self) -> csr_matrix:
        if self._equality_constraints_matrix is None:
            # Create the pairs of points. Pairs of points are actually the possible trips
            idxs = list(combinations(range(self.number_of_points), r=2))
            number_of_pairs = len(idxs)

            # Initialize an empty matrix
            # Use a sparse DOK matrix for initial construction
            equality_constraints_matrix = dok_matrix((self.number_of_points, number_of_pairs), dtype=int)

            # Precompute trips for each town using defaultdict
            town_to_trips = defaultdict(list)
            # Gather the trips per each town
            for trip_idx, (a, b) in enumerate(idxs):
                town_to_trips[a].append(trip_idx)
                town_to_trips[b].append(trip_idx)

            for town, trips in town_to_trips.items():
                equality_constraints_matrix[town, trips] = 1

            self._equality_constraints_matrix = equality_constraints_matrix.tocsr()

        return self._equality_constraints_matrix

    @property
    def inequality_constraints_matrix(self) -> csr_matrix:
        if self._result is None:
            self.solve()

        return self._inequality_constraints_matrix

    @property
    def inequality_constraints_rhs(self) -> np.ndarray:
        if self._result is None:
            self.solve()

        return self._inequality_constraints_rhs

    @property
    def equality_constraints_rhs(self) -> np.ndarray:
        return 2 * np.ones(self.number_of_points)

    @property
    def result(self) -> dict:
        if self._result is None:
            self.solve()

        return self._result

    @property
    def optimal_tour(self) -> List[int]:
        if self._optimal_tour is None:
            self.find_optimal_tour()

        return self._optimal_tour

    @property
    def minimum_distance(self) -> float:
        if self._minimum_distance is None:
            self.solve()

        return self._minimum_distance

    @property
    def decision_variables(self) -> np.ndarray[0 | 1]:
        if self._decision_variables is None:
            self.solve()

        return self._decision_variables

    @property
    def while_loop_iterations(self) -> int:
        if self.result is None:
            self.solve()
        return self._while_loop_iterations

    @property
    def elapsed_time(self):
        if self._result is None:
            self.solve()
        return self._elapsed_time

    def solve(self, verbose: bool = False) -> None:

        distances_df = pd.DataFrame(data=self.distances, index=tuple(combinations(range(self.number_of_points), r=2)))

        start_time = perf_counter()
        result = res = linprog(
            self.distances,
            A_eq=self.equality_constraints_matrix,
            b_eq=self.equality_constraints_rhs,
            bounds=(0, 1),
            integrality=1,
        )

        # assert self.number_of_points == sum(map(bool_, result.x))

        # Optimal solution with subtours
        subtours = find_subtours(solution_array=result.x, distances_df=distances_df)

        if res.success:
            if len(subtours) > 1:
                # Each constraint will be a single row sparse matrix
                ineq_constraints_rows = []
                b_ineq = np.array([])  # Right-hand side of the inequality constraints
        else:
            print(result.message)

        if verbose:
            print("Number of subtours: ", len(subtours))
            print("Minimum distance: ", result.fun)

        while len(subtours) > 1:
            self._while_loop_iterations += 1
            for idx, subtour in enumerate(subtours):

                # Sort the subtour to ensure deterministic generation of combinations
                subtour = sorted(subtour)

                # Generate all possible edges (combinations of 2 nodes) within the subtour
                subtour_edges = list(combinations(subtour, 2))

                # Add a new row to the DOK matrix for the current subtour
                new_row = dok_matrix((1, len(distances_df.index)), dtype=int)
                for edge in subtour_edges:
                    # If the subtour has three edges A, B, C then in order to break the subtour
                    # I have to implement the constraint 1*A + 1*B + 1*C < 2
                    edge_index = distances_df.index.get_loc(edge)
                    # Set the coefficient of the trip equal to 1
                    new_row[0, edge_index] = 1

                ineq_constraints_rows.append(new_row)

                # Update the right-hand side for the new constraint
                b_ineq = np.append(b_ineq, len(subtour) - 1)

            # Convert the DOK matrix to CSR for computation
            A_ineq = vstack(ineq_constraints_rows).tocsr()

            # Solve the problem with the updated constraints
            result = linprog(
                self.distances,
                A_eq=self.equality_constraints_matrix,
                b_eq=self.equality_constraints_rhs,
                A_ub=A_ineq,
                b_ub=b_ineq,
                bounds=(0, 1),
                integrality=1,
            )

            if result.success:
                subtours = find_subtours(solution_array=result.x, distances_df=distances_df)
            else:
                print(result.message)
                break

            if verbose:
                print("Number of subtours: ", len(subtours))
                print("Minimum distance: ", result.fun)
        # End of while loop
        end_time = perf_counter()
        if "A_ineq" in locals():
            self._inequality_constraints_matrix = A_ineq
        if "b_ineq" in locals():
            self._inequality_constraints_rhs = b_ineq

        self._result = result
        self._minimum_distance = result.fun
        self._decision_variables = result.x
        self._elapsed_time = (end_time - start_time) / 60
        self.find_optimal_tour()
        self.all.append(self)

    def coordinates_to_file(self, filename: str) -> None:
        with open(filename, "w") as f:
            lines = [" ".join(map(str, coord)) + "\n" for coord in self.coordinates]
            f.writelines(lines)

    def find_optimal_tour(self) -> List[int]:
        pairs = np.array(tuple(combinations(range(self.number_of_points), r=2)))
        # If the self.result is None it will call the self.solve()
        pairs_of_optimal_tour = pairs[list(map(bool_, self.result.x))]  # masking to get the edge of optimal tour
        tour = [pairs_of_optimal_tour[0, 0]]
        # In each iteration add the town that is connected with the previous one
        for i in range(0, len(pairs_of_optimal_tour)):
            # Find the pair in which the previous town is inside
            for p1, p2 in pairs_of_optimal_tour:
                # Exclude the pair that both of the towns are already included in the tour
                if p1 not in tour or p2 not in tour:
                    # When you find the second pair of each town append the town that is not already inside in the tour
                    if tour[i] == p1:  # tour[i] is the last town in the sequence that represent the optimal tour
                        if p2 not in tour:
                            tour.append(p2)
                            break
                    elif tour[i] == p2:
                        if p1 not in tour:
                            tour.append(p1)
                            break
        self._optimal_tour = tour
        return tour

    def plot_optimal_tour(self, save_path=None, style="fast", color="black") -> None:
        plt.style.use(style)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(self.coordinates[:, 0], self.coordinates[:, 1])
        coordinates_of_optimal_tour = self.coordinates[self.optimal_tour]

        for i in range(len(self.optimal_tour) - 1):
            x_start, y_start = coordinates_of_optimal_tour[i]
            x_end, y_end = coordinates_of_optimal_tour[i + 1]

            ax.arrow(
                x_start,
                y_start,
                dx=x_end - x_start,
                dy=y_end - y_start,
                linestyle="--",
                color="r",
            )

            ax.annotate(
                text=f"{i + 1}",
                xy=(x_start, y_start),
                color=color,
            )

        # Plot the last arrow
        ax.arrow(
            x_end,
            y_end,
            dx=coordinates_of_optimal_tour[0, 0] - x_end,
            dy=coordinates_of_optimal_tour[0, 1] - y_end,
            linestyle="--",
            color="r",
        )

        ax.annotate(text=f"{i + 2}", xy=(x_end, y_end), color=color)

        if save_path:
            save_path = save_path + ".png"
            plt.savefig(save_path)

        plt.show()

    @classmethod
    def all_stats(cls):
        data = [[obj.number_of_points, obj.while_loop_iterations, obj.elapsed_time] for obj in cls.all]
        columns = ["Number of points", "Number of solutions with sub-tours", "Time elapsed (Minutes)"]
        df = pd.DataFrame(data, columns=columns)
        return df

    def result_to_file(self, filename: str) -> None:
        with open(filename, "w") as f:
            f.write(f"Minimum distance: {self.minimum_distance}\n")
            f.write("Optimal Tour:\n")
            for value in self.optimal_tour:
                f.write(f"{value}\n")

    def store_to_json(self, filepath: str | None = None) -> None:

        if filepath is None:
            project_dir = Path(__file__).resolve().parent.parent
            filepath = project_dir.joinpath("solved_tsp_instances.json")

        new_entry = self.to_dict()

        # Check if the file exists and read the existing data
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                try:
                    data = json.load(f)  # Load existing JSON
                except json.JSONDecodeError:
                    data = {}  # Handle empty/corrupted file gracefully
        else:
            data = {}

        # Determine the next available key (auto-incremental)
        next_key = str(max(map(int, data.keys())) + 1) if data else "1"

        # Add the new entry
        data[next_key] = new_entry

        # Write back the updated data without losing previous entries
        with open(filepath, "w") as f:
            json.dump(
                data,
                f,
                default=default,
            )

    def to_dict(self):
        dict_ = {
            "number_of_points": self.n_points,
            "distance_metric": self.distance_metric,
            "coordinates": self.coordinates.tolist(),
            "decision_variables": [round(value) for value in self.decision_variables],
            "optimal_tour": self.optimal_tour,
            "minimum_distance": self.minimum_distance,
            "subtour_revisions": self.while_loop_iterations,
            "elapsed_time": self.elapsed_time,
        }

        return dict_
