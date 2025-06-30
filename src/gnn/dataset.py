import random
from itertools import combinations
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from scipy.spatial.distance import pdist
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

__all__ = ["TSPDataset", "NormalizeEdges", "create_tsp_graph"]


class NormalizeEdges(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph: Data) -> Data:
        """Normalizes edge distances to [0, 1] in-place."""
        edge_distances = graph.edge_attr
        edge_distances = (edge_distances - edge_distances.min()) / (edge_distances.max() - edge_distances.min() + 1e-10)
        graph.edge_attr = edge_distances
        return graph


def create_tsp_graph(
    coords: List[List[float]],
    decision_variables: List[int],
    optimal_tour: Optional[List[int]] = None,
    edge_order: Optional[List[Tuple[int, int]]] = None,
    number_of_points: Optional[int] = None,
) -> Data:
    """Creates a PyG Data object for a TSP instance."""
    num_nodes = len(coords)
    coords_tensor = torch.tensor(coords, dtype=torch.float)

    if edge_order is None:
        edge_order = list(combinations(range(num_nodes), 2))  # (i,j) where i < j

    edge_index = torch.tensor(edge_order, dtype=torch.long).t().contiguous()
    edge_distances = torch.tensor(pdist(coords, "euclidean"), dtype=torch.float)
    labels = torch.tensor(decision_variables, dtype=torch.long)

    return Data(
        x=coords_tensor,
        edge_index=edge_index,
        edge_attr=edge_distances,
        y=labels,
        optimal_tour=optimal_tour,
        num_nodes=number_of_points,
    )


class TSPDataset(Dataset):
    """A PyTorch Geometric Dataset class for Traveling Salesman Problem (TSP) instances.

    This dataset takes solved TSP instances and converts them into graph representations
    suitable for training Graph Neural Networks (GNNs). Each instance represents a complete
    graph where nodes are cities and edges contain distance information.

    Args:
        tsp_instances (List[dict]): A list of dictionaries, where each dictionary represents
            a solved TSP instance with the following structure:
            {
                'coordinates': List[List[float]],  # List of [x,y] coordinates for each city
                'decision_variables': List[int],   # Binary edge labels (1 if in optimal tour)
                'optimal_tour': List[int],         # Sequence of node indices in optimal tour
                'minimum_distance': float          # Length of optimal tour (optional)
            }
            Note: All instances must have the same number of cities.

        transform (Callable, optional): A function/transform that takes in a PyG Data object
            and returns a transformed version. Defaults to None.
        normalize_edges (bool, optional): If True, normalizes edge distances to [0, 1] range.
            Defaults to True.

    Example:
        >>> tsp_data = [
        ...     {
        ...         'coordinates': [[0,0], [1,0], [1,1]],
        ...         'decision_variables': [1, 0, 1],  # (0-1, 0-2, 1-2)
        ...         'optimal_tour': [0, 1, 2],
        ...         'minimum_distance': 2.0
        ...     }
        ... ]
        >>> dataset = TSPDataset(tsp_instances=tsp_data)
        >>> len(dataset)
        1
    """

    def __init__(
        self,
        tsp_instances: List[dict],
        transform: Optional[Callable] = None,
        normalize_edges: bool = True,
        num_samples: Optional[int] = None,
    ):
        super().__init__()
        self.transform = transform
        self.normalize_edges = normalize_edges
        self.tsp_instances = tsp_instances

        # Validate instances after sampling
        self._validate_instances()

        # Apply sampling if requested
        self.tsp_instances = (
            random.sample(tsp_instances, num_samples)
            if num_samples and num_samples < len(tsp_instances)
            else tsp_instances
        )

    def _validate_instances(self):
        """Validates the structure of input TSP instances."""
        required_keys = {"coordinates", "decision_variables"}
        for instance in self.tsp_instances:
            missing_keys = required_keys - set(instance.keys())
            if missing_keys:
                raise KeyError(f"Instance missing required keys: {missing_keys}")

            # Check coordinates are 2D
            if not all(len(coord) == 2 for coord in instance["coordinates"]):
                raise ValueError("All coordinates must be 2D [x,y] pairs")

            # Check decision_variables matches edge count
            n = len(instance["coordinates"])
            expected_edges = n * (n - 1) // 2
            if len(instance["decision_variables"]) != expected_edges:
                raise ValueError(
                    f"Expected {expected_edges} decision variables for {n} cities, "
                    f"got {len(instance['decision_variables'])}"
                )

    def __len__(self) -> int:
        return len(self.tsp_instances)

    def __getitem__(self, idx) -> Data:
        instance = self.tsp_instances[idx]
        graph = create_tsp_graph(
            coords=instance["coordinates"],
            decision_variables=instance["decision_variables"],
            optimal_tour=instance.get("optimal_tour", None),
            number_of_points=instance.get("number_of_points", None),
        )

        if self.transform:
            graph = self.transform(graph)

        if self.normalize_edges:
            graph = NormalizeEdges()(graph)

        return graph

    def get_dataloaders(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,  # validation set ratio
        batch_size: int = 16,
        shuffle: bool = True,
        random_state: int = 42,
        stratify: bool = True,  # Criterion is the number of points
    ):
        """Splits dataset into train/val/test DataLoaders with optional stratification.

        Args:
            test_size: Fraction of dataset to use for testing
            val_size: Fraction of dataset to use for validation
            batch_size: Number of samples before update the weights
            shuffle: bool: Whether to shuffle the batches.
            random_state: For reproducability of the results. Pseudorandom split to train, validation and test set.
            stratify: If True, splits maintain original distribution of city counts
        """
        # Extract number of points for stratification if needed
        stratify_by = None
        if stratify:
            stratify_by = [len(inst["coordinates"]) for inst in self.tsp_instances]

        # First split: train + (val + test)
        train_instances, temp_instances = train_test_split(
            self.tsp_instances,
            test_size=(val_size + test_size),
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify_by,
        )

        # Second split: val + test (adjusting test_size proportionally)
        adjusted_test_size = test_size / (val_size + test_size)
        val_instances, test_instances = train_test_split(
            temp_instances,
            test_size=adjusted_test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=[len(inst["coordinates"]) for inst in temp_instances] if stratify else None,
        )

        # Create datasets
        train_dataset = TSPDataset(train_instances, self.transform, self.normalize_edges)
        val_dataset = TSPDataset(val_instances, self.transform, self.normalize_edges)
        test_dataset = TSPDataset(test_instances, self.transform, self.normalize_edges)

        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        )
