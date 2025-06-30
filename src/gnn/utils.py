from itertools import combinations, cycle, islice, pairwise
from typing import Callable, Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.loader
from scipy.spatial.distance import pdist
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import SGD, Adam, AdamW, RMSprop

from .gat import TSPGNN

__all__ = [
    "beamsearch",
    "train_with_hyperparams",
    "calculate_tour_cost",
    "optimal_tour_as_tuples",
    "calculate_distances",
    "load_gnn_model",
]


def beamsearch(probabilities, edge_index, number_of_nodes, beam_size=5, n_candidates_per_beam_length=15):
    # Create the probabilities matrix
    prob_matrix = torch.zeros(size=(number_of_nodes, number_of_nodes))
    for i, (node_i, node_j) in enumerate(edge_index.t().tolist()):
        prob_matrix[node_i, node_j] = probabilities[i]
        prob_matrix[node_j, node_i] = probabilities[i]

    # Create the nodes (Needed to create the mask)
    nodes_indices = torch.arange(number_of_nodes)

    # Start from node 0 --> dict[tour, score]
    beams = {(0,): 0}
    new_beams = {}
    # We want to create a hamiltonian cycle by adding n-1 nodes to the tour
    for _ in range(number_of_nodes - 1):
        # In each iteration all the tours of previous iteration will be removed
        # In each iteration beam_size * number_of_tours_of_previous_iteration will be added
        for tour, score in beams.items():
            last_node = tour[-1]
            transition_probs_from_last_node = prob_matrix[last_node]

            # Mask the non-visited nodes by excluding those already in the current tour
            # If it is the tour the isin will return True(visited), and after the negation it return the non-visited
            non_visited_nodes_mask = ~torch.isin(
                elements=nodes_indices, test_elements=torch.tensor(tour)
            )  # output [True, False, False ...] size=number of nodes
            indices_of_non_visited_nodes = nodes_indices[non_visited_nodes_mask]

            # Apply the mask to the transition probabilities
            # (Only non visited nodes as candidates for the next destination)
            possible_transitions = transition_probs_from_last_node[non_visited_nodes_mask]

            # This line is required when the beam_size will be larger than the possible nodes
            # So the topk will fail
            k = min(beam_size, possible_transitions.size(0))

            # Find top-k (beam_size) transitions - topk --> values, indices
            # The magic is that indices_of_non_visited_nodes are one to one with the possible_transitions tensor
            probs, topk_nodes_indices = torch.topk(possible_transitions, k=k)

            # We need to map the topk_nodes_indices tensor back to the original tensor's index positions
            original_indices = indices_of_non_visited_nodes[topk_nodes_indices]
            next_nodes = nodes_indices[original_indices]

            # Add the new tours
            for prob, node in zip(probs, next_nodes):
                new_beams[tour + (node.item(),)] = score + prob.item()
            # ====== Increased beam length by one  for all of beams ========================

        # Select the top candidates(highest product of probabilities)
        # from the new beam after this iteration(Redude computations)
        top_candidates = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:n_candidates_per_beam_length]

        beams = dict(top_candidates)

        new_beams.clear()

    # ==================Final beams=====================================
    # Beam with the highest probability is the most possible optimal tour
    predicted_optimal_tour = sorted(beams.items(), key=lambda t: t[1], reverse=True)[0][
        0
    ]  # [(tour1, prob1), (tour2, prob2)... ]

    return predicted_optimal_tour


def load_gnn_model(checkpoint_path, device="auto", **model_params):
    """Load saved GNN model with automatic device handling."""
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = TSPGNN(**model_params)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Return model + optionally other saved data
    return {
        "model": model,
        "optimizer": checkpoint.get("optimizer_state_dict"),
        "loss": checkpoint.get("loss"),
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
    }


def train_with_hyperparams(
    model: nn.Module,
    train_loader: torch_geometric.loader.DataLoader,
    val_loader: torch_geometric.loader.DataLoader,
    device: torch.device,
    # Hyperparameters
    optimizer_name: str = "AdamW",
    criterion: Optional[Callable] = None,
    lr: float = 1e-4,
    pos_weight: float = 3.0,
    num_epochs: int = 20,
    # Training options
    early_stopping_patience: int = 5,
    gradient_clip: float = 1.0,
    print_every: int = 100,
) -> Dict[str, List[float]]:
    """
    Trains the model with hyperparameters using BCEWithLogitsLoss.

    Args:
        model: GNN model to train.
        train_dataset: Dataset for training.
        val_loader: Dataset for validation at the end of each epoch
        device: CUDA or CPU.
        optimizer_name: "Adam", "AdamW", "SGD", or "RMSprop".
        lr: Learning rate.
        pos_weight: Weight for positive class. Note that high values in pos_weight benefit recall against precison.
        num_epochs: Number of training epochs.
        early_stopping_patience: Stop if no improvement after N epochs.
        gradient_clip: Max gradient norm.
        print_every: Print metrics every N batches.

    Returns:
        Dictionary of metrics (loss, precision, recall, F1, ROC-AUC).
    """

    model = model.to(device)

    # Optimizer selection
    optimizers = {
        "Adam": Adam(model.parameters(), lr=lr),
        "AdamW": AdamW(model.parameters(), lr=lr, weight_decay=1e-5),
        "SGD": SGD(model.parameters(), lr=lr, momentum=0.9),
        "RMSprop": RMSprop(model.parameters(), lr=lr, alpha=0.99),
    }
    optimizer = optimizers[optimizer_name]

    if criterion:
        criterion = criterion
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

        # Metrics tracking
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_precision": [],
        "val_precision": [],
        "train_recall": [],
        "val_recall": [],
        "train_roc_auc": [],
        "val_roc_auc": [],
        "train_pr_auc": [],
        "val_pr_auc": [],
    }
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_preds = []
        train_labels = []
        train_probs = []
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader, 1):
            batch = batch.to(device)
            # Prevent accumulation of gradient descents
            optimizer.zero_grad()

            # Forward pass
            logits = model(batch)
            loss = criterion(logits, batch.y.float())

            # Backward pass
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            # Update of weights
            optimizer.step()

            # Collect metrics
            probs = torch.sigmoid(logits).detach()
            preds = (probs > 0.5).float()

            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(batch.y.cpu().numpy())
            train_probs.extend(probs.cpu().numpy())
            train_loss += loss.item()

            # Progress update
            if batch_idx % print_every == 0:
                batch_precision = precision_score(batch.y.cpu(), preds.cpu(), zero_division=0)
                batch_recall = recall_score(batch.y.cpu(), preds.cpu(), zero_division=0)
                print(
                    f"Epoch {epoch:02d} | Batch {batch_idx:03d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Prec: {batch_precision:.2%} | Rec: {batch_recall:.2%}"
                )

        # ===== Validation Phase For each epoch=====
        model.eval()
        val_preds, val_labels, val_probs = [], [], []
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = criterion(logits, batch.y.float())

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                # Collect results from validation set
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch.y.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                val_loss += loss.item()

        # ===== Metrics Calculation =====
        # Training metrics
        train_loss /= len(train_loader)
        train_precision = precision_score(train_labels, train_preds, zero_division=0)
        train_recall = recall_score(train_labels, train_preds, zero_division=0)
        train_roc_auc = roc_auc_score(train_labels, train_probs)
        train_pr_auc = average_precision_score(train_labels, train_probs)

        # Validation metrics
        val_loss /= len(val_loader)
        val_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_roc_auc = roc_auc_score(val_labels, val_probs)
        val_pr_auc = average_precision_score(val_labels, val_probs)

        # ===== Save ALL Metrics =====
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_precision"].append(train_precision)
        metrics["val_precision"].append(val_precision)
        metrics["train_recall"].append(train_recall)
        metrics["val_recall"].append(val_recall)
        metrics["train_roc_auc"].append(train_roc_auc)
        metrics["val_roc_auc"].append(val_roc_auc)
        metrics["train_pr_auc"].append(train_pr_auc)
        metrics["val_pr_auc"].append(val_pr_auc)

        # ===== Progress Reporting =====
        print(f"\nEpoch {epoch:02d} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Prec: {train_precision:.2%} | Val Prec: {val_precision:.2%}")
        print(f"Train Rec: {train_recall:.2%} | Val Rec: {val_recall:.2%}")
        print(f"Train ROC-AUC: {train_roc_auc:.4f} | Val ROC-AUC: {val_roc_auc:.4f}")
        print(f"Train PR-AUC: {train_pr_auc:.4f} | Val PR-AUC: {val_pr_auc:.4f}")
        print("-" * 80)
        # Early stopping
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}!")
                break

    return metrics


def optimal_tour_as_tuples(optimal_tour: tuple[int]) -> tuple[tuple[int, int]]:
    optimal_tour = islice(cycle(optimal_tour), len(optimal_tour) + 1)
    optimal_tour_as_tuples = [sorted(edge) for edge in pairwise(optimal_tour)]
    return optimal_tour_as_tuples


def calculate_distances(coords: list[list[float, float]]) -> pd.Series:
    n_points = len(coords)
    idxs = list(combinations(range(n_points), r=2))
    distances = pd.Series(data=pdist(coords, metric="euclidean"), index=idxs)
    return distances


def calculate_tour_cost(distances, tour: list[int]):
    mask = optimal_tour_as_tuples(tour)
    return distances[mask].sum()
