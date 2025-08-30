# In[0]: Imports
import os
from typing import Callable, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GATConv

# In[1] Model Definition
# W -> (F X F_prime)
# A -> (2*F_prime X 1)

__all__ = ["TSPGNN", "train_tsp_gnn", "FocalLoss"]


class TSPGNN(torch.nn.Module):
    """Edge-prediction GNN for TSP.

    Args:
        node_dim (int): Dummy node feature dimension (set to 1 since we ignore coordinates).
        edge_dim (int): Edge feature dimension (1 for distances).
        hidden_dim (int): Hidden layer dimension. Default: 64.
        num_heads (int): Number of GAT attention heads. Default: 4.
    """

    def __init__(self, node_dim=2, edge_dim=16, hidden_dim=64, num_heads=4):
        super().__init__()

        # Node encoder to capture the spatial information of the node (relative position) -> (N X hidden_dim)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), nn.LeakyReLU(0.2), nn.LayerNorm(hidden_dim)  # 2D coordinates → hidden_dim
        )

        # Edge feature transformation (distance encoding):
        # Why we need this:
        # 1. Raw distances may not have linear relationship with edge importance
        #    - Very short distances may be exponentially more important than medium ones
        #    - Very long distances may be completely irrelevant
        # 2. Allows learning different distance thresholds for different graphs
        # 3. Projects scalar distances into a more expressive feature space
        # Output -> (E x edge_dim)
        # More expressive edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, 32),  # Wider network
            nn.LeakyReLU(0.2),
            nn.Linear(32, edge_dim),  # Higher output dimension
            nn.LayerNorm(edge_dim),  # Added normalization
        )

        # First GAT Layer:
        # Input shapes:
        #   - Node features: (N x node_dim)
        #   - Edge indices: (2 x E)
        #   - Edge attributes: (E x edge_dim)
        #
        # Transformations:
        # 1) Linear projection: (N x node_dim) -> (N x hidden_dim * num_heads)
        #    - Each head gets hidden_dim features
        # 2) Attention mechanism computes attention coefficients:
        #    - For each edge (i,j), computes attention score using:
        #      [h_i || h_j] @ a where a is (2*hidden_dim x 1)
        #    - Applies LeakyReLU and softmax normalization
        # 3) Aggregation:
        #    - For each node, weighted sum of neighbors' projected features
        #    - Output: (N x hidden_dim * num_heads)
        self.conv1 = GATConv(hidden_dim, hidden_dim, edge_dim=edge_dim, heads=num_heads, concat=True, dropout=0.4)

        self.norm1 = nn.LayerNorm(hidden_dim * num_heads)

        # Second GAT Layer:
        # Input shapes:
        #   - Node features: (N x hidden_dim * num_heads)
        #   - Edge indices: (2 x E)
        #   - Edge attributes: (E x edge_dim)
        #
        # Transformations:
        # 1) Linear projection: (N x hidden_dim*num_heads) -> (N x hidden_dim)
        #    - Single head output this time
        # 2) Same attention mechanism as first layer but with:
        #    - a is now (2*hidden_dim x 1)
        # 3) Aggregation outputs: (N x hidden_dim)
        self.conv2 = GATConv(
            in_channels=hidden_dim * num_heads,
            out_channels=hidden_dim,
            edge_dim=edge_dim,
            heads=1,  # Single head
            concat=False,  # No concatenation (just average heads)
            dropout=0.3,
        )

        self.norm2 = nn.LayerNorm(hidden_dim)

        # Edge Prediction MLP:
        # Input shapes for each edge:
        #   - Concatenated [h_i, h_j, edge_attr]: (2*hidden_dim + edge_dim)
        # Transformations:
        # 1) Linear: Input: (hidden_dim + edge_dim) -> Output: (hidden_dim)
        # 2) ReLU activation
        # 3) Linear: (hidden_dim) -> (1)
        # 4) Sigmoid activation for probability
        # 5) Output: (E, 1)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        # Initialize node features
        # self.node_embed: (1 x node_dim)
        # Expanded to: (N x node_dim)
        # x = self.node_embed.expand(data.num_nodes, -1).to(data.edge_index.device)
        x = self.node_encoder(data.x)  # [N, 2] -> [N, hidden_dim]
        edge_attr = data.edge_attr.unsqueeze(1)  # Shape [190] -> [190, 1]
        edge_attr = self.edge_encoder(edge_attr)  # # [E] -> [E, edge_dim] - [E, 16]

        # First GAT Layer
        # Input x: (N x node_dim)
        # Output x: (N x hidden_dim * num_heads)
        x = self.conv1(x, data.edge_index, edge_attr)
        x = self.norm1(F.leaky_relu(x, 0.1))  # Apply norm
        # Second GAT Layer
        # Input x: (N x hidden_dim * num_heads)
        # Output x: (N x hidden_dim)
        x = self.conv2(x, data.edge_index, edge_attr)
        x = self.norm2(F.leaky_relu(x, 0.1))  # Apply norm
        # Prepare edge features
        # row, col: (E,) each
        # x[row]: (E x hidden_dim)
        # x[col]: (E x hidden_dim)
        # data.edge_attr: (E x edge_dim)
        # edge_emb: (E x (2*hidden_dim + edge_dim))
        row, col = data.edge_index
        edge_emb = torch.cat([x[row], x[col], edge_attr], dim=1)

        # Edge predictions
        # Output: (E,) (squeezed from (E x 1))
        return self.edge_mlp(edge_emb).squeeze()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance by downweighting easy examples
    and focusing training on hard misclassified instances.

    Args:
        alpha (float, optional): Weighting factor for the positive class (default: 0.25).
            Set to `1 / pos_weight` to balance classes (e.g., alpha=0.2 for pos_weight=5).
        gamma (float, optional): Focusing parameter (default: 2.0).
            Higher gamma reduces loss contribution from well-classified examples.
        reduction (str, optional): Loss aggregation method. One of:
            - 'mean' (default): Mean loss over all examples.
            - 'sum': Sum loss over all examples.
            - 'none': Return per-example loss.

    Shape:
        - Input (logits): (N, *) where * means any number of additional dimensions.
        - Target (labels): (N, *), same shape as input.
        - Output: Scalar if reduction='mean' or 'sum'; tensor otherwise.

    Example:
        >>> criterion = FocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = torch.randn(10, requires_grad=True)
        >>> labels = torch.randint(0, 2, (10,))
        >>> loss = criterion(logits, labels)
        >>> loss.backward()
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def train_tsp_gnn(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: str = "checkpoints",
    # Hyperparameters
    lr: float = 1e-3,
    pos_weight: float = 3.0,
    num_epochs: int = 20,
    criterion: Callable | None = None,  # Training options
    early_stopping_patience: int = 5,
    gradient_clip: float = 1.0,
    print_every: int = 100,
    warmup_epochs: int = 2,  # New: For LR warmup
) -> Dict[str, List[float]]:
    """
    Enhanced training loop with:
    - Learning rate warmup
    - ReduceLROnPlateau scheduling
    - Model checkpointing
    - Full metrics tracking
    """

    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)

    optimizer = RMSprop(model.parameters(), lr=lr, alpha=0.99)

    # LR warmup scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: min(1.0, (epoch + 1) / warmup_epochs))

    # Main LR scheduler
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=2,
    )

    # Default Loss function
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
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        # ===== Training Phase =====
        model.train()
        train_preds, train_labels, train_probs = [], [], []
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader, 1):
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            logits = model(batch)
            loss = criterion(logits, batch.y.float())

            # Backward pass
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            # Collect results from training set
            probs = torch.sigmoid(logits).detach()
            preds = (probs > 0.5).float()

            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(batch.y.cpu().numpy())
            train_probs.extend(probs.cpu().numpy())
            train_loss += loss.item()

            # ====== Progress reporting during the epoch ========
            if batch_idx % print_every == 0:
                print(f"Train Epoch {epoch:02d} | Batch {batch_idx:03d} | Loss: {loss.item():.4f}")
        # ===== End of one Forward And Backward Pass For All Batches (ALL TRAINING SAMPLES) =====
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

        # ===== Learning Rate Scheduling =====
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            lr_scheduler.step(val_loss)
            new_lr = lr_scheduler.optimizer.param_groups[0]["lr"]
            print(f"Learning rate updated to: {new_lr:.2e}")

        # ===== Checkpointing (Best Model)=====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss,
                    "metrics": metrics,
                },
                os.path.join(save_dir, "best_model.pth"),
            )
        else:
            patience_counter += 1

        # ===== Early Stopping =====
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}!")
            break

        # ===== Progress Reporting for each epoch =====
        print(f"\nEpoch {epoch:02d} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Prec: {train_precision:.2%} | Val Prec: {val_precision:.2%}")
        print(f"Train Rec: {train_recall:.2%} | Val Rec: {val_recall:.2%}")
        print(f"Train ROC-AUC: {train_roc_auc:.4f} | Val ROC-AUC: {val_roc_auc:.4f}")
        print(f"Train PR-AUC: {train_pr_auc:.4f} | Val PR-AUC: {val_pr_auc:.4f}")
        print("-" * 80)

    # Save final model
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_loss,
            "metrics": metrics,
        },
        os.path.join(save_dir, "final_model.pth"),
    )

    return metrics
