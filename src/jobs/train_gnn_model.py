# jobs/train_gnn_model.py
from time import perf_counter
from typing import Callable

import torch
import torch_geometric

from gnn.gat import train_tsp_gnn

from .base_job import BaseJob


class GNNTrainer(BaseJob):
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch_geometric.loader.DataLoader,
        val_loader: torch_geometric.loader.DataLoader,
        device: torch.device,
        save_dir: str = "checkpoints",
        lr: float = 1e-3,
        pos_weight: float = 3.0,
        num_epochs: int = 20,
        criterion: Callable | None = None,
        early_stopping_patience: int = 5,
        gradient_clip: float = 1.0,
        print_every: int = 100,
        warmup_epochs: int = 2,
    ):
        super().__init__()

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.lr = lr
        self.pos_weight = pos_weight
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip = gradient_clip
        self.print_every = print_every
        self.warmup_epochs = warmup_epochs

    def run(self):
        """Main execution method"""

        self.logger.info("Training GNN model.")
        start = perf_counter()
        metrics = train_tsp_gnn(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device,
            save_dir=self.save_dir,
            lr=self.lr,
            pos_weight=self.pos_weight,
            num_epochs=self.num_epochs,
            criterion=self.criterion,
            early_stopping_patience=self.early_stopping_patience,
            gradient_clip=self.gradient_clip,
            print_every=self.print_every,
            warmup_epochs=self.warmup_epochs,
        )
        end = perf_counter()
        elapsed_time = (end - start) / 60
        self.logger.info(f"Training completed.\nProcess took: {elapsed_time} minutes")

        return metrics


default_job = GNNTrainer
