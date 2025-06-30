# jobs/load_model.py
from gnn.utils import load_gnn_model

from .base_job import BaseJob


class GNNLoader(BaseJob):
    def __init__(self, path: str, device="auto", **model_params):
        super().__init__()
        self.path = path
        self.device = device
        self.model_params = model_params

    def run(self):
        self.logger.info(f"Loading model from: {self.path}")
        return load_gnn_model(checkpoint_path=self.path, device=self.device, **self.model_params)


default_job = GNNLoader
