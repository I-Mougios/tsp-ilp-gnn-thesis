# jobs/load_tsp_dataset.py
from dataclasses import dataclass
from typing import Callable, Optional

from pymongo import ASCENDING, MongoClient
from torch_geometric.loader import DataLoader

from configs import Configs
from gnn.dataset import TSPDataset

from .base_job import BaseJob


@dataclass
class TSPDataloaders:
    train_data: DataLoader
    validation_data: DataLoader
    test_data: DataLoader


class TSPDatasetLoader(BaseJob):
    """Loads TSP dataset from MongoDB with configurable sampling"""

    def __init__(
        self,
        transform: Optional[Callable] = None,
        normalize_edges: bool = True,
        num_samples: Optional[int] = None,
        get_default_dataloaders: bool = True,
        test_size: int = 0.2,
        validation_size: int = 0.1,
        batch_size: int = 16,
        shuffle: bool = True,
        random_state: int = 42,
        stratify: bool = True,
    ):

        super().__init__()
        self.transform = transform
        self.normalize_edges = normalize_edges
        self.num_samples = num_samples
        self.get_default_dataloaders = get_default_dataloaders
        self.test_size = test_size
        self.validation_size = validation_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratify = stratify

    def _get_mongo_client(self) -> MongoClient:
        """Create authenticated MongoDB client"""
        try:
            return MongoClient(
                host=Configs.mongodb.host,
                port=Configs.mongodb.get("port", cast=int),
                username=Configs.mongodb.username,
                password=Configs.mongodb.password,
                authSource=Configs.mongodb.get("auth_db", "admin"),
            )
        except Exception as e:
            self.logger.error(f"MongoDB connection failed: {str(e)}")

    def run(self) -> TSPDataset:
        """Main execution method"""
        self.logger.info("Loading TSP dataset from MongoDB")

        try:
            client = self._get_mongo_client()
            db = client[Configs.mongodb.database]
            collection = db[Configs.mongodb.collection]

            query = {}  # Can add filters here
            docs = list(collection.find(query).sort("_id", ASCENDING))

            dataset = TSPDataset(docs, normalize_edges=self.normalize_edges, num_samples=self.num_samples)

            if self.get_default_dataloaders:
                dataloaders = dataset.get_dataloaders(
                    test_size=self.test_size,
                    val_size=self.validation_size,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    random_state=self.random_state,
                    stratify=self.stratify,
                )
                return TSPDataloaders(*dataloaders)

            return dataset

        except Exception as e:
            self.logger.error(f"Dataset loading failed: {str(e)}")
            raise
        finally:
            if "client" in locals():
                client.close()


# Default instance for easy import
default_job = TSPDatasetLoader
