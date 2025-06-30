from pathlib import Path

from pyutils import ConfigMeta

root = Path(__file__).parents[1]
configs_dir = root / "configs"
config_filename = "configs.ini"


class Configs(metaclass=ConfigMeta, config_directory=configs_dir, config_filename=config_filename):
    """
    Configurations about:
        - Hyperparamters during training
        - Connection to MongoDB in order to create the TSPDataset
        - Paths where the training logs and predictions will be stored
    """
