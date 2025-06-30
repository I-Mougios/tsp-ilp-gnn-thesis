# In[0]: Imports
import os
from contextlib import redirect_stdout
from datetime import datetime

import pandas as pd
import torch

from configs import Configs, bool_
from gnn.gat import TSPGNN
from jobs.orchestrator import orchestrator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = Configs.training.get("save_dir", default="checkpoints", cast=str)
os.makedirs(save_dir, exist_ok=True)

# In[2]: Load TSPDataset and get the default dataloaders
dataloaders = orchestrator.run_job(
    job_name="load_tsp_dataset",
    normalize_edges=Configs.dataset.get("normalize_edges", cast=bool_),
    num_samples=Configs.dataset.get("num_samples", cast=int),
    get_default_dataloaders=Configs.dataset.get("get_default_dataloaders", cast=bool_),
    test_size=Configs.dataset.get("test_size", cast=float),
    validation_size=Configs.dataset.get("validation_size", cast=float),
    batch_size=Configs.dataset.get("batch_size", cast=int),
    shuffle=Configs.dataset.get("shuffle", cast=bool_),
    random_state=Configs.dataset.get("random_state", cast=int),
    stratify=Configs.dataset.get("stratify", cast=bool_),
)

train_set = dataloaders.train_data
validation_set = dataloaders.validation_data
test_set = dataloaders.test_data

# In[3]: Define the GNNModel and start the training loop
model = TSPGNN(
    node_dim=Configs.model.get("node_dim", cast=int),
    edge_dim=Configs.model.get("edge_dim", cast=int),
    hidden_dim=Configs.model.get("hidden_dim", cast=int),
    num_heads=Configs.model.get("num_heads", cast=int),
)

now = datetime.now().strftime("%Y%m%d_%H%M%S")
training_log_path = os.path.join(save_dir, f"training_progress_{now}.txt")

with redirect_stdout(open(training_log_path, mode="w")):
    metrics = orchestrator.run_job(
        job_name="train_gnn_model",
        model=model,
        train_loader=train_set,
        val_loader=validation_set,
        device=device,
        save_dir=save_dir,
        lr=Configs.training.get("lr", default=1e-3, cast=float),
        pos_weight=Configs.training.get("pos_weight", default=3.0, cast=float),
        num_epochs=Configs.training.get("num_epochs", default=20, cast=int),
        criterion=None,  # or pass a callable if specified in config
        early_stopping_patience=Configs.training.get("early_stopping_patience", default=5, cast=int),
        gradient_clip=Configs.training.get("gradient_clip", default=1.0, cast=float),
        print_every=Configs.training.get("print_every", default=100, cast=int),
        warmup_epochs=Configs.training.get("warmup_epochs", default=2, cast=int),
    )

# In[4]: Load best model and make predictions

model_path = os.path.join(save_dir, "best_model.pth")
model_dict = orchestrator.run_job(
    "load_model",
    path=model_path,
    device=device,
    node_dim=Configs.model.get("node_dim", cast=int),
    edge_dim=Configs.model.get("edge_dim", cast=int),
    hidden_dim=Configs.model.get("hidden_dim", cast=int),
    num_heads=Configs.model.get("num_heads", cast=int),
)
model = model_dict["model"]
print(model)

# In[5]: Model predictions
results = orchestrator.run_job(
    "get_predictions",
    model=model,
    data=test_set,
    beam_size=Configs.model.get("beam_size", cast=int),
    n_candidates_per_beam_length=Configs.model.get("n_candidates_per_beam_length", cast=int),
    device=device,
)

# In[5]: Result in Dataframe
if "now" not in globals():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

results_csv = os.path.join(save_dir, f"results_{now}.csv")
results_df = pd.DataFrame(results)
results_df.to_csv(results_csv, index=False)
