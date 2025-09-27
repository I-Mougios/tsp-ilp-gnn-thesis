# Solving the Traveling Salesman Problem with Graph Neural Networks

> 🧠 Master's thesis project | 🔍 Research meets Engineering

This project combines **Graph Neural Networks (GNNs)** with **Integer Linear Programming (ILP)** to learn heuristics for the **Traveling Salesman Problem (TSP)**. It leverages **PyTorch Geometric** for graph-based deep learning and **SciPy** for exact optimization to create an end-to-end ML pipeline.

By generating optimal TSP solutions via ILP, this work trains a GNN model to approximate those solutions efficiently, bridging classical optimization with modern machine learning.

---

## 📁 Project Structure

- `gnn/`: GNN model, PyTorch Geometric dataset, and utilities  
- `tsp/`: ILP-based solver to generate optimal ground truth tours  
- `data_generation_scripts/`: Docker setu[ for MongoDB and scripts to generate solutions to TSP problems  
- `jobs.py/`: Modular scripts for training, inference, and data management  
- `configs/`: Configuration management using Python scripts and INI files  
- `pytutils/`: Internal utilities for configuration and common functions

---

## 🚀 Getting Started

### Prerequisites

- Docker & Docker Compose  
- Conda (recommended) or Python 3.10+ environment  
- Git 

### Setup

1. Clone the repo  
```bash
    git clone https://github.com/I-Mougios/tsp-ilp-gnn-thesis.git
    cd tsp-gnn-thesis
```
2. Replicate the virtual environment
```bash
    conda env create -f environment.yml
    conda activate tsp
```
3. Spun up MongoDB using the official image
```bash
    docker-compose up --build
 ```

### 🧮 Integer Linear Programming (ILP) Solver
The tsp/solver.py module formulates the TSP as an ILP problem solved using scipy.optimize.linprog. It generates optimal TSP tours that serve as labeled ground truth for supervised GNN training.

Key features:

- Formulates subtour elimination constraints(Branch & Cut) to ensure valid tours

- Solves optimally for small to medium-sized TSP instances

- Provides edge-level labels for GNN training targets

---

### 🧠 Graph Neural Network (GNN) Model
Implemented in gnn/model.py, this GNN leverages PyTorch Geometric to learn from graphs labeled by the ILP solver.

The model is designed to predict which edges in a fully connected graph belong to the optimal TSP tour, using both node and edge-level features.

🔍 Architecture Overview
Node Encoder: Projects raw node indices into an embedding space

Edge Encoder: Encodes Euclidean distances (edge weights)

Core GNN: Two stacked GATConv layers (Graph Attention Networks)

Output: Edge classification (probability of being part of the optimal tour)

✅ Highlights
Processes fully connected graphs representing TSP nodes

Relies on learned embeddings for nodes and distances for edges

Learns to predict edges belonging to the optimal tour

Attempts efficient approximate inference on unseen graphs

---

### 🛠️ Training & Inference Pipeline
Scripts inside jobs/ handle the full pipeline:

- Data generation: Create solved TSP instances using ILP and store them in a Mongo collection

- Data Loading: Load and pre-process data using PyTorch Geometric Dataset & Dataloader

- Training: Supervised learning using the generated data

- Load model: Load the best model

- Prediction: Inference on new graphs to approximate TSP solutions

- Evaluation: Compare GNN outputs against ILP baselines

MongoDB is used to store datasets, leveraging the Docker setup for reproducibility.

---

### 🔧 Technologies and Libraries Used

- Python 3.10+

- PyTorch & PyTorch Geometric

- SciPy (Linear Programming)

- MongoDB & Docker

- Configuration via Python + INI

- Pre-commit hooks, Black, isort, Flake8 for code quality