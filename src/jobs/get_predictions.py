# jobs/get_predictions.py
from collections import defaultdict

import torch
import torch_geometric

from gnn.utils import beamsearch, calculate_distances, calculate_tour_cost

from .base_job import BaseJob


class TSPPredictor(BaseJob):
    def __init__(
        self,
        model: torch.nn.Module,
        data: torch_geometric.loader.DataLoader,
        beam_size: int,
        n_candidates_per_beam_length: int,
        device: torch.device,
    ):
        super().__init__()
        self.model = model
        self.data = data
        self.beam_size = beam_size
        self.n_candidates_per_beam_length = n_candidates_per_beam_length
        self.device = device

    def run(self):
        self.logger.info("Making predictions ... ")
        self.model.eval()
        results = defaultdict(list)
        with torch.no_grad():  # Disable gradient tracking
            for batch in self.data:
                for i in range(len(batch)):
                    graph = batch[i]
                    graph = graph.to(self.device)

                    logits = self.model(graph)
                    probabilities = torch.sigmoid(logits)

                    predicted_optimal_tour = beamsearch(
                        probabilities=probabilities,
                        edge_index=graph.edge_index,
                        number_of_nodes=graph.num_nodes,
                        beam_size=self.beam_size,
                        n_candidates_per_beam_length=self.n_candidates_per_beam_length,
                    )

                    distances = calculate_distances(coords=graph.x)

                    cost_of_predicted_tour = calculate_tour_cost(distances, predicted_optimal_tour)
                    cost_of_optimal_tour = calculate_tour_cost(distances, graph.optimal_tour)
                    deviation = cost_of_predicted_tour - cost_of_optimal_tour

                    results["number_of_points"].append(graph.num_nodes)
                    results["cost_of_predicted_tour"].append(cost_of_predicted_tour)
                    results["cost_of_optimal_tour"].append(cost_of_optimal_tour)
                    results["deviation"].append(deviation)
                    results["relative_deviation"].append(deviation / cost_of_optimal_tour)

        return results


default_job = TSPPredictor
