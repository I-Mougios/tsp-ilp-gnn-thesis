# jobs/orchestrator.py
import importlib
from typing import Any, Dict, List

from pyutils.logger import logger

from .base_job import BaseJob


class JobOrchestrator:
    """Manages job execution with dependency handling"""

    def __init__(self):
        self.logger = logger
        self.registry: Dict[str, BaseJob] = {}

    def register_job(self, job_class: BaseJob) -> None:
        """
        Register a job class in the registry
        Note:
            1.job class must be a class inherititing from BaseJob, not an instance of BaseJob or a module_name - string
            2.job_name() returns the name of the module for example jobs.load_tsp_dataset -> load_tsp_dataset
        """
        self.registry[job_class.job_name()] = job_class

    def run_job(self, job_name: str, **kwargs) -> Any:
        """
        Execute a single job with error handling
        Notes:
            1. If the JobClass is not registered then it will use the _load_job_module as fallback
            2. If the JobClass is registered using e.g orchestrator.register(TSPDatasetLoader)
            then it will not import the module again
        """
        try:
            self.logger.info(f"Starting job: {job_name}")

            if job_name not in self.registry:
                self._load_job_module(job_name)

            # Create an instance of the job_class and pass all the required arguments to __init__
            job = self.registry[job_name](**kwargs)
            # Execute the run method
            return job.run()

        except Exception as e:
            self.logger.error(f"Job {job_name} failed: {str(e)}")
            raise

    def run_pipeline(self, job_names: List[str]) -> Dict[str, Any]:
        """
        Execute multiple jobs in sequential order.

        Jobs can be specified either by:
        1. Their registered job name (via register_job), or
        2. Their module name (which will be auto-discovered)

        Parameters:
            job_names: A list of identifiers for jobs to execute. Each can be:
                    - A module name in the jobs/ package (must contain default_job)
                    Example: ['load_tsp_dataset', 'preprocess_data']

        Returns:
            A dictionary mapping each job name to its return value, in execution order.

        Raises:
            ImportError: If a module name is provided but cannot be imported
            AttributeError: If a module lacks default_job
            RuntimeError: For job-specific failures during execution

        Example:
            >>> orchestrator.run_pipeline(['load_tsp_dataset', 'train_model'])
            or
            >>> orchestrator.register(TSPDatasetLoader)
            >>> orchestrator.run_pipeline(['load_tsp_dataset']) # in that case the _load_job_module will not be executed
            {'load_tsp_dataset': <TSPDataset>, 'train_model': <TrainedModel>}
        """
        results = {}

        for job_name in job_names:
            results[job_name] = self.run_job(job_name)

        return results

    def _load_job_module(self, module_name: str) -> None:
        """Dynamically import a job module"""
        try:
            # 1. Dynamically import the module
            module = importlib.import_module(f"jobs.{module_name}")
            # 2. Check for a default job class
            if hasattr(module, "default_job"):
                self.register_job(module.default_job)
        except ImportError as e:
            raise ImportError(f"Job module {module_name} not found") from e


# Default orchestrator instance
orchestrator = JobOrchestrator()
