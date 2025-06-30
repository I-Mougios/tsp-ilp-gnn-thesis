from abc import ABC, abstractmethod
from typing import Any

from pyutils.logger import logger

"""
1. Without ABC - no enforcement of required methods
2.   Meaning all classes inheriting from BaseJob must habe the run method
3.   job_name method is a utility method to be shared across all jobs.
     It returns the module_name e.g. jobs.load_tsp_dataset -> load_tsp_dataset
"""


class BaseJob(ABC):
    """Abstract base class for all jobs"""

    def __init__(self):
        self.logger = logger

    @abstractmethod
    def run(self) -> Any:
        """Main execution to be implemented by subclasses"""
        pass

    @classmethod
    def job_name(cls) -> str:
        """Returns the job's module_name"""
        return cls.__module__.split(".")[-1]
