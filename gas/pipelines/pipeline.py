from abc import ABC, abstractmethod
from pathlib import Path

from deepeval.models import DeepEvalBaseLLM

from gas.commons import TaskType
from gas.providers import DataProvider


class Pipeline(ABC):
    def __init__(self, model: DeepEvalBaseLLM, task: str, test_cases_path: str, *args, **kwargs):
        self.data_provider = DataProvider()
        self.task = TaskType(task)
        self.test_cases_path = Path(test_cases_path)
        self.model = model

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the pipeline."""
        pass
