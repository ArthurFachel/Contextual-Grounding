from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class ObjectiveAccuracyMetric(BaseMetric):
    """Objective accuracy metric for evaluating the performance of a model on a specific task."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        """Measure the accuracy of the model's output against the expected output.
        Args:
            test_case (LLMTestCase): The test case containing the model's output and expected output.
        Returns:
            float: The accuracy score of the model's output.
        """
        output = test_case.actual_output.strip().split("\n")[0].split(" ")[0].removesuffix(".")
        self.score = output == test_case.expected_output
        self.success = self.score
        return self.score

    # Async implementation of measure(). If async version for
    # scoring method does not exist, just reuse the measure method.
    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async measure method for evaluating the accuracy of the model's output.
        Args:
            test_case (LLMTestCase): The test case containing the model's output and expected output.

        Returns:
            float: The accuracy score of the model's output.
        """
        return self.measure(test_case)

    def is_successful(self) -> bool:
        """Check if the test case was successful based on the accuracy score.

        Returns:
            bool: True if the test case was successful, False otherwise.
        """
        return self.success

    @property
    def __name__(self) -> str:
        """Name of the metric.
        Returns:
            str: The name of the metric.
        """
        return "Objective Task Accuracy Metric"
