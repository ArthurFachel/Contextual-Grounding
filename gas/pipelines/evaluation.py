"""
Custom evaluation pipeline that bypasses DeepEval's cloud runner entirely.
Uses Gemini as judge via direct metric.measure() calls — no Confident AI, no API key needed.
"""

import json
import os
from pathlib import Path
from typing import Any

import google.generativeai as genai
from deepeval.metrics import AnswerRelevancyMetric, GEval, PromptAlignmentMetric
from deepeval.metrics.base_metric import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from gas.commons import TaskType
from gas.logger import Logger
from gas.metrics import ObjectiveAccuracyMetric
from gas.pipelines.pipeline import Pipeline
from gas.providers import PromptProvider, TestCasesProvider

logging = Logger()
logger = logging.get_logger()



class GeminiJudge(DeepEvalBaseLLM):
    """
    Thin wrapper around google-generativeai that implements DeepEvalBaseLLM.
    Used as the judge model for all LLM-based metrics without any cloud upload.
    """

    def __init__(self, model_name: str = "gemma-3-27b-it", api_key: str | None = None):
        self._model_name = model_name
        api_key = api_key or os.environ.get("GEMINI_KEY")
        if not api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_KEY env var or pass api_key.")
        genai.configure(api_key=api_key)
        self._client = genai.GenerativeModel(model_name)

    def load_model(self):
        return self._client

    def generate(self, prompt: str) -> str:
        response = self._client.generate_content(prompt)
        return response.text

    async def a_generate(self, prompt: str) -> str:
        # DeepEval metrics call a_generate for async paths; fall back to sync
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self._model_name



class MetricResult:
    def __init__(self, name: str, score: float, success: bool, reason: str | None = None):
        self.name = name
        self.score = score
        self.success = success
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "success": self.success,
            "reason": self.reason,
        }



def run_metrics_locally(
    test_cases: list[LLMTestCase],
    metrics: list[BaseMetric],
) -> list[dict[str, Any]]:
    """
    Evaluate each test case against each metric without using DeepEval's
    evaluate() runner. This avoids all Confident AI / cloud upload logic.

    Returns:
        A list of per-test-case result dicts.
    """
    results = []

    for i, tc in enumerate(test_cases):
        tc_results: list[dict[str, Any]] = []
        logger.info(f"Evaluating test case {i + 1}/{len(test_cases)}")

        for metric in metrics:
            try:
                metric.measure(tc)
                result = MetricResult(
                    name=metric.__name__,
                    score=float(metric.score),
                    success=bool(metric.is_successful()),
                    reason=getattr(metric, "reason", None),
                )
            except Exception as e:
                logger.warning(f"Metric {metric.__name__} failed on test case {i + 1}: {e}")
                result = MetricResult(
                    name=metric.__name__,
                    score=0.0,
                    success=False,
                    reason=f"ERROR: {e}",
                )

            tc_results.append(result.to_dict())
            logger.debug(
                f"  [{metric.__name__}] score={result.score:.3f} "
                f"success={result.success} reason={result.reason}"
            )

        results.append(
            {
                "input": tc.input,
                "actual_output": tc.actual_output,
                "expected_output": tc.expected_output,
                "metrics": tc_results,
            }
        )

    return results



def build_summary(
    results: list[dict[str, Any]],
    model_name: str,
    judge_name: str,
    task: str,
) -> list[dict[str, Any]]:
    """Aggregate per-test-case results into per-metric summary stats."""
    totals: dict[str, dict[str, float]] = {}

    for tc in results:
        for m in tc["metrics"]:
            name = m["name"]
            if name not in totals:
                totals[name] = {
                    "total_score": 0.0,
                    "total_success": 0.0,
                    "total_tests": 0.0,
                    "total_cost": 0.0,  # kept for schema compatibility
                }
            totals[name]["total_score"] += m["score"]
            totals[name]["total_success"] += 1 if m["success"] else 0
            totals[name]["total_tests"] += 1

    summary = []
    for metric_name, values in totals.items():
        n = values["total_tests"]
        summary.append(
            {
                "model": model_name,
                "judge": judge_name,
                "task": task,
                "metric": metric_name,
                **values,
                "average_score": values["total_score"] / n if n else 0.0,
                "success_rate": values["total_success"] / n if n else 0.0,
            }
        )

    return summary


# ---------------------------------------------------------------------------
# Drop-in replacement EvaluationPipeline
# ---------------------------------------------------------------------------

class EvaluationPipeline(Pipeline):
    """
    Replacement for the original EvaluationPipeline.
    Differences:
      - Uses GeminiJudge directly instead of GeminiModel from DeepEval
      - Calls metric.measure() per test case instead of deepeval.evaluate()
      - Never contacts Confident AI
    """

    def __init__(
        self,
        model_judge: str,
        results_path: str,
        benchmark_model: DeepEvalBaseLLM | None = None,
        gemini_api_key: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_judge_name = model_judge
        self.results_path = Path(results_path)
        self.benchmark_model = benchmark_model or self.model

        self.judge = GeminiJudge(
            model_name=model_judge,
            api_key=gemini_api_key or os.environ.get("GEMINI_KEY"),
        )

    # ------------------------------------------------------------------
    def _fetch_metrics(self) -> list[BaseMetric]:
        if self.task in (TaskType.CHOICE, TaskType.TF):
            return [ObjectiveAccuracyMetric()]

        return [
            PromptAlignmentMetric(
                prompt_instructions=[PromptProvider.fetch_instruction(self.task)],
                include_reason=True,
                model=self.judge,
                threshold=0.5,
            ),
            AnswerRelevancyMetric(
                threshold=0.5,
                model=self.judge,
                include_reason=True,
            ),
            GEval(
                name="Correctness",
                model=self.judge,
                evaluation_steps=[
                    "Compare the actual output directly with the expected output to verify factual accuracy.",
                    (
                        "Check if all elements mentioned in the expected output are present "
                        "and correctly represented in the actual output."
                    ),
                    (
                        "Assess if there are any discrepancies in details, values, or information "
                        "between the actual and expected outputs."
                    ),
                ],
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
            ),
        ]

    def _print_summary(self, summary: list[dict[str, Any]]) -> None:
        def fmt(entry: dict[str, Any]) -> str:
            return (
                f"total_tests: {int(entry['total_tests'])}\n"
                f"total_score: {entry['total_score']:.4f}\n"
                f"average_score: {entry['average_score']:.4f}\n"
                f"success_rate: {entry['success_rate'] * 100:.2f}%"
            )

        logging.print_table_panel(
            {e["metric"]: fmt(e) for e in summary},
            title="[bold bright_green]Benchmark Summary[/bold bright_green]",
            border_style="green",
        )

    def run(self, *args, **kwargs):
        logger.info("Fetching benchmark metrics...")
        metrics = self._fetch_metrics()
        Logger().print_panel(
            "\n".join([m.__name__ for m in metrics]),
            title="[bold medium_orchid]Benchmark Metrics[/bold medium_orchid]",
            border_style="purple",
            justify="center",
        )

        logger.info("Fetching generated test cases...")
        tests_path = (
            Path(self.test_cases_path)
            / f"{self.benchmark_model.get_model_name()}_{self.task.value}.json"
        )
        dataset = TestCasesProvider.fetch(tests_path)
        logger.info(f"Loaded {len(dataset.test_cases)} test cases.")

        logger.info("Running evaluation locally (no cloud upload)...")
        raw_results = run_metrics_locally(dataset.test_cases, metrics)
        logger.info("Done.")

        # Save raw per-test-case results
        raw_out = (
            self.results_path
            / f"raw_{self.benchmark_model.get_model_name()}_{self.task.value}.json"
        )
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_out, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Raw results saved to {raw_out}")

        logger.info("Creating benchmark summary...")
        summary = build_summary(
            raw_results,
            model_name=self.benchmark_model.get_model_name(),
            judge_name=self.model_judge_name,
            task=self.task.value,
        )
        self._print_summary(summary)

        logger.info("Saving summary...")
        summary_out = (
            self.results_path
            / f"summary_{self.benchmark_model.get_model_name()}_{self.task.value}.json"
        )
        self.data_provider.save(summary, summary_out, "json")
        logger.info(f"Summary saved to {summary_out}")