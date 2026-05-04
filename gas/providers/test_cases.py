from os import PathLike
from pathlib import Path

from deepeval.dataset import EvaluationDataset
from deepeval.models import DeepEvalBaseLLM

from gas.logger import Logger, MyProgress

from .prompt import PromptProvider

logging = Logger()
logger = logging.get_logger()


class TestCasesProvider:
    @staticmethod
    def bulk_create(
        model: DeepEvalBaseLLM,
        prompt_instruction: str,
        dataset: dict[str, list[str]],
        limit: int | None,
    ) -> list[dict[str, str]]:
        """
        Create a list containing LLM Test Cases with generated output.
        Args:
            model (DeepEvalBaseLLM): The model to evaluate.
            prompt_instruction (str): prompt with task instruction
            dataset (dict[str,str]): dataset with input and expected output
            limit (int): total tests limit
        Returns:
            A list containing the LLM test cases with generated output.
        """
        model.load_model()
        test_cases: list[dict[str, str]] = []
        total = limit if limit else len(dataset["question"])

        with MyProgress(None) as progress:
            task = progress.add_task(
                f"[bold bright_green]Generating {total} Test Cases:[/bold bright_green]", total=total
            )
            progress.log(f"Model Name: {model.get_model_name()}")
            for i in range(total):
                question = dataset["question"][i]
                expected_output = dataset["answer"][i]

                messages = PromptProvider.build_messages(
                    prompt_instruction, question, apply_chat_template=model.should_apply_chat_template
                )

                tc = {
                    "input": question,
                    "expected_output": expected_output,
                    "actual_output": model.generate(messages),
                }

                test_cases.append(tc)

                if logging.get_level() == "DEBUG":
                    logging.print_table_panel(
                        tc,
                        console=progress.console,
                        title=f"[bold bright_green]DEBUG Test Case {i + 1} Result[/bold bright_green]",
                        border_style="green",
                        justify="full",
                    )

                progress.update(task, advance=1)
        return test_cases

    @staticmethod
    def fetch(path: str | PathLike) -> EvaluationDataset:
        """
        Fetch the test cases from a JSON file.
        The test cases should be in the following format:

        ..code-block:: json
        [
            {
                "input": "input text",
                "actual_output": "actual output",
                "expected_output": "expected output"
            },
            ...
        ]

        Args:
            path (str | PathLike): path to the JSON file containing test cases

        Returns:
            EvaluationDataset: An EvaluationDataset object containing the test cases.
        """
        full_file_path = Path(path).resolve()
        dataset = EvaluationDataset()

        dataset.add_test_cases_from_json_file(
            file_path=full_file_path,
            input_key_name="input",
            actual_output_key_name="actual_output",
            expected_output_key_name="expected_output",
        )

        return dataset
