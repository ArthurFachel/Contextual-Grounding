import argparse
import os
from typing import Any

import toml
from deepeval.models import DeepEvalBaseLLM
from dotenv import load_dotenv
from deepeval.models import GeminiModel
from gas.logger import Logger
from gas.models import AWSModel, BasePeftModel, DeepInfraModel
from gas.pipelines import EvaluationPipeline, GenerationPipeline
load_dotenv()
logging = Logger()
logger = logging.get_logger()

eval_model = GeminiModel(
    model="gemma-3-27b",
    api_key="AIzaSyBEflssy5YKTYHsn571gTMLZh5T8YX97sU",
    temperature=0
)

def show_info(
    model: DeepEvalBaseLLM,
    task: str,
    seed: int | None,
    generation_params: dict[str, Any],
    limit: int | None = None,
    **kwargs,
) -> None:
    """
    Show information about the benchmark task, model, and generation parameters.
    Args:
        model (DeepEvalBaseLLM): The model to evaluate.
        task (str): The task type.
        seed (int | None): The random seed for reproducibility.
        generation_params (dict[str, Any]): The generation parameters for the model.
        limit (int | None, optional): The limit for the number of test cases to generate. Defaults to None.
        **kwargs: Additional arguments.
    """
    info = {
        "Geoscience benchmark task:": task,
        "Model for evaluation:": model.get_model_name(),
        "LOG LEVEL:": f"{logging.get_level()}",
        "Tests limit:": limit,
        "SEED: ": seed,
        **generation_params,
    }
    logging.print_table_panel(
        info,
        title="[bold medium_orchid]Benchmark General Information[/bold medium_orchid]",
        border_style="purple",
        expand=False,
        justify="center",
    )


def generate(model: DeepEvalBaseLLM, task: str, limit: int | None, test_cases_path: str, **kwargs) -> None:
    """
    Args:
        model (DeepEvalBaseLLM): The model to evaluate.
        task (str): The task type.
        limit (int | None): The limit for the number of test cases to generate.
        test_cases_path (str): Path to the test cases file.
        **kwargs: Additional arguments for the generation pipeline.
    """
    GenerationPipeline(model=model, task=task, test_cases_path=test_cases_path, limit=limit, **kwargs).run()


def evaluate(
    model: BasePeftModel,
    task: str,
    model_judge: str,
    test_cases_path: str,
    results_path: str,
    **kwargs,
) -> None:
    EvaluationPipeline(
        model=eval_model,          # judge (GeminiModel)
        benchmark_model=model,     # model being evaluated ← add this
        task=task,
        test_cases_path=test_cases_path,
        model_judge=model_judge,
        results_path=results_path,
        **kwargs,
    ).run()


def make_shared_commands(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add shared commands to the argument parser.
    Args:
        parser (argparse.ArgumentParser): The argument parser instance.

    Returns:
        argparse.ArgumentParser: The updated argument parser instance with shared commands.
    """
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        required=True,
        dest="task",
        metavar="<TASK>",
        help="Task to evaluate.\nChoices: [%(choices)s]\n\n",
        choices=["NOUN", "CHOICE", "COMPLETION", "TF", "QA"],
        default=None,
    )
    parser.add_argument(
        "-ll",
        "--log-level",
        type=str,
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        metavar="<LEVEL>",
        help="Set the logging level.\nChoices: [%(choices)s]\nDefault: %(default)s\n\n",
        default="INFO",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        dest="config_path",
        metavar="<PATH>",
        help="Path to config file.\nDefault: %(default)s\n\n",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--dotenv",
        type=str,
        dest="dotenv_path",
        metavar="<PATH>",
        help="Path to Dotenv file.\nDefault: %(default)s\n\n",
        default=None,
    )
    return parser


def build_argparser() -> argparse.ArgumentParser:
    """Build the argument parser for the command line interface.

    Returns:
        argparse.ArgumentParser: The argument parser object.
    """
    parser = argparse.ArgumentParser(
        description="GAS: A command line tool for running the Geobenchmark evaluation suite.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser = make_shared_commands(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)

    parent_parser = argparse.ArgumentParser(add_help=False)

    parser_generate = subparsers.add_parser(
        "generate",
        help="Generate the test cases to evaluate.",
        description="Generate test cases for a provided model.",
        parents=[parent_parser],
    )
    parser_generate.add_argument(
        "-l",
        "--limit",
        type=int,
        dest="limit",
        metavar="<LIMIT>",
        help=(
            "Total number of test cases to generate for evaluation.\n" "Default: generates all test cases available\n\n"
        ),
        default=None,
    )
    parser_evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate the generated test cases.",
        description="Evaluate the generated test cases.",
        parents=[parent_parser],
    )
    parser.print_help = lambda: Logger.print_help(parser)  # type: ignore[method-assign, misc, assignment]
    parser_evaluate.print_help = lambda: Logger.print_help(parser_evaluate)  # type: ignore[method-assign, misc, assignment]
    parser_generate.print_help = lambda: Logger.print_help(parser_generate)  # type: ignore[method-assign, misc, assignment]
    return parser


def cli() -> None:
    """Main entry point for the gas command line interface."""
    parser = build_argparser()

    args = parser.parse_args()
    args_dict = vars(args).copy()
    logging.set_level(args_dict.pop("log_level"))

    dotenv_path = args_dict.get("dotenv_path", ".env")
    load_dotenv(dotenv_path=dotenv_path) if dotenv_path else load_dotenv()

    config_path = args_dict.get("config_path", "./config.toml")
    config = toml.load(config_path)

    tc_path = config["environment"]["test_cases_path"]

    key = os.environ.get("AIzaSyBEflssy5YKTYHsn571gTMLZh5T8YX97sU")
    logger.debug(key)

    generation_params = config["generation"]
    model_params = config["model"]
    seed = config["environment"].get("seed", None)
    state = config["environment"].get("state", None)
    if config["environment"].get("bedrock", True) == True:
        model = (
            AWSModel(model_params, generation_params, seed=seed, state=state)
            if not model_params.get("peft", False)
            else BasePeftModel(model_params, generation_params, seed=seed, state=state)
        )
    if config["environment"].get("bedrock", True) == False:
        model = (
            DeepInfraModel(model_params, generation_params, seed=seed , state=state)
            if not model_params.get("peft", False)
            else BasePeftModel(model_params, generation_params, seed=seed, state=state)
        )
    show_info(model=eval_model, seed=seed, generation_params=generation_params, **args_dict)
    match args.command:
        case "generate":
            generate(model=model, test_cases_path=tc_path, **args_dict)
        case "evaluate":
            evaluation_params = config["evaluation"]
            model_judge = evaluation_params.pop("model_judge", "gemini-3-27b")
            evaluate(model=model, model_judge=model_judge, test_cases_path=tc_path, **evaluation_params, **args_dict)
        case _:
            parser.print_help()
    os._exit(0)


if __name__ == "__main__":
    cli()
