from gas.logger import Logger
from gas.pipelines.pipeline import Pipeline
from gas.providers import PromptProvider, TestCasesProvider

logging = Logger()
logger = logging.get_logger()


class GenerationPipeline(Pipeline):
    def __init__(self, limit: int | None, *args, **kwargs):
        super().__init__(**kwargs)
        self.limit = limit

    def run(self, *args, **kwargs) -> None:
        """
        Method to generate the benchmark test cases.
        """

        logger.info("Starting test cases generation pipeline.")
        logger.info("Fetching Geobench dataset...")
        test_data = self.data_provider.fetch_dataset(self.task)
        logger.info("Done.")
        logger.info("Fetching task instruction...")
        instruction = PromptProvider.fetch_instruction(self.task)
        logger.info("Done.")
        logger.info("Generating test cases...")
        test_cases: list[dict[str, str]] = TestCasesProvider.bulk_create(self.model, instruction, test_data, self.limit)
        logger.info("Done.")
        logger.info("Saving test cases...")
        outfile = self.test_cases_path / f"{self.model.get_model_name()}_{self.task.value}.json"
        path = self.data_provider.save(records=test_cases, output_file=outfile)
        logger.info("Done.")
        logger.info(f"Test cases stored at: {path}")
