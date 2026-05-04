import csv
import json
import os
from glob import glob
from pathlib import Path
from typing import Any

from gas.commons import TaskType
from gas.logger import Logger

logging = Logger()
logger = logging.get_logger()


class DataProvider:
    def __init__(self):
        # Load the JSON content into a dictionary
        json_path = Path().cwd() / "gas" / "data" / "geobench" / "geobenchmark_npee.json"
        with Path.open(json_path, encoding="utf-8") as json_file:
            data = json.load(json_file)

        # Create separate dictionaries for each key
        self._tasks_datasets: dict[TaskType, dict[str, list[str]]] = {}

        for type in TaskType:
            self._tasks_datasets[type] = {
                "question": data[type.value.lower()]["question"],
                "answer": data[type.value.lower()]["answer"],
            }

    def fetch_dataset(self, task_type: TaskType) -> dict[str, list[str]]:
        """Fetches the dataset for the specified task type.

        Args:
            task_type (TaskType): The type of task for which to fetch the dataset.

        Returns:
            dict[str, list[str]]: A dictionary containing the dataset for the specified task type.
        """
        return self._tasks_datasets[task_type]

    def find_last_file(self, directory: Path | str, file_pattern: str = "*"):
        """
        find the latest file based on timestamp in the name or modification time.

        Args:
            directory (str): Directory path where files are stored.
            file_pattern (str): Pattern to match filenames (e.g., "*.txt", "*.log").

        Returns:
            str: The path of the latest file.
        """
        # List files matching the pattern
        files = glob(f"{directory}/{file_pattern}")  # noqa: PTH207

        if not files:
            logger.error("No files found matching the pattern.")
            raise FileNotFoundError()

        # Sort files by their modification time
        latest_file = max(files, key=os.path.getmtime)

        logger.debug(f"Latest file path: {latest_file}")
        return latest_file

    def save(self, records: list[dict[str, Any]], output_file: Path, format: str = "json") -> Path:
        """
        Save records to a file in JSON or CSV format.

        Args:
            records (list[dict[str, Any]]): List of dictionaries to save.

        Returns:
            Path: Path to the saved file.
        """
        # Create the directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(file=output_file, mode="w", newline="", encoding="utf-8") as file:
            if format == "json":
                json.dump(records, file, indent=4, ensure_ascii=False)
            elif format == "csv":
                writer = csv.DictWriter(file, fieldnames=records[0].keys(), delimiter=";")
                writer.writeheader()
                writer.writerows(records)

        return output_file
