import argparse
import inspect
import logging
from threading import Lock
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table


class Logger:
    """Logger class for rich console logging."""

    _instance = None
    _lock = Lock()  # Thread-safe initialization

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls, *args, **kwargs)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return  # Prevent re-initialization
        self._initialized = True
        self.console = Console(emoji=False, stderr=False)

        # Configure Rich logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=self.console)],
            encoding="utf-8",
        )
        self.logger = logging.getLogger("rich_logger")
        self.logger.setLevel(logging.DEBUG)  # Default log level
        self.is_verbose = False
        self.panel_params = [
            name for name in inspect.signature(Panel.__init__).parameters if name not in ["self", "kwargs"]
        ]
        self.console_log_params = [
            name for name in inspect.signature(Console.log).parameters if name not in ["self", "kwargs"]
        ]

    def get_logger(self) -> logging.Logger:
        """Get the logger instance
        Returns:
            logging.Logger: The logger instance.
        """
        return self.logger

    def set_level(self, level: str, is_verbose: bool = False) -> None:
        """Set the logging level dynamically.

        Args:
            level (str): The desired logging level. Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
        """
        self.is_verbose = is_verbose
        level = level.upper()
        if level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            self.logger.setLevel(getattr(logging, level))
            self.logger.info(f"Log level set to {level}")
        else:
            self.logger.error(
                f"Invalid log level: {level}. Use one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'."
            )

    def get_level(self) -> str:
        """Get the current logging level.
        returns:
        level (str): friendly name of the current logger level
        """
        level = logging.getLevelName(self.logger.getEffectiveLevel())
        return level

    def print_table_panel(self, record: dict[str, Any], **kwargs) -> None:
        """
        Display an information table inside a panel.

        Args:
            record (dict[str, any]): List of tuples with information to fill the table.
            title (str): Panel title.
            border_style (str): Panel border style.
        """
        panel_kwargs = {k: v for k, v in kwargs.items() if k in self.panel_params}
        console_log_kwargs = {k: v for k, v in kwargs.items() if k in self.console_log_params}
        title = panel_kwargs.pop("title", None)

        table = Table(show_header=False, show_lines=False, show_edge=False, box=None)
        table.add_column("Description", justify="left")
        table.add_column("Value", justify="left")

        for description, value in record.items():
            table.add_row(str(description), str(value))
        with self._lock:
            self.console.log(
                Panel(table, title=title, **panel_kwargs),
                justify=console_log_kwargs.pop("justify", "center"),
                **console_log_kwargs,
            )

    def print_panel(self, information: str, **kwargs) -> None:
        """Display information inside a panel.
        This function takes a string and displays it inside a panel with optional
        parameters for customization. The panel can have a title, border style,
        and other attributes. The function also allows for console logging
        with specific parameters.

        Args:
            information (str): The information to display inside the panel.
        """
        panel_kwargs = {k: v for k, v in kwargs.items() if k in self.panel_params}
        console_log_kwargs = {k: v for k, v in kwargs.items() if k in self.console_log_params}
        with self._lock:
            self.console.log(
                Panel(information, title=panel_kwargs.pop("title", None), **panel_kwargs),
                justify=console_log_kwargs.pop("justify", "full"),
                **console_log_kwargs,
            )

    @staticmethod
    def print_help(parser: argparse.ArgumentParser) -> None:
        """Print the parser's help message beautifully.

        Args:
            parser (argparse.ArgumentParser): The argument parser instance.
        """
        console = Console()
        help_text = parser.format_help()
        console.print(Panel(help_text, title="[bold cyan]Help[/bold cyan]", border_style="blue"))


class MyProgress(Progress):
    """Custom progress bar for displaying tasks in a panel.
    This class extends the Progress class from the rich library to provide
    """

    def __init__(self, console, **kwargs):
        # Default columns for the progress bar
        default_columns = [
            TextColumn("[progress.description]{task.description}"),
            SpinnerColumn(),
            BarColumn(bar_width=None),
            TextColumn("[progress.completed]({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]
        if not console:
            console = Console(stderr=True)
        # Call parent constructor with explicit columns
        super().__init__(*default_columns, console=console, **kwargs)

    def get_renderables(self):
        """
        Get the renderables for the progress bar.

        Yields:
            Renderable: The renderables for the progress bar.
        """
        yield Panel(self.make_tasks_table(self.tasks))
