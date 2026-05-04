from gas.commons import TaskType
from gas.logger import Logger

logging = Logger()
logger = logging.get_logger()


class PromptProvider:
    @staticmethod
    def fetch_instruction(task_type: TaskType) -> str:
        """
        Fetches the task instruction for the specified benchmark type.
        Args:
            banchmark_type: The type of benchmark.
        Returns:
            The task instruction for the specified benchmark type.
        """
        match task_type:
            case TaskType.NOUN:
                instruction = (
                    "The concept below is about college geoscience. "
                    "Provide a concise definition of the specific geoscience concept. "
                    "Answer should be short and to the point. "
                    "Answer should have technical specifications. "
                    "If applicable, answer should have relevant numerical details."
                )
            case TaskType.CHOICE:
                instruction = (
                    "The question below is about college geoscience. "
                    "The question below have answer options (associated with alphabet letters) "
                    "out of which only one is correct. "
                    "Answer the question by selecting the correct option. "
                    'Response should be just "A", "B", "C", or "D". '
                    "Do not add any explanations."
                )
            case TaskType.COMPLETION:
                instruction = (
                    "The sentence below is about college geoscience. "
                    "The sentence below is incomplete and has a blank spaces. "
                    'Blank spaces are represented by "_" and "()". '
                    "Answer the sentence by filling in the blank space. "
                    "Answer should be just the missing words that fits in each blank "
                    "If there are no blank spaces, fill in the end. "
                    "without any additional information."
                )
            case TaskType.TF:
                instruction = (
                    "The statement below is about college geoscience. "
                    "The statement below is either true or false. "
                    "Answer if the statement is true or false. "
                    'Response should be just "True" or "False" '
                    "without any additional information."
                )
            case TaskType.QA:
                instruction = (
                    "The question below is about college geoscience. "
                    "Provide a concise definition of the specific geoscience question. "
                    "Answer should be short and to the point. "
                    "Answer should have technical specifications. "
                    "If applicable, answer should have relevant numerical details."
                )
            case _:
                raise ValueError("Invalid benchmark type")

        task_instruction = (
            "Below is an instruction that describes a task, "
            "paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
        )
        if logging.get_level() == "DEBUG":
            logging.print_panel(
                f"[yellow][align=left]{task_instruction}[/align][/yellow]",
                title="[bold bright_yellow]Prompt Instruction[/bold bright_yellow]",
                border_style="yellow",
            )
        return task_instruction

    @staticmethod
    def build_messages(instruction: str, input: str, apply_chat_template: bool) -> list[dict[str, str]] | str:
        """
        Build messages for the chat template.
        This function takes an instruction and input, and returns a list of messages
        formatted for the chat template. The messages include the instruction, a prompt
        for the user, and a prompt for the assistant. The function also includes an
        option to apply the chat template, which determines whether the messages are
        formatted as a list of dictionaries or as a single string.

        Args:
            instruction (str): instruction for the task
            input (str): input for the task
            apply_chat_template (bool): whether to apply the chat template

        Returns:
            list[dict[str, str]] | str: formatted messages for the prompt
        """
        if apply_chat_template:
            return [
                {"role": "user", "content": f"{instruction}\n"},
                {"role": "assistant", "content": "Okay, I'm ready. Please provide the input.\n"},
                {"role": "user", "content": f"### Input:\n{input}\n\n### Response:\n"},
            ]
        else:
            return f"{instruction}\n" f"### Input:\n{input}\n\n### Response:\n"
