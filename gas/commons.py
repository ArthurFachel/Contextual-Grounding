from enum import Enum


class TaskType(Enum):
    """
    An enumeration of the different types of tasks in the Geobench dataset.
    """

    NOUN = "NOUN"
    CHOICE = "CHOICE"
    COMPLETION = "COMPLETION"
    TF = "TF"
    QA = "QA"
