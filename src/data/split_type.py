"""Enumeration describing the possible dataset splits"""

from enum import Enum


class SplitType(Enum):
    """Provides the dataset split categories"""

    ALL = 0
    TRAIN = 1
    VAL = 2
    TEST = 3
