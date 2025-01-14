"""Enumeration describing the possible dataset splits"""

from enum import Enum


class SplitType(Enum):
    """Provides the dataset split categories"""

    TRAIN = 0
    VAL = 1
    TEST = 2
