# abstract class that all the balance measures are defined by
from typing import List
from abc import ABC, abstractmethod
import pandas as pd

"""
An abstract class that shows the measures method that allows us to actually retrieve the
data balance measures
"""


class BalanceMeasure(ABC):
    def __init__(self, sensitive_cols: List[str]) -> None:
        self._sensitive_cols = sensitive_cols

    @abstractmethod
    def measures(self, df: pd.DataFrame):
        pass
