# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

from typing import Dict, Callable

import numpy as np
import pandas as pd

from databalanceanalysis.databalanceanalysis.constants import Measures
from databalanceanalysis.databalanceanalysis.aggregate_functions import (
    AggregateFunctions,
)

"""
This class computes a set of aggregated balance measures that represents how balanced
the given dataframe is along the given sensitive features.
  
The output is a dictionary that maps the names of the different aggregate measures to their values:
    The following measures are computed:
    - Atkinson Index - https://en.wikipedia.org/wiki/Atkinson_index
    - Theil Index (L and T) - https://en.wikipedia.org/wiki/Theil_index
"""


class AggregateBalanceMeasures:

    AGGREGATE_METRICS: Dict[Measures, Callable[[np.array], float]] = {
        Measures.THEIL_L_INDEX: AggregateFunctions.get_theil_l_index,
        Measures.THEIL_T_INDEX: AggregateFunctions.get_theil_t_index,
        Measures.ATKINSON_INDEX: AggregateFunctions.get_atkinson_index,
    }

    def __init__(self, df: pd.DataFrame, sensitive_cols: List[str]):
        self._benefits = df.groupby(sensitive_cols).size() / df.shape[0]
        self._aggregate_measures = {}
        for measure in self.aggregate_balance_measures:
            func = self.AGGREGATE_METRICS[measure]
            self._aggregate_measures[measure] = func(self._benefits)

    @property
    def measures(self) -> Dict[str, float]:
        return self._aggregate_measures
