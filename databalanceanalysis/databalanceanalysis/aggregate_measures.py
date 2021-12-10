# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

from typing import Dict, Callable, List

import numpy as np
import pandas as pd

from databalanceanalysis.databalanceanalysis.constants import Measures
import databalanceanalysis.databalanceanalysis.balance_metric_functions as BalanceMetricFunctions


"""
This class computes a set of aggregated balance measures that represents how balanced
the given dataframe is along the given sensitive features.
  
The output is a dictionary that maps the names of the different aggregate measures to their values:
    The following measures are computed:
    - Atkinson Index - https://en.wikipedia.org/wiki/Atkinson_index
    - Theil Index (L and T) - https://en.wikipedia.org/wiki/Theil_index
"""


class AggregateBalanceMeasure:

    AGGREGATE_METRICS: Dict[Measures, Callable[[np.array], float]] = {
        Measures.THEIL_L_INDEX: BalanceMetricFunctions.get_theil_l_index,
        Measures.THEIL_T_INDEX: BalanceMetricFunctions.get_theil_t_index,
        Measures.ATKINSON_INDEX: BalanceMetricFunctions.get_atkinson_index,
    }

    def __init__(self, df: pd.DataFrame, sensitive_cols: List[str]):
        self._benefits = df.groupby(sensitive_cols).size() / df.shape[0]
        self._aggregate_measures_dict = {}
        for measure, func in self.AGGREGATE_METRICS.items():
            self._aggregate_measures_dict[measure] = [func(self._benefits)]
        self._aggregate_measures = pd.DataFrame.from_dict(self._aggregate_measures_dict)

    @property
    def measures(self) -> pd.DataFrame:
        return self._aggregate_measures
