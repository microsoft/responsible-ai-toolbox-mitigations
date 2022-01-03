# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

from typing import Dict, Callable, List

import numpy as np
import pandas as pd
from databalanceanalysis.databalanceanalysis.balance_measure import BalanceMeasure

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


class AggregateBalanceMeasure(BalanceMeasure):

    AGGREGATE_METRICS: Dict[Measures, Callable[[np.array], float]] = {
        Measures.THEIL_L_INDEX: BalanceMetricFunctions.get_theil_l_index,
        Measures.THEIL_T_INDEX: BalanceMetricFunctions.get_theil_t_index,
        Measures.ATKINSON_INDEX: BalanceMetricFunctions.get_atkinson_index,
    }

    def __init__(self, sensitive_cols: List[str]):
        super().__init__(sensitive_cols=sensitive_cols)

    def measures(self, df: pd.DataFrame) -> pd.DataFrame:
        _aggregate_measures_dict = {}
        _benefits = df.groupby(self._sensitive_cols).size() / df.shape[0]
        for measure, func in self.AGGREGATE_METRICS.items():
            _aggregate_measures_dict[measure] = [func(_benefits)]
        _aggregate_measures = pd.DataFrame.from_dict(_aggregate_measures_dict)
        return _aggregate_measures
