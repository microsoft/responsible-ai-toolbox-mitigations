# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.
from typing import Dict, Callable, List, Optional

import numpy as np
import pandas as pd
from databalanceanalysis.databalanceanalysis.balance_measure import BalanceMeasure

from databalanceanalysis.databalanceanalysis.constants import (
    Measures,
)
import databalanceanalysis.databalanceanalysis.balance_metric_functions as BalanceMetricFunctions


"""
This class computes data balance measures for sensitive columns based on a reference distribution.
For now, we only support a uniform reference distribution.

The output is a dictionary that maps the sensitive column name to another dictionary:
    the dictionary for each sensitive column contains a mapping of the name of a measure to its value 
        - Kullback-Leibler Divergence - https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
        - Jensen-Shannon Distance - https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
        - Wasserstein Distance - https://en.wikipedia.org/wiki/Wasserstein_metric
        - Infinity Norm Distance - https://en.wikipedia.org/wiki/Chebyshev_distance
        - Total Variation Distance - https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures
        - Chi-Squared Test - https://en.wikipedia.org/wiki/Chi-squared_test
    There is one dictionary for each of the sensitive columns specified
"""


class DistributionBalanceMeasure(BalanceMeasure):
    DISTRIBUTION_METRICS: Dict[Measures, Callable[[np.array, np.array], float]] = {
        Measures.KL_DIVERGENCE: BalanceMetricFunctions.get_kl_divergence,
        Measures.JS_DISTANCE: BalanceMetricFunctions.get_js_distance,
        Measures.WS_DISTANCE: BalanceMetricFunctions.get_ws_distance,
        Measures.INF_NORM_DISTANCE: BalanceMetricFunctions.get_infinity_norm_distance,
        Measures.TOTAL_VARIANCE_DISTANCE: BalanceMetricFunctions.get_total_variation_distance,
        Measures.CHISQ_PVALUE: BalanceMetricFunctions.get_chisq_pvalue,
        Measures.CHISQ: BalanceMetricFunctions.get_chi_squared,
    }

    def __init__(self, sensitive_cols: List[str]):
        super().__init__(sensitive_cols=sensitive_cols)

    # given the distribution, get the column of values
    def _get_ref_col(self, n: int, ref_dist: Dict[str, float] = None) -> np.array:
        # if ref_dist == None:
        uniform_val: float = 1.0 / n
        return np.ones(n) * uniform_val

    def _get_distribution_measures(
        self, df: pd.DataFrame, sensitive_col: str
    ) -> Dict[Measures, float]:
        f_obs = df.groupby(sensitive_col).size().reset_index(name="count")
        sum_obs = f_obs["count"].sum()
        obs = f_obs["count"] / sum_obs
        ref = self._get_ref_col(f_obs.shape[0])
        f_ref = ref * sum_obs

        # TODO future can change depending on the reference distribution
        measures = {"feature_name": sensitive_col}
        for measure, func in self.DISTRIBUTION_METRICS.items():
            if measure in [Measures.CHISQ_PVALUE, Measures.CHISQ]:
                measures[measure] = func(f_obs["count"], f_ref)
            else:
                measures[measure] = func(obs, ref)

        return measures

    def _get_all_distribution_measures(
        self, df: pd.DataFrame, sensitive_cols: List[str]
    ) -> pd.DataFrame:
        all_measures = [
            self._get_distribution_measures(df, col) for col in sensitive_cols
        ]
        return pd.DataFrame.from_dict(all_measures)

    def measures(self, df: pd.DataFrame) -> pd.DataFrame:
        _distribution_measures = self._get_all_distribution_measures(
            df, self._sensitive_cols
        )
        return _distribution_measures
