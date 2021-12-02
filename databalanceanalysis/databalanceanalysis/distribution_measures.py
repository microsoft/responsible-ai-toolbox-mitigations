# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.
from typing import Dict, Callable, List, Optional

import numpy as np
import pandas as pd

from databalanceanalysis.databalanceanalysis.constants import (
    Measures,
)
from databalanceanalysis.databalanceanalysis.distribution_functions import (
    DistributionFunctions,
)

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


class DistributionBalanceMeasures:
    DISTRIBUTION_METRICS: Dict[Measures : Callable[[np.array, np.array], float]] = {
        Measures.KL_DIVERGENCE: DistributionFunctions.get_kl_divergence,
        Measures.JS_DISTANCE: DistributionFunctions.get_js_distance,
        Measures.WS_DISTANCE: DistributionFunctions.get_ws_distance,
        Measures.INF_NORM_DISTANCE: DistributionFunctions.get_infinity_norm_distance,
        Measures.TOTAL_VARIANCE_DISTANCE: DistributionFunctions.get_total_variation_distance,
        Measures.CHISQ_PVALUE: DistributionFunctions.get_chisq_pvalue,
        Measures.CHISQ: DistributionFunctions.get_chi_squared,
    }

    def __init__(self, df: pd.DataFrame, sensitive_cols: List[str]):
        self._df = df
        self._sensitive_cols = sensitive_cols
        self._distribution_measures = self.get_all_distribution_measures(
            df, sensitive_cols
        )

    def get_ref_col(self, ref_dist: str, n: int) -> np.array:
        if ref_dist == "uniform":
            uniform_val: float = 1.0 / n
            return np.ones(n) * uniform_val
        else:
            raise Exception("reference distribution not implemented")

    def get_distribution_measures(
        self, df: pd.DataFrame, sensitive_col: str, ref_dist: Optional[str] = "uniform"
    ) -> Dict[Measures, float]:
        f_obs = df.groupby(sensitive_col).size().reset_index(name="count")
        sum_obs = f_obs["count"].sum()
        obs = f_obs["count"] / sum_obs
        ref = self.get_ref_col(ref_dist, f_obs.shape[0])
        f_ref = ref * sum_obs

        # TODO can change depending on the reference distribution
        measures = {}
        for measure, func in self.DISTRIBUTION_METRICS.items():
            if measure in [Measures.CHISQ_PVALUE, Measures.CHISQ]:
                measures[measure] = func(f_obs["count"], f_ref)
            else:
                measures[measure] = func(obs, ref)

        return measures

    def get_all_distribution_measures(
        self, df: pd.DataFrame, sensitive_cols: List[str]
    ) -> Dict[str, Dict[Measures, float]]:
        all_measures = {}
        for col in sensitive_cols:
            all_measures[col] = self.get_distribution_measures(df, col)
        return all_measures

    @property
    def measures(self) -> Dict[str, Dict[Measures, float]]:
        return self._distribution_measures
