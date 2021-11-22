import numpy as np
from databalanceanalysis.databalanceanalysis.constants import (
    distribution_measures_to_func,
    Measures,
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


class DistributionMeasures:
    def __init__(self, df, sensitive_cols):
        self._df = df
        self._sensitive_cols = sensitive_cols
        self._distribution_measures = self.get_all_distribution_measures(
            df, sensitive_cols
        )

    def get_ref_col(self, ref_dist, n):
        if ref_dist == "uniform":
            uniform_val = 1 / n
            return np.ones(n) * uniform_val
        else:
            raise Exception("reference distribution not implemented")

    def get_distribution_measures(self, df, sensitive_col, ref_dist="uniform"):
        f_obs = df.groupby(sensitive_col).size().reset_index(name="count")
        sum_obs = f_obs["count"].sum()
        obs = f_obs["count"] / sum_obs
        ref = self.get_ref_col(ref_dist, f_obs.shape[0])
        f_ref = ref * sum_obs

        # TODO can change depending on the reference distribution
        measures = {}
        for measure, func in distribution_measures_to_func.items():
            if measure in [Measures.CHISQ_PVALUE, Measures.CHISQ]:
                measures[measure] = func(f_obs["count"], f_ref)
            else:
                measures[measure] = func(obs, ref)

        return measures

    def get_all_distribution_measures(self, df, sensitive_cols):
        all_measures = {}
        for col in sensitive_cols:
            all_measures[col] = self.get_distribution_measures(df, col)
        return all_measures

    @property
    def measures(self):
        return self._distribution_measures
