import numpy as np
from distribution_functions import (
    get_cross_entropy,
    get_kl_divergence,
    get_js_distance,
    get_ws_distance,
    get_infinity_norm_distance,
    get_total_variation_distance,
    get_chi_squared,
    get_chisq_pvalue,
)
from constants import distribution_measures_to_func


class DistributionMeasures:
    def __init__(self, df, sensitive_cols):
        self._df = df
        self._sensitive_cols = sensitive_cols

        dist_measures = {}
        for col in sensitive_cols:
            dist_measures[col] = self.get_distribution_measures(self, df, col)
        self._distribution_measures = dist_measures

    def get_ref_col(self, ref_dist, n):
        if ref_dist == "uniform":
            uniform_val = 1 / n
            return np.ones() * uniform_val
        else:
            raise Exception("reference distribution not implemented")

    def get_distribution_measures(self, df, sensitive_col, ref_dist="uniform"):
        f_obs = df.groupby(sensitive_col).size().reset_index(name="count")
        sum_obs = f_obs.sum()
        obs = f_obs / sum_obs
        ref = self.get_ref_col(ref_dist, f_obs.size)
        f_ref = ref * sum_obs

        # can change depending on reference distribution
        measures = {}
        for measure, func in distribution_measures_to_func.items():
            if measure in [Measures.CHISQ_PVALUE, Measures.CHISQ]:
                measures[measure.value] = func(f_obs, f_ref)
            else:
                measures[measure.value] = func(obs, ref)

        return measures

    @property
    def distribution_measures(self):
        return self._distribution_measures

