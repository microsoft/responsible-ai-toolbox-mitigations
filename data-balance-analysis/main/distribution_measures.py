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
        f_obs = df[sensitive_col].nunique()
        sum_obs = f_obs.sum()
        obs = f_obs / sum_obs
        ref = self.get_ref_col(ref_dist, f_obs.size)

        # can change depending on reference distribution
        measures = {}
        measures["cross_entropy"] = get_cross_entropy(obs, ref)
        measures["kl_divergence"] = get_kl_divergence(obs, ref)
        measures["js_distance"] = get_js_distance(obs, ref)
        measures["ws_distance"] = get_ws_distance(obs, ref)
        measures["infinity_norm_distance"] = get_infinity_norm_distance(obs, ref)
        measures["total_variation_distance"] = get_total_variation_distance(obs, ref)
        measures["chisq"] = get_chi_squared(obs, ref)
        measures["chisq_pvalue"] = get_chisq_pvalue(obs, ref)
        return measures

    @property
    def distribution_measures(self):
        return self._distribution_measures

