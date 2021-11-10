import numpy as np
from databalanceanalysis.constants import distribution_measures_to_func, Measures


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

        # can change depending on reference distribution
        measures = {}
        for measure, func in distribution_measures_to_func.items():
            if measure in [Measures.CHISQ_PVALUE, Measures.CHISQ]:
                measures[measure.value] = func(f_obs["count"], f_ref)
            else:
                measures[measure.value] = func(obs, ref)

        return measures

    def get_all_distribution_measures(self, df, sensitive_cols):
        all_measures = {}
        for col in sensitive_cols:
            all_measures[col] = self.get_distribution_measures(df, col)
        return all_measures

    @property
    def distribution_measures(self):
        return self._distribution_measures
