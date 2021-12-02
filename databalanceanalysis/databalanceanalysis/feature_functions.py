# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

from typing import Any

import numpy as np
import pandas as pd

"""
Helper functions to calculate measures for individual features
"""


class FeatureFunctions:
    def get_demographic_parity(
        self, p_pos: float, p_feature: float, p_pos_feature: float, total_count: float
    ) -> float:
        return p_pos_feature / p_feature

    def get_point_mutual(
        self, p_pos: float, p_feature: float, p_pos_feature: float, total_count: float
    ) -> float:
        dp = self.get_demographic_parity(p_pos, p_feature, p_pos_feature)
        return -np.inf if dp == 0 else np.log(dp)

    def get_sorenson_dice(
        self, p_pos: float, p_feature: float, p_pos_feature: float, total_count: float
    ) -> float:
        return p_pos_feature / (p_feature + p_pos)

    def get_jaccard_index(
        self, p_pos: float, p_feature: float, p_pos_feature: float, total_count: float
    ) -> float:
        return p_pos_feature / (p_feature + p_pos - p_pos_feature)

    def get_kr_correlation(
        self, p_feature: float, p_pos: float, p_pos_feature: float, total_count: int
    ) -> float:
        a = np.pow(total_count, 2) * (
            1 - 2 * p_feature - 2 * p_pos + 2 * p_pos_feature + 2 * p_pos * p_feature
        )
        b = total_count * (2 * p_feature + 2 * p_pos - 4 * p_pos_feature - 1)
        c = np.pow(total_count, 2) * np.sqrt(
            (p_feature - np.pow(p_feature, 2)) * (p_pos - np.pow(p_pos, 2))
        )
        return (a + b) / c

    def get_log_likelihood_ratio(
        self, p_pos: float, p_feature: float, p_pos_feature: float, total_count: int
    ) -> float:
        return np.log(p_pos_feature / p_pos)

    def get_t_test_stat(
        self, p_pos: float, p_feature: float, p_pos_feature: float, total_count: int
    ) -> float:
        return (p_pos - (p_feature * p_pos)) / np.sqrt(p_feature * p_pos)

    def get_t_test_p_value(self, metric: float, num_unique_vals: int) -> float:
        t_statistic = metric
        dof = num_unique_vals - 1
        return scipy.stats.t.sf(np.abs(t_statistic), dof)
