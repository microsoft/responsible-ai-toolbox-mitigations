# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

from itertools import count
import numpy as np

"""
Helper functions to calculate each of the individual aggregate measures 
"""


class AggregateFunctions:
    def get_generalized_entropy_index(
        benefits: np.array,
        alpha: float,
        use_abs_val: bool,
        error_tolerance: float = 1e-12,
    ) -> float:
        if use_abs_val:
            benefits = np.absolute(benefits)
        benefits_mean = np.mean(benefits)
        norm_benefits = benefits / benefits_mean
        count = norm_benefits.size
        if abs(alpha - 1.0) < error_tolerance:
            gei = np.sum(norm_benefits * np.log(norm_benefits)) / count
        elif abs(alpha) < error_tolerance:
            gei = np.sum(-np.log(norm_benefits)) / count
        else:
            gei = np.sum(np.power(norm_benefits, alpha) - 1.0) / (
                count * alpha * (alpha - 1.0)
            )
        return gei

    def get_atkinson_index(
        benefits: np.array, epsilon: float = 1.0, error_tolerance: float = 1e-12
    ) -> float:
        count = benefits.size
        benefits_mean = np.mean(benefits)
        norm_benefits = benefits / benefits_mean
        alpha = 1 - epsilon
        if abs(alpha) < error_tolerance:
            ati = 1.0 - np.power(np.prod(norm_benefits), 1.0 / count)
        else:
            power_mean = np.sum(np.power(norm_benefits, alpha)) / count
            ati = 1.0 - np.power(power_mean, 1.0 / alpha)
        return ati

    def get_theil_t_index(benefits: np.array) -> float:
        return self.get_generalized_entropy_index(benefits, 1.0, True)

    def get_theil_l_index(benefits: np.array) -> float:
        return self.get_generalized_entropy_index(benefits, 0.0, True)
