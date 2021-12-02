# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

import numpy as np
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from scipy.stats import chisquare

"""
Helper functions to calculate distribution measures given an observed and reference column
"""


class DistributionFunctions:
    def get_cross_entropy(obs: np.array, ref: np.array) -> float:
        return -np.sum(ref + np.log2(obs))

    def get_kl_divergence(obs: np.array, ref: np.array) -> float:
        return entropy(obs, qk=ref)

    def get_js_distance(obs: np.array, ref: np.array) -> float:
        avg = (obs + ref) / 2
        divergence = (entropy(obs, avg) + entropy(ref, avg)) / 2
        distance = np.sqrt(divergence)
        return distance

    def get_ws_distance(obs: np.array, ref: np.array) -> float:
        return wasserstein_distance(obs, ref)

    def get_infinity_norm_distance(obs: np.array, ref: np.array) -> float:
        return distance.chebyshev(obs, ref)

    def get_total_variation_distance(obs: np.array, ref: np.array) -> float:
        return 0.5 * np.sum(np.abs(obs - ref))

    # f_obs is the frequencies of the observation rather than the arrays themselves
    def get_chi_squared(f_obs: np.array, f_ref: np.array) -> float:
        res = chisquare(f_obs, f_ref)
        chisq: float = res[0]
        return chisq

    def get_chisq_pvalue(f_obs: np.array, f_ref: np.array) -> float:
        res = chisquare(f_obs, f_ref)
        pvalue: float = res[1]
        return pvalue
