# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

import numpy as np
import scipy

from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from scipy.stats import chisquare

"""
Helper functions to calculate each of the individual aggregate measures
"""

# Aggregate Balance Measure Helper Functions
def _get_generalized_entropy_index(
    benefits: np.array,
    alpha: float,
    use_abs_val: bool,
    error_tolerance: float = 1e-12,
) -> float:
    """
    Returns general Entropy index: measure of redundancy in the data
    https://en.wikipedia.org/wiki/Generalized_entropy_index

            parameters:
                benefits:
                alpha:
                use_abs_val:
                error_tolerance

            returns:
                gei (float) general entropy index for
    """
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
        gei = np.sum(np.power(norm_benefits, alpha) - 1.0) / (count * alpha * (alpha - 1.0))
    return gei


def get_atkinson_index(benefits: np.array, epsilon: float = 1.0, error_tolerance: float = 1e-12) -> float:
    """
    Returns the Atkinson INdex which is a measure of income inequality that indicates percentage  that would need to been foregone to have equal shares of income
    https://en.wikipedia.org/wiki/Atkinson_index

    """
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
    """
    Returns thiel t index measure of income inequality that is more sensitive to differences at the top of the distribution
    https://en.wikipedia.org/wiki/Theil_index
    """
    return _get_generalized_entropy_index(benefits, 1.0, True)


def get_theil_l_index(benefits: np.array) -> float:
    """
    Returns theil L index which is measure of income inequality that is more sensitive to differences at the lower
    end of the distribution
    https://en.wikipedia.org/wiki/Theil_index
    """
    return _get_generalized_entropy_index(benefits, 0.0, True)


# Distribution Balance Measures Helper Functions


def get_cross_entropy(obs: np.array, ref: np.array) -> float:
    """
    Returns cross entropy
    """
    return -np.sum(ref + np.log2(obs))


def get_kl_divergence(obs: np.array, ref: np.array) -> float:
    """
    Returns Kullback-Leibler (KL) divergence
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    """
    return entropy(obs, qk=ref)


def get_js_distance(obs: np.array, ref: np.array) -> float:
    """
    Returns Jenson-Shannon Distance
    https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    """
    avg = (obs + ref) / 2
    divergence = (entropy(obs, avg) + entropy(ref, avg)) / 2
    distance = np.sqrt(divergence)
    return distance


def get_ws_distance(obs: np.array, ref: np.array) -> float:
    """
    Returns Wasserstein Distance
    https://en.wikipedia.org/wiki/Wasserstein_metric
    """
    return wasserstein_distance(obs, ref)


def get_infinity_norm_distance(obs: np.array, ref: np.array) -> float:
    """
    Returns Infinity norm distance (also known as the Chebyshev distance)
    https://en.wikipedia.org/wiki/Chebyshev_distance
    """
    return distance.chebyshev(obs, ref)


def get_total_variation_distance(obs: np.array, ref: np.array) -> float:
    """
    Returns Total variation distance
    https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures
    """
    return 0.5 * np.sum(np.abs(obs - ref))


def get_chi_squared(f_obs: np.array, f_ref: np.array) -> float:
    """
    Returns Chi squared test statistic
    f_obs is the frequencies of the observation rather than the arrays themselves
    https://en.wikipedia.org/wiki/Chi-squared_test
    """
    res = chisquare(f_obs, f_ref)
    chisq: float = res[0]
    return chisq


def get_chisq_pvalue(f_obs: np.array, f_ref: np.array) -> float:
    """
    Returns p-value of the chi squared test
    https://en.wikipedia.org/wiki/Chi-squared_test
    """
    res = chisquare(f_obs, f_ref)
    pvalue: float = res[1]
    return pvalue


# Feature Balance Measures Helper functions
def get_demographic_parity(p_pos: float, p_feature: float, p_pos_feature: float, total_count: int) -> float:
    """
    Returns Demographic Parity
    """
    return p_pos_feature / p_feature


def get_point_mutual(p_pos: float, p_feature: float, p_pos_feature: float, total_count: int) -> float:
    """
    Returns Point Mutual Information
    """
    dp = get_demographic_parity(p_pos, p_feature, p_pos_feature, total_count)
    return -np.inf if dp == 0 else np.log(dp)


def get_sorenson_dice(p_pos: float, p_feature: float, p_pos_feature: float, total_count: int) -> float:
    """
    Returns Sorensen-Dice - used to gauge the similarity between two samples
    https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    """

    return p_pos_feature / (p_feature + p_pos)


def get_jaccard_index(p_pos: float, p_feature: float, p_pos_feature: float, total_count: int) -> float:
    """
    Returns Jaccard Index - used for gauging similarity and diversity of sample sets
    https://en.wikipedia.org/wiki/Jaccard_index
    """
    return p_pos_feature / (p_feature + p_pos - p_pos_feature)


def get_kr_correlation(p_feature: float, p_pos: float, p_pos_feature: float, total_count: int) -> float:
    """
    Returns Kendall Rank Correlation - ordinal association between two measured quantities
    https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
    """
    a = np.power(total_count, 2) * (1 - 2 * p_feature - 2 * p_pos + 2 * p_pos_feature + 2 * p_pos * p_feature)
    b = total_count * (2 * p_feature + 2 * p_pos - 4 * p_pos_feature - 1)
    c = np.power(total_count, 2) * np.sqrt((p_feature - np.power(p_feature, 2)) * (p_pos - np.power(p_pos, 2)))
    return (a + b) / c


def get_log_likelihood_ratio(p_pos: float, p_feature: float, p_pos_feature: float, total_count: int) -> float:
    """
    Returns Log likelihood ratio - calculate the degree to which the data supports one variable versus another
    https://en.wikipedia.org/wiki/Likelihood_function#Likelihood_ratio
    """
    return -np.inf if p_pos_feature == 0 else np.log(p_pos_feature / p_pos)


def get_t_test_stat(p_pos: float, p_feature: float, p_pos_feature: float, total_count: int) -> float:
    """
    Returns t-test statistic - t test compares the means of two groups and this gets a value to be looked
    up on t-distribution
    https://en.wikipedia.org/wiki/Student's_t-test
    """
    return (p_pos - (p_feature * p_pos)) / np.sqrt(p_feature * p_pos)


def get_t_test_p_value(metric: float, num_unique_vals: int) -> float:
    """
    Returns t-test p-value - t test compares the means of two groups and this gets the p-value to be compares to some
    alpha to check for statistical significance
    https://en.wikipedia.org/wiki/Student's_t-test
    """
    t_statistic = metric
    dof = num_unique_vals - 1
    return scipy.stats.t.sf(abs(t_statistic), dof)
