import numpy as np
import itertools
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from scipy.stats import chisquare

# # Metrics for Comparison with Reference Distribution
# - Many distance measures available in toolkit to find how far is expected and observed distribution of label and sensitive column attributes
def get_cross_entropy(obs, ref):
    p = ref
    q = obs
    return -np.sum(p + np.log2(q))


def get_kl_divergence(obs, ref):
    p = np.asarray(obs)
    q = np.asarray(ref)
    return entropy(p, qk=q)


## Jensen-Shannon Distance
# - Measuring the similarity between two probability distributions.
# Symmetrized and smoothed version of the Kullback–Leibler divergence.
def get_js_distance(obs, ref):
    p = np.asarray(obs)
    q = np.asarray(ref)
    m = (p + q) / 2
    divergence = (entropy(p, m) + entropy(q, m)) / 2
    distance = np.sqrt(divergence)
    return distance


def get_ws_distance(obs, ref):
    p = np.asarray(obs)
    q = np.asarray(ref)
    return wasserstein_distance(p, q)


def get_infinity_norm_distance(obs, ref):
    p = np.asarray(obs)
    q = np.asarray(ref)
    return distance.chebyshev(p, q)


def get_total_variation_distance(obs, ref):
    p = np.asarray(obs)
    q = np.asarray(ref)
    total_var_dis = 0.5 * np.sum(np.abs(p - q))
    return total_var_dis


# f_obs is the frequencies of the observation rather than the arrays themselves
def get_chi_squared(f_obs, f_ref):

    res = chisquare(f_obs, f_ref)
    chisq = res[0]
    return chisq


def get_chisq_pvalue(f_obs, f_ref):
    res = chisquare(f_obs, f_ref)
    pvalue = res[1]
    return pvalue
