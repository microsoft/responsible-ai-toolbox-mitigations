# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

from enum import Enum


class Measures(str, Enum):
    """Provide the supported dataset imbalance metrics"""

    DEMOGRAPHIC_PARITY = "dp"
    POINTWISE_MUTUAL_INFO = "pmi"
    SD_COEF = "sdc"
    JACCARD_INDEX = "ji"
    KR_CORRELATION = "krc"
    LOG_LIKELIHOOD = "llr"
    TTEST = "t_test"
    TTEST_PVALUE = "ttest_pvalue"
    ATKINSON_INDEX = "atkinson_index"
    THEIL_L_INDEX = "theil_l_index"
    THEIL_T_INDEX = "theil_t_index"
    CROSS_ENTROPY = "cross_entropy"
    KL_DIVERGENCE = "kl_divergence"
    JS_DISTANCE = "js_dist"
    WS_DISTANCE = "wasserstein_dist"
    INF_NORM_DISTANCE = "inf_norm_dist"
    CHISQ = "chi_sq_stat"
    CHISQ_PVALUE = "chi_sq_p_value"
    TOTAL_VARIANCE_DISTANCE = "total_variation_dist"


# Display name to be used for the UI
measure_to_display_name = {
    Measures.DEMOGRAPHIC_PARITY: "Demographic Parity",
    Measures.POINTWISE_MUTUAL_INFO: "Pointwise Mutual Information",
    Measures.SD_COEF: "Sorenson-Dice Coefficient",
    Measures.JACCARD_INDEX: "Jaccard Index",
    Measures.KR_CORRELATION: "Kendall Rank Correlation",
    Measures.LOG_LIKELIHOOD: "Log Likelihood Ratio",
    Measures.TTEST: "T-test Value",
    Measures.TTEST_PVALUE: "T-test p-value",
    Measures.ATKINSON_INDEX: "Atkinson Index",
    Measures.THEIL_L_INDEX: "Theil L Index",
    Measures.THEIL_T_INDEX: "Theil T Index",
    Measures.CROSS_ENTROPY: "Cross Entropy",
    Measures.KL_DIVERGENCE: "Kullback-Leibler Divergence",
    Measures.JS_DISTANCE: "Jenson-Shannan Distance",
    Measures.WS_DISTANCE: "Wasserstein Distance",
    Measures.INF_NORM_DISTANCE: "Infinity Norm Distance",
    Measures.CHISQ: "Chisquare Value",
    Measures.CHISQ_PVALUE: "Chisquare p-value",
    Measures.TOTAL_VARIANCE_DISTANCE: "Total Variance Distance",
}
