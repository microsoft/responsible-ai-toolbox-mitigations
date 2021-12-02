# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

from enum import Enum
from numpy import log

import databalanceanalysis.databalanceanalysis.feature_functions as feature_functions
import databalanceanalysis.databalanceanalysis.distribution_functions as distribution_functions
import databalanceanalysis.databalanceanalysis.aggregate_functions as aggregate_functions


class Measures(str, Enum):
    """Provide the supported dataset imbalance metrics"""

    DEMOGRAPHIC_PARITY = "demographic_parity"
    POINTWISE_MUTUAL_INFO = "pointwise_mutual_info"
    SD_COEF = "sd_coef"
    JACCARD_INDEX = "jaccard_index"
    KR_CORRELATION = "kr_correlation"
    LOG_LIKELIHOOD = "log_likelihood"
    TTEST = "ttest"
    TTEST_PVALUE = "ttest_pvalue"
    ATKINSON_INDEX = "atkinson_index"
    THEIL_L_INDEX = "theil_l_index"
    THEIL_T_INDEX = "theil_t_index"
    CROSS_ENTROPY = "cross_entropy"
    KL_DIVERGENCE = "kl_divergence"
    JS_DISTANCE = "js_distance"
    WS_DISTANCE = "ws_distance"
    INF_NORM_DISTANCE = "inf_norm_distance"
    CHISQ = "chisq"
    CHISQ_PVALUE = "chisq_pvalue"
    TOTAL_VARIANCE_DISTANCE = "total_variance_distance"


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
