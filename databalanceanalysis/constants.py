from enum import Enum

from numpy import log
from databalanceanalysis.feature_functions import (
    get_demographic_parity,
    get_point_mutual,
    get_sorenson_dice,
    get_jaccard_index,
    log_likelihood_ratio,
    t_test_value,
)

from databalanceanalysis.distribution_functions import (
    get_kl_divergence,
    get_js_distance,
    get_ws_distance,
    get_infinity_norm_distance,
    get_total_variation_distance,
    get_chi_squared,
    get_chisq_pvalue,
)

from databalanceanalysis.aggregate_functions import (
    get_atkinson_index,
    get_thiel_l_index,
    get_thiel_t_index,
)


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
    THIEL_L_INDEX = "thiel_l_index"
    THIEL_T_INDEX = "thiel_t_index"
    CROSS_ENTROPY = "cross_entropy"
    KL_DIVERGENCE = "kl_divergence"
    JS_DISTANCE = "js_distance"
    WS_DISTANCE = "ws_distance"
    INF_NORM_DISTANCE = "inf_norm_distance"
    CHISQ = "chisq"
    CHISQ_PVALUE = "chisq_pvalue"
    TOTAL_VARIANCE_DISTANCE = "total_variance_distance"


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
    Measures.THIEL_L_INDEX: "Thiel L Index",
    Measures.THIEL_T_INDEX: "Thiel T Index",
    Measures.CROSS_ENTROPY: "Cross Entropy",
    Measures.KL_DIVERGENCE: "Kullback-Leibler Divergence",
    Measures.JS_DISTANCE: "Jenson-Shannan Distance",
    Measures.WS_DISTANCE: "Wasserstein Distance",
    Measures.INF_NORM_DISTANCE: "Infinity Norm Distance",
    Measures.CHISQ: "Chisquare Value",
    Measures.CHISQ_PVALUE: "Chisquare p-value",
    Measures.TOTAL_VARIANCE_DISTANCE: "Total Variance Distance",
}

distribution_balance_measures = {
    Measures.KL_DIVERGENCE,
    Measures.JS_DISTANCE,
    Measures.WS_DISTANCE,
    Measures.INF_NORM_DISTANCE,
    Measures.CHISQ_PVALUE,
    Measures.CHISQ,
    Measures.TOTAL_VARIANCE_DISTANCE,
}

feature_balance_measures = {
    Measures.DEMOGRAPHIC_PARITY,
    Measures.POINTWISE_MUTUAL_INFO,
    Measures.SD_COEF,
    Measures.JACCARD_INDEX,
    Measures.KR_CORRELATION,
    Measures.LOG_LIKELIHOOD,
    Measures.TTEST_PVALUE,
    Measures.TTEST,
}

aggregate_balance_measures = {
    Measures.ATKINSON_INDEX,
    Measures.THIEL_T_INDEX,
    Measures.THIEL_L_INDEX,
}

feature_measures_to_func = {
    Measures.DEMOGRAPHIC_PARITY: get_demographic_parity,
    Measures.POINTWISE_MUTUAL_INFO: get_point_mutual,
    Measures.SD_COEF: get_sorenson_dice,
    Measures.JACCARD_INDEX: get_jaccard_index,
    # Measures.KR_CORRELATION: get_kr_correlation,
    Measures.LOG_LIKELIHOOD: log_likelihood_ratio,
    # Measures.TTEST_PVALUE: t_test_value,
    Measures.TTEST: t_test_value,
}

distribution_measures_to_func = {
    Measures.KL_DIVERGENCE: get_kl_divergence,
    Measures.JS_DISTANCE: get_js_distance,
    Measures.WS_DISTANCE: get_ws_distance,
    Measures.INF_NORM_DISTANCE: get_infinity_norm_distance,
    Measures.CHISQ_PVALUE: get_chisq_pvalue,
    Measures.CHISQ: get_chi_squared,
    Measures.TOTAL_VARIANCE_DISTANCE: get_total_variation_distance,
}

aggregate_measures_to_func = {
    Measures.THIEL_L_INDEX: get_thiel_l_index,
    Measures.THIEL_T_INDEX: get_thiel_t_index,
    Measures.ATKINSON_INDEX: get_atkinson_index,
}
