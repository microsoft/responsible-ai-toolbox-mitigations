# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

from typing import Dict, Callable, List, Tuple

import pandas as pd
import itertools
from raimitigations.databalanceanalysis.balance_measure import BalanceMeasure

from raimitigations.databalanceanalysis.constants import Measures
import raimitigations.databalanceanalysis.balance_metric_functions as BalanceMetricFunctions


"""
This class computes data balance measures based on two different feature values within the same class
"""


class FeatureBalanceMeasure(BalanceMeasure):
    CLASS_A = "ClassA"
    CLASS_B = "ClassB"

    FEATURE_METRICS: Dict[Measures, Callable[[float, float, float, float], float]] = {
        Measures.DEMOGRAPHIC_PARITY: BalanceMetricFunctions.get_demographic_parity,
        Measures.POINTWISE_MUTUAL_INFO: BalanceMetricFunctions.get_point_mutual,
        Measures.SD_COEF: BalanceMetricFunctions.get_sorenson_dice,
        Measures.JACCARD_INDEX: BalanceMetricFunctions.get_jaccard_index,
        Measures.KR_CORRELATION: BalanceMetricFunctions.get_kr_correlation,
        Measures.LOG_LIKELIHOOD: BalanceMetricFunctions.get_log_likelihood_ratio,
        Measures.TTEST: BalanceMetricFunctions.get_t_test_stat,
    }

    OVERALL_METRICS: Dict[Tuple[Measures, Measures], Callable[[float, int], float]] = {
        (
            Measures.TTEST_PVALUE,
            Measures.TTEST,
        ): BalanceMetricFunctions.get_t_test_p_value,
    }

    def __init__(self, sensitive_cols: List[str], label_col: str):
        super().__init__(sensitive_cols=sensitive_cols)
        self._label_col = label_col

    def _get_individual_feature_measures(
        self,
        df: pd.DataFrame,
        sensitive_col: str,
        label_col: str,
        label_pos_val: any = 1,
    ):
        num_rows = df.shape[0]
        p_feature_col = df[sensitive_col].value_counts().rename("p_feature") / num_rows
        p_pos_feature_col = (
            df[df[label_col] == label_pos_val][sensitive_col].value_counts().rename("p_pos_feature") / num_rows
        )
        new_df = pd.concat([p_feature_col, p_pos_feature_col], axis=1)
        new_df["p_pos_feature"] = new_df["p_pos_feature"].fillna(0)
        new_df["p_pos"] = df[df[label_col] == label_pos_val].shape[0] / num_rows
        for measure, func in self.FEATURE_METRICS.items():
            new_df[measure.value] = new_df.apply(
                lambda x: func(x["p_pos"], x["p_feature"], x["p_pos_feature"], num_rows),
                axis=1,
            )
        return new_df

    # dataframe version with a column for the classes and then column for each gap measure
    def _get_gaps(self, df: pd.DataFrame, sensitive_col: str, label_col: str) -> pd.DataFrame:
        metrics_df = self._get_individual_feature_measures(df, sensitive_col, label_col)
        unique_vals = df[sensitive_col].unique()
        # list of tuples of the pairings of classes
        pairs = list(itertools.combinations(unique_vals, 2))
        gap_df = pd.DataFrame(
            pairs,
            columns=[FeatureBalanceMeasure.CLASS_A, FeatureBalanceMeasure.CLASS_B],
        )
        gap_df["FeatureName"] = sensitive_col
        for measure in self.FEATURE_METRICS.keys():
            classA_metric = gap_df[FeatureBalanceMeasure.CLASS_A].apply(lambda x: metrics_df.loc[x])[measure.value]
            classB_metric = gap_df[FeatureBalanceMeasure.CLASS_B].apply(lambda x: metrics_df.loc[x])[measure.value]
            gap_df[measure.value] = classA_metric - classB_metric

        # For overall stats
        for (measure, test_stat), func in self.OVERALL_METRICS.items():
            gap_df[measure.value] = gap_df[test_stat.value].apply(lambda x: func(x, len(unique_vals)))
        return gap_df

    def _get_all_gaps(self, df: pd.DataFrame, sensitive_cols: List[str], label_col: str) -> pd.DataFrame:
        gap_list = [self._get_gaps(df, col, label_col) for col in sensitive_cols]
        return pd.concat(gap_list)

    def measures(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The output is a dictionary that maps the sensitive column table to Pandas dataframe containing the following

            * A feature value within the sensitive feature.
            * Another feature value within the sensitive feature.
            * It contains the following measures of the gaps between the two classes

                * Demographic Parity - https://en.wikipedia.org/wiki/Fairness_(machine_learning)
                * Pointwise Mutual Information - https://en.wikipedia.org/wiki/Pointwise_mutual_information
                * Sorensen-Dice Coefficient - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
                * Jaccard Index - https://en.wikipedia.org/wiki/Jaccard_index
                * Kendall Rank Correlation - https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
                * Log-Likelihood Ratio - https://en.wikipedia.org/wiki/Likelihood_function#Likelihood_ratio
                * t-test - https://en.wikipedia.org/wiki/Student's_t-test

        This output dataframe contains a row per combination of feature values for each sensitive feature.

        :param df: the df to calculate all of the feature balance measures on
        :type df: pd.DataFrame
        :return: a dataframe that contains 4 columns, first column is the sensitive feature's name, 2nd column is one possible value of that sensitive feature,
            the 3rd column is a different possible value of that feature and the last column is a dictionary which indicates
        :rtype: pd.DataFrame
        """
        _feature_measures = self._get_all_gaps(df, self._sensitive_cols, self._label_col)

        return _feature_measures
