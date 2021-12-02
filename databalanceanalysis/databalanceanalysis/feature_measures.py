# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

from typing import Dict, Callable, List

import pandas as pd
import itertools

from databalanceanalysis.databalanceanalysis.constants import Measures
from databalanceanalysis.databalanceanalysis.feature_functions import FeatureFunctions

"""
 The output is a dictionary that maps the sensitive column table to Pandas dataframe containing the following
  - A feature value within the sensitive feature.
  - Another feature value within the sensitive feature.
  - It contains the following measures of the gaps between the two classes
    - Demographic Parity - https://en.wikipedia.org/wiki/Fairness_(machine_learning)
    - Pointwise Mutual Information - https://en.wikipedia.org/wiki/Pointwise_mutual_information
    - Sorensen-Dice Coefficient - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    - Jaccard Index - https://en.wikipedia.org/wiki/Jaccard_index
    - Kendall Rank Correlation - https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
    - Log-Likelihood Ratio - https://en.wikipedia.org/wiki/Likelihood_function#Likelihood_ratio
    - t-test - https://en.wikipedia.org/wiki/Student's_t-test
This output dataframe contains a row per combination of feature values for each sensitive feature.
"""


class FeatureBalanceMeasure:
    CLASS_A = "classA"
    CLASS_B = "classB"

    FEATURE_METRICS: Dict[Measures, Callable[[float, float, float, float], float]] = {
        Measures.DEMOGRAPHIC_PARITY: FeatureFunctions.get_demographic_parity,
        Measures.POINTWISE_MUTUAL_INFO: FeatureFunctions.get_point_mutual,
        Measures.SD_COEF: FeatureFunctions.get_sorenson_dice,
        Measures.JACCARD_INDEX: FeatureFunctions.get_jaccard_index,
        Measures.KR_CORRELATION: FeatureFunctions.get_kr_correlation,
        Measures.LOG_LIKELIHOOD: FeatureFunctions.get_log_likelihood_ratio,
        Measures.TTEST: FeatureFunctions.get_t_test_stat,
    }

    OVERALL_METRICS: Dict[Tuple[], Callable[[float,int], float]] = {
        (Measures.TTEST_PVALUE, Measures.TTEST): FeatureFunctions.get_t_test_p_value,
    }

    def __init__(self, df: pd.DataFrame, sensitive_cols: List[str], label_col: str):
        self._df = df
        self._sensitive_cols = sensitive_cols
        self._label_col = label_col
        self._feature_measures = self.get_all_gaps(df, sensitive_cols, label_col)

    def get_individual_feature_measures(self, df: pd.DataFrame, sensitive_col: str, label_col: str, label_pos_val: any=1):
        num_rows = df.shape[0]
        p_feature_col = df[sensitive_col].value_counts().rename("p_feature") / num_rows
        p_pos_feature_col = (
            df[df[label_col] == label_pos_val][sensitive_col].value_counts().rename("p_pos_feature")
            / num_rows
        )
        new_df = pd.concat([p_feature_col, p_pos_feature_col], axis=1)
        new_df["p_pos"] = df[df[label_col] == label_pos_val].shape[0] / num_rows
        for measure, func in self.FEATURE_METRICS.items():
            new_df[measure] = new_df.apply(
                lambda x: func(x["p_pos"], x["p_feature"], x["p_pos_feature"], num_rows), axis=1
            )
        return new_df

    # dataframe version with a column for the classes and then column for each gap measure
    def get_gaps(self, df: pd.DataFrame, sensitive_col: str, label_col: str) -> pd.DataFrame:
        metrics_df = self.get_individual_feature_measures(df, sensitive_col, label_col)
        unique_vals = df[sensitive_col].unique()
        # list of tuples of the pairings of classes
        pairs = list(itertools.combinations(unique_vals, 2))
        gap_df = pd.DataFrame(
            pairs,
            columns=[FeatureBalanceMeasure.CLASS_A, FeatureBalanceMeasure.CLASS_B],
        )
        for measure in self.FEATURE_METRICS.keys():
            classA_metric = gap_df[FeatureBalanceMeasure.CLASS_A].apply(
                lambda x: metrics_df.loc[x]
            )[measure]
            classB_metric = gap_df[FeatureBalanceMeasure.CLASS_B].apply(lambda x: metrics_df.loc[x])[
                measure
            ]
            gap_df[measure] = classA_metric - classB_metric
        # TODO add feature overall metrics
        for (measure, test_stat), func in self.OVERALL_METRICS.items():
            gap_df[measure] = gap_df[test_stat].apply(lambda x: func(test_stat, len(unique_vals)))
        return gap_df

    # gives dictionary with all the gaps between class a and class b
    def get_gaps_given_classes(self, sensitive_col: str, class_a: str, class_b: str) -> Dict[Measures: float]:
        curr_df = self._feature_measures[sensitive_col]
        return curr_df[
            (curr_df[FeatureBalanceMeasure.CLASS_A] == class_a) & (curr_df[FeatureBalanceMeasure.CLASS_B] == class_b)
        ].to_dict("records")

    def get_all_gaps(self, df: pd.DataFrame, sensitive_cols: List[str], label_col: str) -> Dict[str, pd.DataFrame]:
        gap_dict = {}
        for col in sensitive_cols:
            gap_dict[col] = self.get_gaps(df, col, label_col)
        return gap_dict

    @property
    def measures(self) -> Dict[str, pd.DataFrame]:
        return self._feature_measures
