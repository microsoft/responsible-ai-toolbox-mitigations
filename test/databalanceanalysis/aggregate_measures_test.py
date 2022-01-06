import pytest
import sys

sys.path.append("../../../ResponsibleAIToolbox-Mitigation")
from databalanceanalysis.databalanceanalysis.aggregate_measures import (
    AggregateBalanceMeasure,
)
from databalanceanalysis.databalanceanalysis.constants import Measures


def test_one_feature(small_df):
    feature1 = small_df.columns[1]
    agg_measures = AggregateBalanceMeasure([feature1])
    expected = {
        Measures.ATKINSON_INDEX: 0.03850028646172776,
        Measures.THEIL_L_INDEX: 0.039261011885461196,
        Measures.THEIL_T_INDEX: 0.03775534151008828,
    }
    assert agg_measures.measures(small_df).iloc[0].to_dict() == pytest.approx(expected)


def test_both_features(small_df):
    feature1 = small_df.columns[1]
    feature2 = small_df.columns[2]
    agg_measures = AggregateBalanceMeasure([feature1, feature2])
    expected = {
        Measures.ATKINSON_INDEX: 0.030659793186437745,
        Measures.THEIL_L_INDEX: 0.03113963808639034,
        Measures.THEIL_T_INDEX: 0.03624967113471546,
    }
    assert agg_measures.measures(small_df).iloc[0].to_dict() == pytest.approx(expected)
