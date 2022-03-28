import pytest
import sys
import os
import pandas as pd

from raimitigations.databalanceanalysis import AggregateBalanceMeasure
from raimitigations.databalanceanalysis import Measures


@pytest.fixture
def small_df():
    filepath = "test/databalanceanalysis/"
    return pd.read_csv(os.path.join(os.getcwd(), filepath + "test_df.csv"))


def test_one_feature(small_df):
    feature1 = small_df.columns[1]
    agg_measures = AggregateBalanceMeasure([feature1])
    expected = {
        "atkinson_index": 0.03850028646172776,
        "theil_l_index": 0.039261011885461196,
        "theil_t_index": 0.03775534151008828,
    }
    assert agg_measures.measures(small_df).iloc[0].to_dict() == pytest.approx(expected)


def test_both_features(small_df):
    feature1 = small_df.columns[1]
    feature2 = small_df.columns[2]
    agg_measures = AggregateBalanceMeasure([feature1, feature2])
    expected = {
        "atkinson_index": 0.030659793186437745,
        "theil_l_index": 0.03113963808639034,
        "theil_t_index": 0.03624967113471546,
    }
    assert agg_measures.measures(small_df).iloc[0].to_dict() == pytest.approx(expected)
