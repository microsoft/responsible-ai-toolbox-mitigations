import pandas as pd
from pytest import approx
import os
from databalanceanalysis.aggregate_measures import AggregateMeasures
from databalanceanalysis.constants import Measures


def test_one_feature():
    small_df = pd.read_csv(os.path.join(os.getcwd(), "test_df.csv"))
    feature1 = small_df.columns[1]
    agg_measures = AggregateMeasures(small_df, [feature1])
    expected = {
        Measures.ATKINSON_INDEX: 0.03850028646172776,
        Measures.THIEL_L_INDEX: 0.039261011885461196,
        Measures.THIEL_T_INDEX: 0.03775534151008828,
    }
    assert agg_measures.aggregate_measures == approx(expected)


def test_both_features():
    small_df = pd.read_csv(os.path.join(os.getcwd(), "test_df.csv"))
    feature1 = small_df.columns[1]
    feature2 = small_df.columns[2]
    agg_measures = AggregateMeasures(small_df, [feature1, feature2])
    expected = {
        Measures.ATKINSON_INDEX: 0.030659793186437745,
        Measures.THIEL_L_INDEX: 0.03113963808639034,
        Measures.THIEL_T_INDEX: 0.03624967113471546,
    }
    assert agg_measures.aggregate_measures == approx(expected)
