from abc import abstractproperty
from pandas.core import api
import pytest
import sys
import os
import pandas as pd

sys.path.append("../../../responsible-ai-mitigations")
from raimitigations.databalanceanalysis import FeatureBalanceMeasure

# run the tests using this command: python -m pytest


@pytest.fixture
def small_df():
    filepath = "test/databalanceanalysis/"
    return pd.read_csv(os.path.join(os.getcwd(), filepath + "test_df.csv"))


def test_feature_balance_measures(small_df):
    label = small_df.columns[0]
    feature1 = small_df.columns[1]
    gender1 = small_df[feature1].unique()[0]
    gender2 = small_df[feature1].unique()[1]
    feat_measures = FeatureBalanceMeasure([feature1], label)
    exp_male_female = {
        "FeatureName": feature1,
        "ClassA": gender1,
        "ClassB": gender2,
        "dp": 0.16666666666666669,
        "krc": 0.18801108758923135,
        "sdc": 0.1190476190476191,
        "ji": 0.20000000000000004,
        "llr": 0.6931471805599454,
        "pmi": 0.4054651081081645,
        "t_test": 0.19245008972987523,
    }
    feat_dict = feat_measures.measures(small_df).iloc[0].to_dict()
    assert feat_dict["FeatureName"] == exp_male_female["FeatureName"]
    assert feat_dict["ClassA"] == exp_male_female["ClassA"]
    assert feat_dict["ClassB"] == exp_male_female["ClassB"]
    assert feat_dict["dp"] == exp_male_female["dp"]
    assert feat_dict["krc"] == exp_male_female["krc"]
    assert feat_dict["sdc"] == exp_male_female["sdc"]
    assert feat_dict["ji"] == exp_male_female["ji"]
    assert feat_dict["llr"] == exp_male_female["llr"]
    assert feat_dict["pmi"] == exp_male_female["pmi"]
