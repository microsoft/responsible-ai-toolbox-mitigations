from abc import abstractproperty
from pandas.core import api
import pytest
import sys
import os
import pandas as pd

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
    feature2 = small_df.columns[2]
    feat_measures = FeatureBalanceMeasure([feature1], label)
    exp_male_female = {
        "feature_name": feature1,
        "classA": gender1,
        "classB": gender2,
        "dp": 0.16666666666666669,
        "krc": 0.18801108758923135,
        "sdc": 0.1190476190476191,
        "ji": 0.20000000000000004,
        "llr": 0.6931471805599454,
        "pmi": 0.4054651081081645,
        "t_test": 0.19245008972987523,
    }

    assert feat_measures.measures(small_df).iloc[0].to_dict()[
        "feature_name"
    ] == pytest.approx(exp_male_female["feature_name"])
