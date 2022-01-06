from abc import abstractproperty
from pandas.core import api
import pytest
import sys

sys.path.append("../../../ResponsibleAIToolbox-Mitigation")
from databalanceanalysis.databalanceanalysis.feature_measures import (
    FeatureBalanceMeasure,
)
from databalanceanalysis.databalanceanalysis.constants import Measures

# run the tests using this command: python -m pytest


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
        Measures.DEMOGRAPHIC_PARITY: 0.16666666666666669,
        Measures.KR_CORRELATION: 0.18801108758923135,
        Measures.SD_COEF: 0.1190476190476191,
        Measures.JACCARD_INDEX: 0.20000000000000004,
        Measures.LOG_LIKELIHOOD: 0.6931471805599454,
        Measures.POINTWISE_MUTUAL_INFO: 0.4054651081081645,
        Measures.TTEST: 0.19245008972987523,
    }

    assert feat_measures.measures(small_df).iloc[0].to_dict()[
        "feature_name"
    ] == pytest.approx(exp_male_female["feature_name"])
