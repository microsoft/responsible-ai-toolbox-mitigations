import pytest
import pandas as pd

from databalanceanalysis.databalanceanalysis.distribution_measures import (
    DistributionBalanceMeasure,
)
from databalanceanalysis.databalanceanalysis.constants import Measures


def test_distribution_measures(small_df):
    feature1 = small_df.columns[1]
    feature2 = small_df.columns[2]
    dist_measures = DistributionBalanceMeasure([feature1, feature2])
    exp_feature_1 = {
        "feature_name": feature1,
        Measures.KL_DIVERGENCE: 0.03775534151008829,
        Measures.JS_DISTANCE: 0.09785224086736323,
        Measures.INF_NORM_DISTANCE: 0.1111111111111111,
        Measures.TOTAL_VARIANCE_DISTANCE: 0.1111111111111111,
        Measures.WS_DISTANCE: 0.07407407407407407,
        Measures.CHISQ: 0.6666666666666666,
        Measures.CHISQ_PVALUE: 0.7165313105737893,
    }
    df = dist_measures.measures(small_df)
    gender_measures = df.loc[df["feature_name"] == feature1].iloc[0].to_dict()
    print(gender_measures)
    assert gender_measures == pytest.approx(exp_feature_1)
