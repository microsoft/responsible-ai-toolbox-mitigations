import pytest
from pytest import approx
import pandas as pd
import sys
import os


sys.path.append("../../../responsible-ai-mitigations")
from raimitigations.databalanceanalysis import DistributionBalanceMeasure


@pytest.fixture
def small_df():
    filepath = "test/databalanceanalysis/"
    return pd.read_csv(os.path.join(os.getcwd(), filepath + "test_df.csv"))


def test_distribution_measures(small_df):
    feature1 = small_df.columns[1]
    feature2 = small_df.columns[2]
    dist_measures = DistributionBalanceMeasure([feature1, feature2])
    exp_feature_1 = {
        "FeatureName": feature1,
        "kl_divergence": 0.03775534151008829,
        "js_dist": 0.09785224086736323,
        "inf_norm_dist": 0.1111111111111111,
        "total_variation_dist": 0.1111111111111111,
        "wasserstein_dist": 0.07407407407407407,
        "chi_sq_stat": 0.6666666666666666,
        "chi_sq_p_value": 0.7165313105737893,
    }
    df = dist_measures.measures(small_df)
    gender_measures = df.loc[df["FeatureName"] == feature1].iloc[0].to_dict()
    assert gender_measures["FeatureName"] == exp_feature_1["FeatureName"]
    assert gender_measures["kl_divergence"] == approx(exp_feature_1["kl_divergence"])
    assert gender_measures["js_dist"] == approx(exp_feature_1["js_dist"])
    assert gender_measures["inf_norm_dist"] == approx(exp_feature_1["inf_norm_dist"])
    assert gender_measures["total_variation_dist"] == approx(
        exp_feature_1["total_variation_dist"]
    )
    assert gender_measures["wasserstein_dist"] == approx(
        exp_feature_1["wasserstein_dist"]
    )
    assert gender_measures["chi_sq_stat"] == approx(exp_feature_1["chi_sq_stat"])
    assert gender_measures["chi_sq_p_value"] == approx(exp_feature_1["chi_sq_p_value"])

