import pytest
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
        "feature_name": feature1,
        "kl_divergence": 0.03775534151008829,
        "js_dist": 0.09785224086736323,
        "inf_norm_dist": 0.1111111111111111,
        "total_variation_dist": 0.1111111111111111,
        "wasserstein_dist": 0.07407407407407407,
        "chi_sq_stat": 0.6666666666666666,
        "chi_sq_p_value": 0.7165313105737893,
    }
    df = dist_measures.measures(small_df)
    gender_measures = df.loc[df["feature_name"] == feature1].iloc[0].to_dict()
    print(gender_measures)
    assert gender_measures == pytest.approx(exp_feature_1)
