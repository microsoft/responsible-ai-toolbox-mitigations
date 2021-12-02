import pandas as pd
import os
from databalanceanalysis.distribution_measures import DistributionBalanceMeasures
from pytest import approx


def test_distribution_measures():
    small_df = pd.read_csv(os.path.join(os.getcwd(), "test_df.csv"))
    feature1 = small_df.columns[1]
    feature2 = small_df.columns[2]
    dist_measures = DistributionBalanceMeasures(small_df, [feature1, feature2])
    exp_feature_1 = {
        "kl_divergence": 0.03775534151008829,
        "js_distance": 0.09785224086736323,
        "inf_norm_distance": 0.1111111111111111,
        "total_variance_distance": 0.1111111111111111,
        "ws_distance": 0.07407407407407407,
        "chisq": 0.6666666666666666,
        "chisq_pvalue": 0.7165313105737893,
    }
    gender_measures = dist_measures.measures[feature1]
    print(gender_measures)
    assert gender_measures == approx(exp_feature_1)
