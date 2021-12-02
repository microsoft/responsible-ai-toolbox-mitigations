import pandas as pd
import os
from databalanceanalysis.feature_measures import FeatureBalanceMeasures
from databalanceanalysis.constants import Measures

# run the tests using this command: python -m pytest


def test_feature_balance_measures():
    small_df = pd.read_csv(os.path.join(os.getcwd(), "test_df.csv"))
    label = small_df.columns[0]
    feature1 = small_df.columns[1]
    gender1 = small_df[feature1].unique()[0]
    gender2 = small_df[feature1].unique()[1]
    feature2 = small_df.columns[2]
    feat_measures = FeatureBalanceMeasures(small_df, [feature1, feature2], label)
    exp_male_female = {
        Measures.DEMOGRAPHIC_PARITY: 0.16666666666666669,
        Measures.SD_COEF: 0.1190476190476191,
        Measures.JACCARD_INDEX: 0.20000000000000004,
        Measures.LOG_LIKELIHOOD: 0.6931471805599454,
        Measures.POINTWISE_MUTUAL_INFO: 0.4054651081081645,
        Measures.TTEST: 0.19245008972987523,
    }
    gender_measures = feat_measures.measures[feature1]
    print(feat_measures.get_gaps_given_classes(feature1, gender1, gender2))
    print(gender_measures)
