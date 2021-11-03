import pandas as pd
import itertools

from constants import feature_measures_to_func


class FeatureMeasures:
    def __init__(self, df, sensitive_cols, label_col):
        self._df = df
        self._sensitive_cols = sensitive_cols
        self._label_col = label_col

def get_feature_metrics(df, sensitive_col, label_col):
    # TODO check that label col is 0 or 1  column
    num_rows = df.shape[0]
    feature_values = df[sensitive_col].unique()
    p_feature_col = df[sensitive_col].value_counts().rename("p_feature") / num_rows
    p_pos_feature_col = df[df[label_col] == 1][sensitive_col].value_counts().rename("p_pos_feature") / num_rows
    new_df = pd.concat([p_feature_col, p_pos_feature_col], axis=1)
    new_df["p_pos"] = df[df[label_col] == 1].shape[0] / num_rows
    for metric, func in feature_measures_to_func.items():
        new_df[metric.value] = new_df.apply(lambda x: func(x["p_pos"], x["p_feature"], x["p_pos_feature"]), axis=1)
    return new_df

def get_gaps(df, sensitive_col, label_col):
    metrics_df = get_feature_metrics(df, sensitive_col, label_col)
    unique_vals = df[sensitive_col].unique()
    pairs = list(itertools.combinations(unique_vals, 2))  # list of tuples of the pairings of classes
    gap_df = pd.DataFrame(pairs, columns=["classA", "classB"])
    for measure in feature_measures_to_func.keys() :
        classA_metric = gap_df["classA"].apply(lambda x: metrics_df.loc[x])[measure.value]
        classB_metric = gap_df["classB"].apply(lambda x: metrics_df.loc[x])[measure.value]
        gap_df[measure.value + "_gap"] = classA_metric-classB_metric
    return gap_df

def get_all_gaps(df, sensitive_cols, label_col):
    gap_dict  ={} 
    for col in sensitive_cols:
        gap_dict[col] = get_gaps(df, col, label_col)
    return gap_dict
