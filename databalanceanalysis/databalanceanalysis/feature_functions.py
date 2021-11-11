import numpy as np

# Demographic Parity

# given a column and a class, get the total number from that class


def get_total_count(df):
    return df.size[0]


def p_feature(df, sensitive_col_name, feature_value):
    return (df.loc[df[sensitive_col_name] == feature_value]).size[0] / get_total_count(
        df
    )


def p_pos_feature(
    df, sensitive_col_name, label_col_name, feature_value, pos_label_value
):
    return df.loc[
        df[sensitive_col_name] == feature_value & df[label_col_name] == pos_label_value
    ].size / get_total_count(df)


def p_pos(df, label_col_name, pos_label_value=1):
    return df.loc[df[label_col_name] == pos_label_value].size[0] / get_total_count(df)


def get_prob_class(df, sensitive_col_name, feature_value, label_col_name):
    total = p_feature(df, sensitive_col_name, feature_value)
    pos = p_pos_feature(df, sensitive_col_name, label_col_name, feature_value)
    return pos / total


def get_demographic_parity(p_pos, p_feature, p_pos_feature):
    return p_pos_feature / p_feature


def get_point_mutual(p_pos, p_feature, p_pos_feature):
    dp = get_demographic_parity(p_pos, p_feature, p_pos_feature)
    if dp == 0:
        return -np.inf
    else:
        return np.log(dp)


def get_sorenson_dice(p_pos, p_feature, p_pos_feature):
    return p_pos_feature / (p_feature + p_pos)


def get_jaccard_index(p_pos, p_feature, p_pos_feature):
    return p_pos_feature / (p_feature + p_pos - p_pos_feature)


def get_kendall_rank(total_count, p_feature, p_pos, p_pos_feature):
    a = np.pow(total_count, 2) * (
        1 - 2 * p_feature - 2 * p_pos + 2 * p_pos_feature + 2 * p_pos * p_feature
    )
    b = total_count * (2 * p_feature + 2 * p_pos - 4 * p_pos_feature - 1)
    c = np.pow(total_count, 2) * np.sqrt(
        (p_feature - np.pow(p_feature, 2)) * (p_pos - np.pow(p_pos, 2))
    )
    return (a + b) / c


def log_likelihood_ratio(p_pos, p_feature, p_pos_feature):
    return np.log(p_pos_feature / p_pos)


def t_test_value(p_pos, p_feature, p_pos_feature):
    return (p_pos - (p_feature * p_pos)) / np.sqrt(p_feature * p_pos)


def t_test_pvalue(t_statistic, n):
    return scipy.stats.t.sf(np.abs(t_statistic), n - 1)
