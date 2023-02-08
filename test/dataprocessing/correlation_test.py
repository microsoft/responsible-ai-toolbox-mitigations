import os
import pytest
import conftest as utils
from raimitigations.dataprocessing import (
    CorrelatedFeatures,
    BasicImputer,
    DataMinMaxScaler,
    DataStandardScaler,
)


def _run_assertions(new_df, cor_feat, include_label=True):
    assert utils.check_valid_columns(new_df.columns.to_list(), cor_feat.get_selected_features(), include_label), (
        f"The list of selected columns is different from the columns in the final df:"
        f"\nnew_df.columns = {new_df.columns.to_list()}\n"
        f"cor_feat.get_selected_features() = {cor_feat.get_selected_features()}"
    )

    fixed_cols = [col for col in cor_feat.df_info.columns.to_list() if col not in cor_feat.cor_features]
    assert utils.check_fixed_col(fixed_cols, cor_feat.get_selected_features()), (
        f"The selected features does not include the fixed features.\n"
        f"fixed features = {fixed_cols}\n"
        f"selected features = {cor_feat.get_selected_features()}"
    )


# -----------------------------------
def _get_object_list(df=None, label_col=None, X=None, y=None, use_index=True):
    utils.check_valid_input(df, label_col, X, y)

    cor_feat_list = []

    if use_index:
        cor_features = [0, 5, 2, 8, 7, 10]
    else:
        cor_features = ["num_0", "num_5", "num_2", "num_c1_num_1", "num_c0_num_0", "CN_1_num_1"]

    cor_feat = CorrelatedFeatures(df=df, label_col=label_col, X=X, y=y)
    cor_feat_list.append(cor_feat)

    cor_feat = CorrelatedFeatures(
        df=df,
        label_col=label_col,
        X=X,
        y=y,
        method_num_num=["kendall", "spearman", "pearson"],
        num_corr_th=0.8,
        num_pvalue_th=0.01,
        method_num_cat="model",
        model_metrics=["f1", "auc", "precision"],
        tie_method="cardinality",
        save_json=True,
    )
    cor_feat_list.append(cor_feat)

    cor_feat = CorrelatedFeatures(
        df=df,
        label_col=label_col,
        X=X,
        y=y,
        method_num_num=None,
        method_cat_cat=None,
        method_num_cat="jensen",
        jensen_th=0.7,
        tie_method="var",
        save_json=True,
    )
    cor_feat_list.append(cor_feat)

    cor_feat = CorrelatedFeatures(
        df=df,
        label_col=label_col,
        X=X,
        y=y,
        cor_features=cor_features,
        method_num_num=["kendall", "pearson"],
        method_cat_cat=None,
        method_num_cat="anova",
        levene_pvalue=0.0,
        anova_pvalue=0.01,
        omega_th=0.6,
        cat_corr_th=0.7,
        cat_pvalue_th=0.01,
        tie_method="var",
        save_json=True,
        compute_exact_matches=False,
        verbose=False,
    )
    cor_feat_list.append(cor_feat)

    return cor_feat_list


# -----------------------------------
def _run_main_commands(df, label_col, cor_feat, df_in_fit=True):
    df = df.copy()
    try:
        if df_in_fit:
            cor_feat.fit(df=df, label_col=label_col)
        else:
            cor_feat.fit()
    except Exception as error:
        error_msg = str(error)
        if "Only one class present in y_true" not in error_msg:
            raise ValueError(
                f"ERROR: the following error occured while fitting the CorrelatedFeatures class: {error_msg}"
            )
        return

    _ = cor_feat.get_summary(print_summary=True)
    new_df = cor_feat.transform(df)

    _run_assertions(new_df, cor_feat)


# -----------------------------------
def test_df_const(df_full, label_col_name):
    df = df_full

    obj_list = _get_object_list(df, label_col_name, use_index=False)
    for obj in obj_list:
        _run_main_commands(df, label_col_name, obj, df_in_fit=False)
        obj.update_selected_features(
            num_corr_th=0.5,
            num_pvalue_th=0.001,
            model_metrics=["accuracy", "precision"],
            metric_th=0.7,
            cat_corr_th=0.9,
        )

    os.remove(obj_list[0].json_summary)
    os.remove(obj_list[0].json_corr)
    os.remove(obj_list[0].json_uncorr)


# -----------------------------------
def test_xy_const(df_full, label_col_name):
    df = df_full

    X = df.drop(columns=[label_col_name])
    y = df[label_col_name]
    obj_list = _get_object_list(X=X, y=y, use_index=False)
    for obj in obj_list:
        _run_main_commands(df, label_col_name, obj, df_in_fit=False)

    os.remove(obj_list[0].json_summary)
    os.remove(obj_list[0].json_corr)
    os.remove(obj_list[0].json_uncorr)


# -----------------------------------
def test_col_name(df_full, label_col_name):
    df = df_full

    obj_list = _get_object_list(use_index=False)
    for obj in obj_list:
        _run_main_commands(df, label_col_name, obj, df_in_fit=True)

    os.remove(obj_list[0].json_summary)
    os.remove(obj_list[0].json_corr)
    os.remove(obj_list[0].json_uncorr)


# -----------------------------------
def test_col_index(df_full, label_col_index):
    df = df_full

    obj_list = _get_object_list(use_index=True)
    for obj in obj_list:
        _run_main_commands(df, label_col_index, obj, df_in_fit=True)

    os.remove(obj_list[0].json_summary)
    os.remove(obj_list[0].json_corr)
    os.remove(obj_list[0].json_uncorr)


# -----------------------------------
def test_no_col_name(df_full, label_col_index):
    df = df_full

    df.columns = [i for i in range(df.shape[1])]
    obj_list = _get_object_list(use_index=True)
    for obj in obj_list:
        _run_main_commands(df, label_col_index, obj, df_in_fit=True)

    os.remove(obj_list[0].json_summary)
    os.remove(obj_list[0].json_corr)
    os.remove(obj_list[0].json_uncorr)


# -----------------------------------
def test_numpy_in(df_full, label_col_name, label_col_index):
    df = df_full
    X = df.drop(columns=[label_col_name])
    y = df[label_col_name]
    Xnp = X.to_numpy()
    ynp = y.to_numpy()
    dfnp = df.to_numpy()

    obj = CorrelatedFeatures()
    obj.fit(X=Xnp, y=ynp)
    new_df = obj.transform(Xnp)
    _run_assertions(new_df, obj, include_label=False)

    obj = CorrelatedFeatures()
    obj.fit(df=dfnp, label_col=label_col_index)
    new_df = obj.transform(dfnp)
    _run_assertions(new_df, obj)

    y = df[[label_col_name]]
    obj = CorrelatedFeatures()
    obj.fit(X=X, y=y)
    new_df = obj.transform(X)
    _run_assertions(new_df, obj, include_label=False)

    os.remove(obj.json_summary)
    os.remove(obj.json_corr)
    os.remove(obj.json_uncorr)


# -----------------------------------
def test_errors(df_full, label_col_name):
    df = df_full
    X = df.drop(columns=[label_col_name])
    y = df[label_col_name]
    obj = CorrelatedFeatures()
    with pytest.raises(Exception):
        obj.fit(df=df, label_col=[label_col_name])
    with pytest.raises(Exception):
        obj.fit(df=df, label_col=None)
    with pytest.raises(Exception):
        obj.fit(df=None, label_col=label_col_name)
    with pytest.raises(Exception):
        obj.fit(df=df, label_col=99)
    with pytest.raises(Exception):
        obj.fit(X=None, y=y)
    with pytest.raises(Exception):
        obj.fit(X=X, y=None)
    with pytest.raises(Exception):
        obj.fit(X=X, y=2)
    with pytest.raises(Exception):
        obj.fit()

    obj.fit(df=df, label_col=label_col_name)
    df.columns = [i for i in range(df.shape[1])]
    with pytest.raises(Exception):
        obj.transform(df)

    imputer = BasicImputer()
    s1 = DataMinMaxScaler(include_cols=["num_0", "num_1", "num_2"])
    scaler = DataStandardScaler(include_cols=["num_3", "num_4", "num_5"], transform_pipe=[imputer, s1])
    with pytest.raises(Exception):
        obj = CorrelatedFeatures(transform_pipe=scaler)
    with pytest.raises(Exception):
        obj = CorrelatedFeatures(transform_pipe=[scaler])

    os.remove(obj.json_summary)
    os.remove(obj.json_corr)
    os.remove(obj.json_uncorr)

    with pytest.raises(Exception):
        CorrelatedFeatures(method_num_num="kendall")
    with pytest.raises(Exception):
        CorrelatedFeatures(method_num_num=["ken"])
    with pytest.raises(Exception):
        CorrelatedFeatures(method_num_cat="j")
    with pytest.raises(Exception):
        CorrelatedFeatures(method_num_cat="model", model_metrics="a")
    with pytest.raises(Exception):
        CorrelatedFeatures(method_num_cat="model", model_metrics=["a"])
    with pytest.raises(Exception):
        CorrelatedFeatures(method_cat_cat="a")
    with pytest.raises(Exception):
        CorrelatedFeatures(tie_method="a")
