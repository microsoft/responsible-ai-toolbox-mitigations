import os
import pytest
from sklearn.tree import DecisionTreeClassifier
import conftest as utils
from raimitigations.dataprocessing import (
    SeqFeatSelection,
    EncoderOrdinal,
    CatBoostSelection,
    DataMinMaxScaler,
    CorrelatedFeatures,
)


# -----------------------------------
def _get_object_list(df=None, label_col=None, X=None, y=None, use_index=True):
    utils.check_valid_input(df, label_col, X, y)

    feat_sel_list = []

    if use_index:
        include_cols1 = [0, 5]
        include_cols2 = [1, 2, 3, 4]
        fixed_cols = [0, 1, 2]
        cat_col = [9, 10]
    else:
        include_cols1 = ["num_0", "num_5"]
        include_cols2 = ["num_1", "num_2", "num_3", "num_4"]
        fixed_cols = ["num_0", "num_1", "num_2"]
        cat_col = ["CN_0_num_0", "CN_1_num_1"]

    enc = EncoderOrdinal()
    scaler1 = DataMinMaxScaler(include_cols=include_cols1)
    scaler2 = DataMinMaxScaler(include_cols=include_cols2)
    feat_sel = SeqFeatSelection(
        df=df, label_col=label_col, X=X, y=y, scoring="roc_auc", n_jobs=4, transform_pipe=[enc, scaler1, scaler2]
    )
    feat_sel_list.append(feat_sel)

    feat_sel = SeqFeatSelection(
        df=df,
        label_col=label_col,
        X=X,
        y=y,
        estimator=DecisionTreeClassifier(max_features="sqrt"),
        n_feat=(5, 7),
        fixed_cols=fixed_cols,
        n_jobs=4,
        save_json=True,
    )
    feat_sel_list.append(feat_sel)

    feat_sel = CatBoostSelection(
        df=df,
        label_col=label_col,
        X=X,
        y=y,
        catboost_log=True,
        n_feat=7,
        fixed_cols=fixed_cols,
        catboost_plot=False,
        steps=1,
        algorithm="shap",
        save_json=True,
    )
    feat_sel_list.append(feat_sel)

    feat_sel = CatBoostSelection(
        df=df,
        label_col=label_col,
        X=X,
        y=y,
        catboost_log=False,
        catboost_plot=False,
        cat_col=cat_col,
        steps=1,
        algorithm="predict",
        save_json=False,
    )
    feat_sel_list.append(feat_sel)

    return feat_sel_list


# -----------------------------------
def _run_main_commands(df, label_col, feat_sel, df_in_fit=True):
    df = df.copy()
    if df_in_fit:
        feat_sel.fit(df=df, label_col=label_col)
    else:
        feat_sel.fit()

    _ = feat_sel.get_summary()
    new_df = feat_sel.transform(df)

    assert utils.check_valid_columns(new_df.columns.to_list(), feat_sel.get_selected_features()), (
        f"The list of selected columns is different from the columns in the final df:"
        f"\nnew_df.columns = {new_df.columns.to_list()}\n"
        f"cor_feat.get_selected_features() = {feat_sel.get_selected_features()}"
    )

    if feat_sel.fixed_cols is not None:
        assert utils.check_fixed_col(feat_sel.fixed_cols, feat_sel.get_selected_features()), (
            f"The selected features does not include the fixed features.\n"
            f"fixed features = {feat_sel.fixed_cols}\n"
            f"selected features = {feat_sel.get_selected_features()}"
        )

    if feat_sel.save_json:
        os.remove(feat_sel.json_summary)


# -----------------------------------
def test_df_const(df_full, label_col_name):
    df = df_full

    obj_list = _get_object_list(df, label_col_name, use_index=False)
    for obj in obj_list:
        _run_main_commands(df, label_col_name, obj, df_in_fit=False)


# -----------------------------------
def test_xy_const(df_full, label_col_name):
    df = df_full

    X = df.drop(columns=[label_col_name])
    y = df[label_col_name]
    obj_list = _get_object_list(X=X, y=y, use_index=False)
    for obj in obj_list:
        _run_main_commands(df, label_col_name, obj, df_in_fit=False)


# -----------------------------------
def test_col_name(df_full, label_col_name):
    df = df_full

    obj_list = _get_object_list(use_index=False)
    for obj in obj_list:
        _run_main_commands(df, label_col_name, obj, df_in_fit=True)


# -----------------------------------
def test_col_index(df_full, label_col_index):
    df = df_full

    obj_list = _get_object_list(use_index=True)
    for obj in obj_list:
        _run_main_commands(df, label_col_index, obj, df_in_fit=True)


# -----------------------------------
def test_no_col_name(df_full, label_col_index):
    df = df_full

    df.columns = [i for i in range(df.shape[1])]
    obj_list = _get_object_list(use_index=True)
    for obj in obj_list:
        _run_main_commands(df, label_col_index, obj, df_in_fit=True)


# -----------------------------------
def test_regression_seqfeat(df_regression, label_col_name):
    df = df_regression
    label_col = label_col_name

    feat_sel = SeqFeatSelection(verbose=False)
    feat_sel.fit(df=df, label_col=label_col)
    new_df = feat_sel.transform(df)
    feat_sel.get_summary()

    assert utils.check_valid_columns(new_df.columns.to_list(), feat_sel.get_selected_features()), (
        f"The list of selected columns is different from the columns in the final df:"
        f"\nnew_df.columns = {new_df.columns.to_list()}\n"
        f"feat_sel.get_selected_features() = {feat_sel.get_selected_features()}"
    )

    assert feat_sel.regression, "ERROR: Regression task not detected in the regression test."


# -----------------------------------
def test_regression_catboost(df_regression, label_col_name):
    df = df_regression
    label_col = label_col_name
    n_feat = 7
    fixed_cols = ["num_0", "CN_1_num_1"]

    feat_sel = CatBoostSelection(
        catboost_log=False,
        n_feat=n_feat,
        fixed_cols=fixed_cols,
        catboost_plot=False,
        steps=1,
        algorithm="shap",
        verbose=False,
    )
    feat_sel.fit(df=df, label_col=label_col)
    new_df = feat_sel.transform(df)
    feat_sel.get_summary()

    assert utils.check_valid_columns(new_df.columns.to_list(), feat_sel.get_selected_features()), (
        f"The list of selected columns is different from the columns in the final df:"
        f"\nnew_df.columns = {new_df.columns.to_list()}\n"
        f"feat_sel.get_selected_features() = {feat_sel.get_selected_features()}"
    )

    assert utils.check_fixed_col(fixed_cols, feat_sel.get_selected_features()), (
        f"The selected features does not include the fixed features.\n"
        f"fixed features = {fixed_cols}\n"
        f"selected features = {feat_sel.get_selected_features()}"
    )

    assert n_feat == len(feat_sel.get_selected_features()), (
        f"The number of selected features ({len(feat_sel.get_selected_features())}) is different "
        f"from the number of features fo select (n_feat = {n_feat})"
    )

    assert feat_sel.regression, "ERROR: Regression task not detected in the regression test."


# -----------------------------------
def test_special_cases(df_full, label_col_name):
    df = df_full.copy()
    cor = CorrelatedFeatures(method_num_cat="jensen")
    obj = CatBoostSelection(transform_pipe=[cor])
    obj.fit(df=df, label_col=label_col_name)
    os.remove(cor.json_summary)
    os.remove(cor.json_corr)
    os.remove(cor.json_uncorr)


# -----------------------------------


def test_errors_seqfeat(df_full, label_col_name):
    with pytest.raises(Exception):
        SeqFeatSelection(n_feat=None),
    with pytest.raises(Exception):
        SeqFeatSelection(n_feat="b"),
    with pytest.raises(Exception):
        SeqFeatSelection(n_feat=3.0),

    obj_list = [
        SeqFeatSelection(n_feat=99),
        SeqFeatSelection(n_feat=(3, 5, 7)),
        SeqFeatSelection(n_feat=(3, "5")),
        SeqFeatSelection(n_feat=(5, 3)),
        SeqFeatSelection(n_feat=(-1, 5)),
        SeqFeatSelection(n_feat=(3, 99)),
    ]
    for obj in obj_list:
        df = df_full.copy()
        with pytest.raises(Exception):
            obj.fit(df=df, label_col=label_col_name)


# -----------------------------------


def test_errors_catboost(df_full, label_col_name):
    with pytest.raises(Exception):
        CatBoostSelection(algorithm="aaa")
    with pytest.raises(Exception):
        CatBoostSelection(estimator=DecisionTreeClassifier())

    obj_list = [
        CatBoostSelection(n_feat=4.0),
        CatBoostSelection(n_feat=-1),
        CatBoostSelection(fixed_cols=["num_0", "num_1", "num_2"], n_feat=3),
        CatBoostSelection(cat_col="aaa"),
    ]
    for obj in obj_list:
        df = df_full.copy()
        with pytest.raises(Exception):
            obj.fit(df=df, label_col=label_col_name)
