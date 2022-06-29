from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC
import pytest
import conftest as utils
from raimitigations.dataprocessing import (
    Rebalance,
    EncoderOrdinal,
    DataMinMaxScaler,
    DataStandardScaler,
)


# -----------------------------------
def _get_object_list(cat_cols, df=None, label_col=None, X=None, y=None):
    utils.check_valid_input(df, label_col, X, y)

    rebalance_list = []

    rebalance = Rebalance(df=df, rebalance_col=label_col, X=X, y=y)
    rebalance_list.append(rebalance)

    rebalance = Rebalance(df=df, rebalance_col=label_col, X=X, y=y, cat_col=cat_cols, k_neighbors=6, under_sampler=True)
    rebalance_list.append(rebalance)

    scaler = DataStandardScaler()
    rebalance = Rebalance(
        df=df,
        rebalance_col=label_col,
        X=X,
        y=y,
        transform_pipe=[scaler],
        over_sampler=False,
        under_sampler=RandomUnderSampler(),
    )
    rebalance_list.append(rebalance)

    enc = EncoderOrdinal()
    scaler = DataMinMaxScaler(include_cols=cat_cols)
    rebalance = Rebalance(df=df, rebalance_col=label_col, X=X, y=y, cat_col=cat_cols, transform_pipe=[enc, scaler])
    rebalance_list.append(rebalance)

    rebalance = Rebalance(
        df=df, rebalance_col=label_col, X=X, y=y, strategy_over="not minority", strategy_under="majority"
    )
    rebalance_list.append(rebalance)

    rebalance = Rebalance(
        df=df, rebalance_col=label_col, X=X, y=y, strategy_over="minority", strategy_under="not majority"
    )
    rebalance_list.append(rebalance)

    rebalance = Rebalance(df=df, rebalance_col=label_col, X=X, y=y, strategy_over="all")
    rebalance_list.append(rebalance)

    return rebalance_list


# -----------------------------------
def _run_main_commands(df, label_col, transf, df_in_fit=True):
    df = df.copy()
    if df_in_fit:
        _ = transf.fit_resample(df=df, rebalance_col=label_col)
    else:
        _ = transf.fit_resample()


# -----------------------------------
def test_df_const(df_full_nan, label_col_name):
    df = df_full_nan

    cat_col = ["CN_1_num_1", "CC_1_num_1", "CN_0_num_0", "CC_0_num_0"]
    obj_list = _get_object_list(cat_col, df, label_col_name)
    for obj in obj_list:
        _run_main_commands(df, label_col_name, obj, df_in_fit=False)


# -----------------------------------
def test_xy_const(df_full_nan, label_col_name):
    df = df_full_nan

    X = df.drop(columns=[label_col_name])
    y = df[label_col_name]
    cat_col = ["CN_1_num_1", "CC_1_num_1", "CN_0_num_0", "CC_0_num_0"]
    obj_list = _get_object_list(cat_col, X=X, y=y)
    for obj in obj_list:
        _run_main_commands(df, label_col_name, obj, df_in_fit=False)


# -----------------------------------
def test_xy_const_rebalance_col(df_full_nan, label_col_name):
    df = df_full_nan

    X = df.drop(columns=["CN_1_num_1"])
    y = df["CN_1_num_1"]
    cat_col = ["CC_1_num_1", "CN_0_num_0", "CC_0_num_0"]
    obj_list = _get_object_list(cat_col, X=X, y=y)
    for obj in obj_list:
        _run_main_commands(df, label_col_name, obj, df_in_fit=False)


# -----------------------------------
def test_col_name(df_full_nan, label_col_name):
    df = df_full_nan

    cat_col = ["CN_1_num_1", "CC_1_num_1", "CN_0_num_0", "CC_0_num_0"]
    obj_list = _get_object_list(cat_col)
    for obj in obj_list:
        _run_main_commands(df, label_col_name, obj, df_in_fit=True)


# -----------------------------------
def test_col_index(df_full_nan, label_col_index):
    df = df_full_nan

    cat_col = [10, 12, 9, 11]
    obj_list = _get_object_list(cat_col)
    for obj in obj_list:
        _run_main_commands(df, label_col_index, obj, df_in_fit=True)


# -----------------------------------
def test_no_col_name(df_full_nan, label_col_index):
    df = df_full_nan

    cat_col = [10, 12, 9, 11]
    df.columns = [i for i in range(df.shape[1])]
    obj_list = _get_object_list(cat_col)
    for obj in obj_list:
        _run_main_commands(df, label_col_index, obj, df_in_fit=True)


# -----------------------------------
def test_errors(df_full_nan, label_col_name):
    df = df_full_nan
    obj = Rebalance()
    with pytest.raises(Exception):
        obj.fit_resample(df=df, rebalance_col="CN_0_num_0")
    with pytest.raises(Exception):
        obj.fit_resample(df=df, rebalance_col="num_1")

    obj = Rebalance(strategy_over="a")
    with pytest.raises(Exception):
        obj.fit_resample(df=df, rebalance_col=label_col_name)

    obj = Rebalance(strategy_over=1)
    with pytest.raises(Exception):
        obj.fit_resample(df=df, rebalance_col=label_col_name)

    obj = Rebalance(over_sampler=SMOTEN(), under_sampler=True)
    with pytest.raises(Exception):
        obj.fit_resample(df=df, rebalance_col=label_col_name)

    df_num = df.copy()
    df_num = df_num[["num_0", "num_1", "num_2", label_col_name]]
    df_cat = df.copy()
    df_cat = df_cat[["CN_0_num_0", "CN_1_num_1", label_col_name]]

    obj = Rebalance(over_sampler=SMOTEN())
    with pytest.raises(Exception):
        obj.fit_resample(df=df, rebalance_col=label_col_name)

    obj = Rebalance(over_sampler=SMOTEN())
    with pytest.raises(Exception):
        obj.fit_resample(df=df_num, rebalance_col=label_col_name)

    obj = Rebalance(over_sampler=SMOTENC(categorical_features=["CN_0_num_0", "CN_1_num_1"]))
    with pytest.raises(Exception):
        obj.fit_resample(df=df_cat, rebalance_col=label_col_name)

    obj = Rebalance(under_sampler=TomekLinks(), strategy_under=0.2)
    with pytest.raises(Exception):
        obj.fit_resample(df=df_cat, rebalance_col=label_col_name)

    obj = Rebalance(over_sampler=False, under_sampler=False)
    with pytest.raises(Exception):
        obj.fit_resample(df=df, rebalance_col=label_col_name)

    obj = Rebalance(over_sampler=10)
    with pytest.raises(Exception):
        obj.fit_resample(df=df, rebalance_col=label_col_name)

    obj = Rebalance(under_sampler=10)
    with pytest.raises(Exception):
        obj.fit_resample(df=df, rebalance_col=label_col_name)

    obj = Rebalance(k_neighbors="a")
    with pytest.raises(Exception):
        obj.fit_resample(df=df, rebalance_col=label_col_name)