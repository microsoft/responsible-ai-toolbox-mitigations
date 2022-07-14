import pytest
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
    QuantileTransformer,
    Normalizer,
)

from raimitigations.dataprocessing import (
    DataNormalizer,
    DataStandardScaler,
    DataMinMaxScaler,
    DataQuantileTransformer,
    DataRobustScaler,
    DataPowerTransformer,
    BasicImputer,
    CorrelatedFeatures,
)


# -----------------------------------
def _get_object_list(df=None, use_index=True):

    scaler_list = []

    if use_index:
        exclude_cols1 = [6]
        exclude_cols2 = [0, 3, 2, 1, 10, 6]
        include_cols_g1 = [[0, 1, 2], [3], [4], [5]]
    else:
        exclude_cols1 = ["label"]
        exclude_cols2 = ["num_0", "num_3", "num_2", "num_1", "CN_1_num_1", "label"]
        include_cols_g1 = [["num_0", "num_1", "num_2"], ["num_3"], ["num_4"], ["num_5"]]

    scaler = DataNormalizer(scaler_obj=Normalizer(), df=df, exclude_cols=exclude_cols1)
    scaler_list.append(scaler)

    scaler = DataStandardScaler(scaler_obj=StandardScaler(), df=df, exclude_cols=exclude_cols2)
    scaler_list.append(scaler)

    imputer = BasicImputer()
    s1 = DataPowerTransformer(scaler_obj=PowerTransformer(), include_cols=include_cols_g1[0])
    s2 = DataMinMaxScaler(scaler_obj=MinMaxScaler(), include_cols=include_cols_g1[0])
    s3 = DataQuantileTransformer(scaler_obj=QuantileTransformer(), include_cols=include_cols_g1[0])
    scaler = DataRobustScaler(
        scaler_obj=RobustScaler(),
        df=df,
        include_cols=include_cols_g1[0],
        transform_pipe=[imputer, s1, s2, s3],
    )
    scaler_list.append(scaler)

    scaler = DataNormalizer(df=df)
    scaler_list.append(scaler)

    scaler = DataStandardScaler(df=df)
    scaler_list.append(scaler)

    scaler = DataPowerTransformer(df=df)
    scaler_list.append(scaler)

    scaler = DataMinMaxScaler(df=df)
    scaler_list.append(scaler)

    scaler = DataQuantileTransformer(df=df)
    scaler_list.append(scaler)

    scaler = DataRobustScaler(df=df)
    scaler_list.append(scaler)

    return scaler_list


# -----------------------------------
def _run_main_commands(df, transf, df_in_fit=True):
    df = df.copy()
    if df_in_fit:
        transf.fit(df=df)
    else:
        transf.fit()
    new_df = transf.transform(df)

    has_inverse = hasattr(transf.__class__, "_inverse_transform")
    if has_inverse:
        _ = transf.inverse_transform(new_df)


# -----------------------------------
def test_df_const(df_full_nan):
    df = df_full_nan

    obj_list = _get_object_list(df, use_index=False)
    for obj in obj_list:
        _run_main_commands(df, obj, df_in_fit=False)


# -----------------------------------
def test_col_name(df_full_nan):
    df = df_full_nan

    obj_list = _get_object_list(df=None, use_index=False)
    for obj in obj_list:
        _run_main_commands(df, obj, df_in_fit=True)


# -----------------------------------
def test_col_index(df_full_nan):
    df = df_full_nan

    obj_list = _get_object_list(df=None, use_index=True)
    for obj in obj_list:
        _run_main_commands(df, obj, df_in_fit=True)


# -----------------------------------
def test_no_col_name(df_full_nan):
    df = df_full_nan

    df.columns = [i for i in range(df.shape[1])]
    obj_list = _get_object_list(df=None, use_index=True)
    for obj in obj_list:
        _run_main_commands(df, obj, df_in_fit=True)


# -----------------------------------
def test_errors(df_full_nan):
    with pytest.raises(Exception):
        obj = DataStandardScaler(scaler_obj=RobustScaler())
    with pytest.raises(Exception):
        obj = DataRobustScaler(scaler_obj=StandardScaler())
    with pytest.raises(Exception):
        obj = DataQuantileTransformer(scaler_obj=StandardScaler())
    with pytest.raises(Exception):
        obj = DataPowerTransformer(scaler_obj=StandardScaler())
    with pytest.raises(Exception):
        obj = DataNormalizer(scaler_obj=StandardScaler())
    with pytest.raises(Exception):
        obj = DataMinMaxScaler(scaler_obj=StandardScaler())

    df_num = df_full_nan.copy()
    num_col = [col for col in df_num.columns if "C" not in col]
    df_num.drop(columns=num_col, inplace=True)
    obj = DataStandardScaler()
    with pytest.raises(Exception):
        obj.fit(df=df_num)

    feat_sel = CorrelatedFeatures()
    obj = DataStandardScaler(transform_pipe=[feat_sel])
    with pytest.raises(Exception):
        obj.fit(df=df_full_nan)

    obj = DataNormalizer()
    obj.fit(df=df_full_nan)
    new_df = obj.transform(df_full_nan)
    with pytest.raises(Exception):
        _ = obj.inverse_transform(new_df)

    obj = DataStandardScaler(include_cols=["num_0"], exclude_cols=["num_1"])
    with pytest.raises(Exception):
        obj.fit(df=df_full_nan)
