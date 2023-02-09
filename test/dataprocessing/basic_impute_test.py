import numpy as np
import pytest
from raimitigations.dataprocessing import BasicImputer

COL_WITH_NAN = ["num_0", "num_3", "num_4", "CN_0_num_0", "CC_1_num_1"]
COL_WITH_NAN_IND = [0, 3, 4, 9, 12]


# -----------------------------------
def _get_object_list(df=None, use_index=True):

    imputer_list = []

    if use_index:
        col_impute1 = [0, 1, 2, 3, 10]
        specific_col1 = {10: {"missing_values": np.nan, "strategy": "constant", "fill_value": -100}}
        col_impute2 = [0, 1, 2, 3]
    else:
        col_impute1 = ["num_0", "num_1", "num_2", "num_3", "CN_1_num_1"]
        specific_col1 = {"CN_1_num_1": {"missing_values": np.nan, "strategy": "constant", "fill_value": -100}}
        col_impute2 = ["num_0", "num_1", "num_2", "num_3"]
    categorical = {"missing_values": np.nan, "strategy": "constant", "fill_value": "missing"}
    numerical = {"missing_values": np.nan, "strategy": "most_frequent", "fill_value": "missing"}

    imputer = BasicImputer(df=df)
    imputer_list.append(imputer)

    imputer = BasicImputer(df=df, col_impute=col_impute1, specific_col=specific_col1)
    imputer_list.append(imputer)

    imputer = BasicImputer(df=df, col_impute=col_impute2, categorical=categorical)
    imputer_list.append(imputer)

    imputer = BasicImputer(df=df, numerical=numerical)
    imputer_list.append(imputer)

    return imputer_list


# -----------------------------------
def _run_main_commands(df, transf, df_in_fit=True):
    df = df.copy()
    if df_in_fit:
        transf.fit(df=df)
    else:
        transf.fit()
    new_df = transf.transform(df)

    nan_dict = new_df.isna().any()
    nan_dict = {key: nan_dict[key] for key in nan_dict.keys() if key in transf.col_impute}
    assert not all(nan_dict.values()), (
        "ERROR: Not all missing values were removed. Something went wrong.\n"
        f"The columns with nan are: {nan_dict}\n"
        f"The columns that should've been imputed are: {transf.col_impute}"
    )


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
    obj_list = [
        BasicImputer(categorical=["list"]),
        BasicImputer(specific_col={"99": {"missing_values": np.nan, "strategy": "constant", "fill_value": -100}}),
        BasicImputer(specific_col={99: {"missing_values": np.nan, "strategy": "constant", "fill_value": -100}}),
        BasicImputer(categorical={"missing_values": np.nan, "strategy": "constant"}),
        BasicImputer(categorical={"missing_values": np.nan, "fill_value": -100}),
        BasicImputer(categorical={"strategy": "constant", "fill_value": -100}),
        BasicImputer(categorical={"missing_values": np.nan, "strategy": "constant", "fill_value": np.nan}),
    ]

    for obj in obj_list:
        df_cp = df_full_nan.copy()
        with pytest.raises(Exception):
            obj.fit(df=df_cp)
