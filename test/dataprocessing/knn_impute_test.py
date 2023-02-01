import numpy as np
import pandas as pd
import pytest
from raimitigations.dataprocessing import KNNDataImputer

from sklearn.experimental import enable_iterative_imputer  # noqa # pylint: disable=unused-import
from sklearn.impute import IterativeImputer, KNNImputer

COL_WITH_NAN = ["num_0", "num_3", "num_4", "CN_0_num_0", "CC_1_num_1"]
COL_WITH_NAN_IND = [0, 3, 4, 9, 12]


# -----------------------------------
def _get_object_list(df=None, use_index=True):

    imputer_list = []

    if use_index:
        col_impute1 = [0, 1, 2, 3, 4] # numerical only
        col_impute2 = [0, 3, 4, 9, 12] # numerical and categorical
        col_impute3 = [9, 12] # categorical only
        col_impute4 = [0, 1, 2, 3, 10]  # numerical and categorical with non-missing data
    else:
        col_impute1 = ["num_0", "num_1", "num_2", "num_3", "num_4"]
        col_impute2 = ["num_0", "num_3", "num_4", "CN_0_num_0", "CC_1_num_1"]
        col_impute3 = ["CN_0_num_0", "CC_1_num_1"]
        col_impute4 = ["num_0", "num_1", "num_2", "num_3", "CN_1_num_1"]
    knn_params = {
        "missing_values": np.nan,
        "n_neighbors": 4,
        "weights": "uniform",
        "metric": "nan_euclidean",
        "copy": True,
    }

    imputer = KNNDataImputer(df=df, enable_encoder=True)
    imputer_list.append(imputer)

    imputer = KNNDataImputer(df=df, col_impute=col_impute1, enable_encoder=False, knn_params=knn_params)
    imputer_list.append(imputer)

    imputer = KNNDataImputer(df=df, col_impute=col_impute1, enable_encoder=True, knn_params=knn_params)
    imputer_list.append(imputer)

    imputer = KNNDataImputer(df=df, col_impute=col_impute2, enable_encoder=True, knn_params=knn_params)
    imputer_list.append(imputer)

    imputer = KNNDataImputer(df=df, col_impute=col_impute3, enable_encoder=True, knn_params=knn_params)
    imputer_list.append(imputer)

    imputer = KNNDataImputer(df=df, col_impute=col_impute4, enable_encoder=True, knn_params=knn_params)
    imputer_list.append(imputer)

    imputer = KNNDataImputer(df=df, col_impute=col_impute1, enable_encoder=False, sklearn_obj=KNNImputer())
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

    with pytest.raises(Exception):
        KNNDataImputer(col_impute=["num_0"], sklearn_obj=IterativeImputer())

    with pytest.raises(ValueError):
        KNNDataImputer(col_impute=["num_0"], enable_encoder='not boolean')

    fit_obj_list = [
        KNNDataImputer(col_impute=["num_0", "num_1", "num_2", "num_3", "num_4"], knn_params=["list"]),
        KNNDataImputer(col_impute=["num0"]),
        KNNDataImputer(col_impute=[99]),
        KNNDataImputer(col_impute=["num_0", "num_1", "num_2", "num_3", "num_4"], knn_params={
            "missing_values": np.nan,
            "weights": "uniform",
            "metric": "nan_euclidean",
            "copy": True,
        }),
    ]

    for obj in fit_obj_list:
        df_cp = df_full_nan.copy()
        with pytest.raises(Exception):
            obj.fit(df=df_cp)

    df_cp = df_full_nan.copy()
    df_cp2 = df_cp.copy()
    df_cp2.at[2, "num_0"] = 'x'
    df_cp3 = df_cp.drop(columns=["num_0"])
    
    transf_obj_list = [
        (KNNDataImputer(df=df_cp, col_impute=["num_0", "num_3", "num_4", "CN_0_num_0", "CC_1_num_1"], enable_encoder=False), df_cp),
        (KNNDataImputer(df=df_cp, col_impute=["num_0", "num_3", "num_4", "CN_0_num_0", "CC_1_num_1"], enable_encoder=True), df_cp3),
        (KNNDataImputer(df=df_cp, col_impute=["num_0", "num_3", "num_4"], enable_encoder=False), df_cp2),
    ]

    for obj in transf_obj_list:
        obj[0].fit()
        with pytest.raises(Exception):
            obj[0].transform(df=obj[1])
