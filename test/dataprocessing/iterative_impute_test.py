import numpy as np
import pandas as pd
import pytest
from raimitigations.dataprocessing import IterativeDataImputer
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer  # noqa # pylint: disable=unused-import
from sklearn.impute import IterativeImputer, KNNImputer

COL_WITH_NAN = ["num_0", "num_3", "num_4", "CN_0_num_0", "CC_1_num_1"]
COL_WITH_NAN_IND = [0, 3, 4, 9, 12]


# -----------------------------------
def _get_object_list(df=None, use_index=True):

    imputer_list = []

    if use_index:
        col_impute1 = [0, 1, 2] # numerical only
        col_impute2 = [0, 9] # numerical and categorical
        col_impute3 = [12] # categorical only
        col_impute4 = [1, 2, 3, 10]  # numerical and categorical with non-missing data
    else:
        col_impute1 = ["num_0", "num_1", "num_2"]
        col_impute2 = ["num_0", "CN_0_num_0"]
        col_impute3 = ["CC_1_num_1"]
        col_impute4 = ["num_1", "num_2", "num_3", "CN_1_num_1"]
    iterative_params = {
        'estimator': LinearRegression(),
        'missing_values': np.nan,
        'sample_posterior': False,
        'max_iter': 1,
        'tol': 1e-3,
        'n_nearest_features': None,
        'initial_strategy': 'mean',
        'imputation_order': 'ascending',
        'skip_complete': False,
        'min_value': -np.inf,
        'max_value': np.inf,
        'random_state': 100}

    imputer = IterativeDataImputer(df=df, enable_encoder=True)
    imputer_list.append(imputer)

    imputer = IterativeDataImputer(df=df, col_impute=col_impute1, enable_encoder=False, iterative_params=iterative_params)
    imputer_list.append(imputer)

    imputer = IterativeDataImputer(df=df, col_impute=col_impute1, enable_encoder=True, iterative_params=iterative_params)
    imputer_list.append(imputer)

    imputer = IterativeDataImputer(df=df, col_impute=col_impute2, enable_encoder=True, iterative_params=iterative_params)
    imputer_list.append(imputer)

    imputer = IterativeDataImputer(df=df, col_impute=col_impute3, enable_encoder=True, iterative_params=iterative_params)
    imputer_list.append(imputer)

    imputer = IterativeDataImputer(df=df, col_impute=col_impute4, enable_encoder=True, iterative_params=iterative_params)
    imputer_list.append(imputer)

    imputer = IterativeDataImputer(df=df, col_impute=col_impute1, enable_encoder=False, sklearn_obj=IterativeImputer(estimator=LinearRegression(), max_iter=1))
    imputer_list.append(imputer)

    imputer = IterativeDataImputer(df=df, col_impute=col_impute2, enable_encoder=True, iterative_params=iterative_params)
    imputer_list.append(imputer)

    return imputer_list


# -----------------------------------
def _run_main_commands(df, obj, df_in_fit=True):
    df = df.copy()
    if df_in_fit:
        obj.fit(df=df)
    else:
        obj.fit()
    new_df = obj.transform(df)

    nan_dict = new_df.isna().any()
    nan_dict = {key: nan_dict[key] for key in nan_dict.keys() if key in obj.col_impute}
    assert not all(nan_dict.values()), (
        "ERROR: Not all missing values were removed. Something went wrong.\n"
        f"The columns with nan are: {nan_dict}\n"
        f"The columns that should've been imputed are: {obj.col_impute}"
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
    df_edited = df.copy()
    df_edited['CN_0_num_0'] = df['CN_0_num_0'].replace({"val0_0": "val0_2", "val0_1": "val0_4"})
    for obj in obj_list:
        if obj == obj_list[-1]:
            obj.df = df
            _run_main_commands(df_edited, obj, df_in_fit=False)
        else:
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
        IterativeDataImputer(col_impute=["num_0"], sklearn_obj=KNNImputer())
    with pytest.raises(Exception):
        IterativeDataImputer(col_impute=["num_0"], enable_encoder='not boolean')

    fit_obj_list = [
        IterativeDataImputer(col_impute=["num_0", "num_1", "num_2", "num_3", "num_4"], iterative_params=["list"]),
        IterativeDataImputer(col_impute=["num0"]),
        IterativeDataImputer(col_impute=[99]),
        IterativeDataImputer(col_impute=["num_0", "num_1", "num_2", "num_3", "num_4"], iterative_params={
            'estimator': LinearRegression(),
            'missing_values': np.nan,
            'sample_posterior': False,
            'max_iter': 1,
            'tol': 1e-3,
            'n_nearest_features': None,
            'initial_strategy': 'mean',
            'min_value': -np.inf,
            'max_value': np.inf,
            'random_state': None}),
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
        (IterativeDataImputer(df=df_cp, col_impute=["num_0", "num_3", "num_4", "CN_0_num_0", "CC_1_num_1"], enable_encoder=False), df_cp),
        (IterativeDataImputer(df=df_cp, col_impute=["num_0", "num_3", "num_4", "CN_0_num_0", "CC_1_num_1"], enable_encoder=True), df_cp3),
        (IterativeDataImputer(df=df_cp, col_impute=["num_0", "num_3", "num_4"], enable_encoder=False), df_cp2),
    ]

    for obj in transf_obj_list:
        obj[0].fit()
        with pytest.raises(Exception):
            obj[0].transform(df=obj[1])
