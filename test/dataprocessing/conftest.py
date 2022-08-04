import random
import pytest
import numpy as np
import torch

from raimitigations.dataprocessing import create_dummy_dataset


SEED = 42


# -----------------------------------
def _set_seed():
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)


# -----------------------------------
@pytest.fixture
def df_num():
    _set_seed()
    df = create_dummy_dataset(
        samples=500,
        n_features=6,
        n_num_num=0,
        n_cat_num=0,
        n_cat_cat=0,
    )
    return df


# -----------------------------------
@pytest.fixture
def label_col_name():
    return "label"


# -----------------------------------
@pytest.fixture
def df_full():
    _set_seed()
    df = create_dummy_dataset(
        samples=500,
        n_features=6,
        n_num_num=2,
        n_cat_num=2,
        n_cat_cat=0,
        num_num_noise=[0.01, 0.02],
        pct_change=[0.03, 0.05],
    )
    return df


# -----------------------------------
@pytest.fixture
def df_regression():
    _set_seed()
    df = create_dummy_dataset(
        samples=500,
        n_features=6,
        n_num_num=2,
        n_cat_num=2,
        n_cat_cat=0,
        num_num_noise=[0.01, 0.02],
        pct_change=[0.03, 0.05],
        regression=True,
    )
    return df


# -----------------------------------
@pytest.fixture
def df_full_nan():
    _set_seed()
    def add_nan(vec, pct):
        vec = list(vec)
        nan_index = random.sample(range(len(vec)), int(pct * len(vec)))
        for index in nan_index:
            vec[index] = np.nan
        return vec

    df = create_dummy_dataset(
        samples=500,
        n_features=6,
        n_num_num=2,
        n_cat_num=2,
        n_cat_cat=2,
        num_num_noise=[0.01, 0.05],
        pct_change=[0.05, 0.1],
    )
    col_with_nan = ["num_0", "num_3", "num_4", "CN_0_num_0", "CC_1_num_1"]
    for col in col_with_nan:
        if col != "label":
            df[col] = add_nan(df[col], 0.1)
    return df


# -----------------------------------
@pytest.fixture
def label_col_index():
    return 6


# -----------------------------------
def check_valid_columns(final_list, selected, include_label=True):
    if include_label:
        size = len(final_list) == len(selected) + 1
    else:
        size = len(final_list) == len(selected)
    for v in selected:
        if v not in final_list:
            return False
        break
    return size


# -----------------------------------
def check_fixed_col(fixed, selected):
    for value in fixed:
        if type(value) == int:
            value = str(value)
        if value not in selected:
            return False
    return True


# -----------------------------------
def check_valid_input(df, label_col, X, y):
    if df is not None or label_col is not None:
        if df is None or label_col is None:
            raise ValueError("ERROR: please provide a valid (df, label_col) tuple.")
    elif X is not None or y is not None:
        if X is None or y is None:
            raise ValueError("ERROR: please provide a valid (X, y) tuple.")


# -----------------------------------
def check_cols_num(df, cols):
    for col in cols:
        if df.dtypes[col] == "object":
            return False
    return True
