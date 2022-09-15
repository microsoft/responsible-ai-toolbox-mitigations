import numpy as np
import pandas as pd
import re


# -----------------------------------
def is_str_number(string: str):
    # check if string is in the format '42'
    if string.isdigit():
        return True
    # check if string is in the format '4.2'
    if re.match(r"^-?\d+(?:\.\d+)$", string) is not None:
        return True
    # check if string is in the format '.42'
    if re.match(r"(\.\d+)$", string) is not None:
        return True
    return False


# -----------------------------------
def get_cat_cols(df: pd.DataFrame, subset: list = None):
    """
    Returns a list of all categorical columns in the dataset
    df. If subset != None, check for categorical columns
    only in the columns within the subset list.

    :param df: the dataset being analyzed;
    :param subset: the list of columns that should be analyzed.
        If subset is None, then check all columns.
    :return: a list with the name of all categorical columns.
    :rtype: list
    """

    def test_if_categorical(value):
        if type(value) == str and not is_str_number(value):
            return 1
        return 0

    cat_col = []
    col_list = subset
    if subset is None:
        col_list = df.columns
    for i, col in enumerate(col_list):
        is_cat_arr = df[col].apply(test_if_categorical)
        if np.any(is_cat_arr):
            cat_col.append(col)
    return cat_col


# -----------------------------------
def ordinal_to_onehot(arr: list, n_class: int):
    """
    Converts a list of ordinal values that ranges
    from [0, n_class] to a one-hot matrix with
    shape (len(arr), n_class).

    :param arr: a list of labels
    :param n_class: the number of classes in arr
    :return: a list of lists (a matrix) of one-hot
        encodings, where each label in arr is one-hot
        encoded according to the maximum number of
        classes n_class.
    :rtype: list of lists
    """
    onehot_matrix = []
    for value in arr:
        onehot = [0 for _ in range(n_class)]
        onehot[int(value)] = 1
        onehot_matrix.append(onehot)
    return onehot_matrix


# -----------------------------------
def err_float_01(param, param_name):
    """
    Raises an error if param is not in the range [0, 1].
    param_name represents the name of the parameter 'param'.
    This makes it easier for the user to identify where the
    error occured.

    :param param: the numerical parameter being analyzed;
    :param param_name: the internal name used for the parameter
        provided in param.
    """
    if type(param) != float or param < 0.0 or param > 1.0:
        raise ValueError(
            f"ERROR: invalid value for parameter '{param_name}'. " + "Expected a float value between [0.0, 1.0]."
        )
