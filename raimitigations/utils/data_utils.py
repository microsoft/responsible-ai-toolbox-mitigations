import numpy as np
import pandas as pd
from scipy import stats
import re
from typing import Union, Tuple

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
    if not isinstance(param, (np.floating, float)) or param < 0.0 or param > 1.0:
        raise ValueError(
            f"ERROR: invalid value for parameter '{param_name}'. " + "Expected a float value between [0.0, 1.0]."
        )


# -----------------------------------
def _transform_ordinal_encoder_with_new_values(
    encoder: object, df: Union[pd.DataFrame, np.ndarray]
) -> Tuple[Union[pd.DataFrame, np.ndarray], dict]:
    """
    Encodes categorical data using an existing EncoderOrdinal mapping
    as well as a complimentary mapping for new values. It creates a new mapping for
    values unseen at fit time on top of the existing mapping.

    :param encoder: an EncoderOrdinal object to use for the base mapping;
    :param df: pandas dataframe containing new unmapped values to encode;

    :return: the encoded dataframe, the corresponding inverse-encoding map.
    :rtype: pd.DataFrame or np.ndarray, dictionary.
    """
    df_cpy = df.copy()
    df_cpy.columns = df.columns.astype(str)
    encoder_mapping = encoder.get_mapping()
    new_mapping = encoder_mapping.copy()
    inverse_map_dicts = {}

    for col in encoder.col_encode:
        max_encoding = max(encoder_mapping[col]["labels"])
        for val in (df_cpy[col]).unique():
            if val not in encoder_mapping[col]["values"] and not pd.isna(val):
                new_mapping[col]["values"].append(val)
                new_mapping[col]["labels"].append(max_encoding + 1)
                max_encoding += 1

        map_simple = dict(zip(new_mapping[col]["values"], new_mapping[col]["labels"]))
        df_cpy[col] = df_cpy[col].map(map_simple, na_action=None)

        inverse_map_dicts[col] = dict(zip(new_mapping[col]["labels"], new_mapping[col]["values"]))

    return df_cpy, inverse_map_dicts


# -------------------------------------
def _inverse_transform_ordinal_encoder_with_new_values(
    inverse_mapping: dict,
    df: Union[pd.DataFrame, np.ndarray],
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Inverse-transforms encoding created by __ordinal_encoder_with_new_values().

    :param inverse_mapping: a dictionary mapping each column to a dictionary
        that maps encoding labels to the corresponding values of that column;
    :param df: pandas dataframe to inverse-transform its encoding;

    :return: the inverse transformed dataframe.
    :rtype: pd.DataFrame or np.ndarray.
    """
    df_cpy = df.copy()
    for col in inverse_mapping.keys():
        df_cpy[col] = df[col].astype("Float64").astype("Int64").map(inverse_mapping[col])
        if df_cpy[col].isnull().values.any():
            # if inverse encode failed for some values in the column, don't inverse encode column
            df_cpy[col] = df[col]
            print(
                f"WARNING: Encoding was not reverse transformed for column: {col}."
                + " Note that encoded columns are not guaranteed to reverse transform if they have imputed values."
            )

    return df_cpy


# -------------------------------------
def freedman_diaconis(data: pd.Series):
    """
    Computes the optimal number of bins for a set of data using the
    Freedman Diaconis rule.

    :param data: the data column used to compute the number of bins.
    """
    data = np.asarray(data, dtype=np.float_)
    iqr = stats.iqr(data, rng=(25, 75), scale=1.0, nan_policy="omit")
    N = data.size
    bw = (2 * iqr) / np.power(N, 1 / 3)

    min_val, max_val = data.min(), data.max()
    datrng = max_val - min_val
    result = int((datrng / bw) + 1)
    return result
