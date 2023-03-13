import numpy as np

# -----------------------------------
def checkfloattype(val: any) -> float:
    """
    Checks if a given value is float-like.

    :param val: the value to test;

    :return: the value itself if it is float-like or np.inf otherwise;
    :rtype: float or np.inf.
    """
    try:
        return float(val)
    except ValueError:
        return np.inf

# -----------------------------------
def is_num(col_vals: list, num_thresh: float) -> bool:
    """
    Determines whether a list of data is numerical.

    :param num_thresh: a float threshold of the minimum ratio of float-like values of a numerical column;
    
    :return: a boolean flag determining if a column is numerical;
    :rtype: bool.
    """
    float_count = 0.0
    for val in col_vals:
        try:
            float(str(val).strip())
            float_count = float_count + 1.0
        except ValueError:
            pass

    return float_count / len(col_vals) > num_thresh

# -----------------------------------
def is_cat(col_vals: list, cat_thresh: float) -> bool:
    """
    Determines whether data is categorical.

    :param cat_thresh: a float threshold of the maximum ratio of unique string data of a categorical column;

    :return: a boolean flag determining if a column is categorical;
    :rtype: bool.
    """
    max_unique = cat_thresh * len(col_vals)
    counts = {}
    for val in col_vals:
        if val not in counts:
            counts[val] = 0
        counts[val] = counts[val] + 1

    total_unique = len([k for k in counts if counts[k] <= 1]) + 0.0  # total values with more than 1 frequency
    return total_unique < max_unique  # check if less than the maximum threshold for unique values.
