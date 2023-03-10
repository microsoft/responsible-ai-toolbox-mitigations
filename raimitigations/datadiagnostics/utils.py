import numpy as np

# -----------------------------------
def checkfloattype(num):
    try:
        return float(num)
    except ValueError:
        return np.inf


# -----------------------------------
def calculate_mad(arr):
    """Median Absolute Deviation: a "Robust" version of standard deviation.
    Indices variability of the sample.
    https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    arr = np.ma.array(arr).compressed()  # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def is_num(col_vals, num_thresh):
    """
    Internal method to determine whether data is numerical
    """
    float_count = 0.0
    for val in col_vals:
        try:
            float(str(val).strip())
            float_count = float_count + 1.0
        except ValueError:
            pass

    return float_count / len(col_vals) > num_thresh


def is_cat(col_vals, cat_thresh):
    """
    Internal method to determine whether data is categorical
    defaults to number of distinct values is N/LogN
    """
    max_unique = cat_thresh * len(col_vals)
    counts = {}
    for val in col_vals:
        if val not in counts:
            counts[val] = 0
        counts[val] = counts[val] + 1

    total_unique = len([k for k in counts if counts[k] <= 1]) + 0.0  # total values with more than 1 frequency
    return total_unique < max_unique  # check if less than the maximum threshold for unique values.
