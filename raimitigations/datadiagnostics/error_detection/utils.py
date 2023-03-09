import numpy as np
#import usaddress #TODO: Install

#TODO: fix doc strings for all these functions

# -----------------------------------
def checkfloattype(num):
    try:
        return float(num)
    except ValueError:
        return np.inf
# -----------------------------------
def _mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
    Indices variability of the sample.
    https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    arr = np.ma.array(arr).compressed()  # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def _is_addr(col_vals, addr_thresh):
    """
    Internal method to determine whether the column is an address
    """
    '''
    addr_count = 0.0
    for val in col_vals:
        try:
            addr_vals = usaddress.parse(val)
            addr_strings = [v for v in addr_vals if v[1] == 'Recipient']
            if len(addr_strings) != len(addr_vals):
                addr_count = addr_count + 1.0
        except:
            pass

    return (addr_count / len(col_vals) > addr_thresh)
    '''
    return False

def _is_num(col_vals, num_thresh):
    """
    Internal method to determine whether data is numerical
    """
    float_count = 0.0
    for val in col_vals:  #TODO: fyi before: datum here is a row and data is a list of rows.
        try:
            float(str(val).strip())
            float_count = float_count + 1.0
        except ValueError:
            pass

    return (float_count/len(col_vals) > num_thresh) #TODO: fyi: before: we divide by data because its the total # of rows, and so length of column.

def _is_cat(col_vals, cat_thresh):
    """
    Internal method to determine whether data is categorical
    defaults to number of distinct values is N/LogN
    """
    counts = {}
    for val in col_vals:
        if val not in counts:
            counts[val] = 0
        counts[val] = counts[val] + 1

    total = len([k for k in counts if counts[k] > 1])+0.0 #total values with more than 1 frequency
    return (total < cat_thresh)
