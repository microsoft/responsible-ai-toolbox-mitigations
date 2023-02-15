import numpy as np
from .error_module import ErrorModule

class DistributionErrorModule(ErrorModule):
    """
    This module detects values that appear more or less frequently than typical in the dataset using standard deviation.

    :param thresh: a standard deviation count threshold to determine how many stds can the distribution count of a non-erroneous value be beyond the distribution mean. This parameter defaults at 3.5;

    :param fail_thresh: minimum number of unique values for a successful run of the distribution error detection module. This parameter defaults at 2;
    """

    # -----------------------------------
    def __init__(self, thresh=3.5, fail_thresh=2):
        self.thresh = thresh
        self.fail_thresh = fail_thresh

    # -----------------------------------
    def predict(self, vals: list) -> set:
        """
        Predicts and returns a set of the subset of a domain that is potentially
        erroneous.

        :param vals: a list of values to predict distribution errors on;
        """
        dist = {}
        for v in vals:
            if v not in dist:
                dist[v] = 0 #initialize the distribution of this value at 0 if not seen before

            dist[v] = dist[v] + 1 # increment by 1 (a counter for each value's frequency/distribution in the data)

        vals_set = set(vals) # set of values, removes duplicates
        dist_vals = [dist[v] for v in vals_set] #list of distribution values

        dist_std = np.std(dist_vals) # std of all value distributions
        dist_mean = np.mean(dist_vals)  # mean of all value distributions

        erroneous_vals = set()

        #fail if number of unique values is less than the fail threshold, or if
        #dist_std not informative
        if len(dist_vals) <= self.fail_thresh or dist_std < 1.0:  # if the std of dist is <1 #TODO: can we do len(valsv) instead
            return erroneous_vals

        for val in vals_set:
            if np.abs(dist[val] - dist_mean) > self.thresh * dist_std:
                erroneous_vals.add(val)

        return erroneous_vals

    # -----------------------------------
    def get_erroneous_rows_in_col(self, erroneous_vals, dataset, col):
        """
        Given the error set found by predict, this method maps the errors to particular rows
        in the column, returning a list of erroneous row indices.

        :param erroneous_vals: a set of errors predicted by the predict function;

        :param dataset: dataset containing the column of data evaluated for errors;

        :param col: name or index of column that has been evaluated for errors;
        """

        col_vals = np.array(dataset[col]) #TODO: doublecheck this returns column not row values even with 2d lists which data[row][col], not[col][row]
        erroneous_indices = [] 
        for e_val in erroneous_vals:
            erroneous_indices.extend(list(np.where(col_vals == e_val)[0]))

        """
        erroneous_rows = []
        indices = []

        for row_idx, row in enumerate(dataset):
            val = row[col] #loop one val at a time in a column
            match = False
            for e_val in erroneous_vals:
                if e_val == val:
                    match = True #TODO: This should end the loop, are there duplicates in erroneous_vals? No.
            if match:
                erroneous_rows.append(row)
                indices.append(row_idx)
                
        return indices
        """
        return erroneous_indices

    # -----------------------------------
    def description(self):
        """
        Returns a description of the error.
        """
        return f"DistributionError: values was found to have a distribution count of greater than > {str(self.thresh)} stds beyond the mean distribution count."

    # -----------------------------------
    def get_available_types(self):
        """
        Returns a list of data types available for prediction using this error detection module.
        """
        return ['numerical', 'categorical', 'string']
