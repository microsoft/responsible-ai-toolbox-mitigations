import numpy as np
from .error_module import ErrorModule

class QuantitativeErrorModule(ErrorModule):
    """
    This module detects both quantitative parsing failures and abnormal values in a dataset using standard deviation.

    :param thresh: a standard deviation count threshold to determine how many stds can a non-erroneous value be beyond the values' mean. This parameter defaults at 3.5;
    """

    # -----------------------------------
    def __init__(self, thresh=3.5):
        self.thresh = thresh

    # -----------------------------------
    def _predict(self, vals):
        """
        Predicts and returns a set of the subset of a domain that is potentially
        erroneous.
        :param vals: a list of values to predict quantitative errors on;
        """
        vals = filter(lambda x : not np.isinf(x) , [self.checktype(v) for v in list(vals)])  # TODO: does this simply filter non floats (checktype()) and np.inf?

        std = np.std(vals)
        mean = np.mean(vals)

        vals_set = set(vals) # get unique values

        erroneous_vals = set()
        for val in vals_set:
            if np.abs(val - mean) > self.thresh * std:
                erroneous_vals.add(val)

        return erroneous_vals

    # -----------------------------------
    def checktype(self, num):
        try:
            return float(num)
        except ValueError:
            return np.inf

    # -----------------------------------
    def get_erroneous_rows_in_col(self, col_vals):
        """
        Given the error set found by predict, this method maps the errors to particular rows
        in the column, returning a list of erroneous row indices.

        :param col_vals: a list of values to predict quantitative errors on;
        :return:
        :rtype:
        """
        erroneous_vals = self._predict(col_vals)
        erroneous_indices = []
        for e_val in erroneous_vals:
            erroneous_indices.extend(list(np.where(col_vals == e_val)[0]))

        '''
        erroneous_indices = []
        eset = set(erroneous_vals)

        for i, d in enumerate(dataset):
            val = self.checktype(d[col])

            if val in eset or np.isinf(val): #TODO: This seems wrong, why would we add non-floats or inf to the error indicies but predict doesn't add them to the error list, just ignore them, these aren't quantitative errors.
                erroneous_indices.append(i)
        '''
        return erroneous_indices

    # -----------------------------------
    def description(self):
        """
        Returns a description of the error.
        """
        return f"QuantitativeError: value was found to be greater than > {str(self.thresh)} stds beyond the mean."

    # -----------------------------------
    def get_available_types(self):
        """
        Returns a list of data types available for prediction using this error detection module.
        """
        return ['numerical']
