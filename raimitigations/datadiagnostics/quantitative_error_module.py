import numpy as np
from .error_module import ErrorModule
from .utils import checkfloattype

class QuantitativeErrorModule(ErrorModule):
    """
    This module predicts both quantitative parsing failures and abnormal values in a dataset using standard deviation.

    :param thresh: a standard deviation count threshold to determine how many stds can a non-erroneous value be beyond the values' mean. This parameter defaults at 3.5;
    """
    # -----------------------------------
    def __init__(self, thresh: float =3.5):
        self.thresh = thresh
        self.module_name = "QuantitativeErrorModule"

    # -----------------------------------
    def _predict(self, vals: list) -> set:
        """
        Predicts and returns a set of the subset of a domain that is potentially
        erroneous.

        :param vals: a list of values to predict quantitative errors on;

        :return: a set of predicted erroneous values;
        :rtype: a set.
        """
        vals = list(filter(lambda x: not np.isinf(x), [checkfloattype(v) for v in list(vals)]))

        std = np.std(vals)
        mean = np.mean(vals)

        vals_set = set(vals)  # get unique values

        erroneous_vals = set()
        for val in vals_set:
            if np.abs(val - mean) > self.thresh * std:
                erroneous_vals.add(val)

        return erroneous_vals

    # -----------------------------------
    def _description(self) -> str:
        """
        Returns a description of the error.
        """
        return f"QuantitativeError: value was found to be greater than > {str(self.thresh)} stds beyond the mean."

    # -----------------------------------
    def _get_available_types(self) -> list:
        """
        Returns a list of data types available for prediction using this error prediction module.
        """
        return ["numerical"]
