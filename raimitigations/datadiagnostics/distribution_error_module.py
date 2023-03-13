import numpy as np
from .error_module import ErrorModule

class DistributionErrorModule(ErrorModule):
    """
    This module predicts values that appear more or less frequently than typical in the dataset using standard deviation.

    :param thresh: a standard deviation count threshold to determine how many stds can the distribution count of a non-erroneous value be beyond the distribution mean. This parameter defaults at 3.5;

    :param fail_thresh: minimum number of unique values for a successful run of the distribution error prediction module. This parameter defaults at 2;
    """

    # -----------------------------------
    def __init__(self, thresh=3.5, fail_thresh=2):
        self.thresh = thresh
        self.fail_thresh = fail_thresh
        self.module_name = "DistributionErrorModule"

    # -----------------------------------
    def _predict(self, vals: list) -> set:
        """
        Predicts and returns a set of the subset of a domain that is potentially
        erroneous.

        :param vals: a list of values to predict distribution errors on;

        :return: a set of predicted erroneous values;
        :rtype: a set.
        """
        dist = {}
        vals = [x for x in vals if str(x) != "nan"]
        vals_set = set(vals)
        for v in vals:
            if v not in dist:
                dist[v] = 0 
            dist[v] = dist[v] + 1

        val_scores = [dist[v] for v in vals_set]

        dist_std = np.std(val_scores)
        dist_mean = np.mean(val_scores)

        erroneous_vals = set()

        # fail if number of unique values is less than the fail threshold, or if dist_std is not informative
        if len(val_scores) <= self.fail_thresh or dist_std < 1.0:
            return erroneous_vals

        for val in vals_set:
            if np.abs(dist[val] - dist_mean) > self.thresh * dist_std:
                erroneous_vals.add(val)

        return erroneous_vals

    # -----------------------------------
    def _description(self) -> str:
        """
        Returns a description of the error.
        """
        return f"DistributionError: values was found to have a distribution count of greater than > {str(self.thresh)} stds beyond the mean distribution count."

    # -----------------------------------
    def _get_available_types(self) -> list:
        """
        Returns a list of data types available for prediction using this error prediction module.
        """
        return ["numerical", "categorical", "string"]
