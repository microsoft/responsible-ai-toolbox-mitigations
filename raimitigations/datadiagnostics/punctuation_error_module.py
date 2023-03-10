import numpy as np
import re
from .error_module import ErrorModule


class PuncErrorModule(ErrorModule):

    """
    This module detects attributes that are only punctuation, whitespace, etc.
    """

    # -----------------------------------

    def __init__(self):
        self.module_name = "PuncErrorModule"
        pass

    # -----------------------------------
    def _predict(self, strings):
        """
        Predicts and returns a set of the subset of a domain that is potentially
        erroneous.

        :param strings: a list of string values to predict punctuation errors on;
        """
        erroneous_vals = set()
        strings = [x for x in strings if str(x) != "nan"]
        vals_set = set(strings)  # get unique values

        for s in vals_set:
            sstrip = re.sub(r"\W+", "", str(s).lower())
            cleaned_string = sstrip.lower().strip()
            if len(cleaned_string) == 0:
                erroneous_vals.add(s)

        return erroneous_vals

    # -----------------------------------
    def get_erroneous_rows_in_col(self, col_vals):
        """
        Given the error set found by predict, this method maps the errors to particular rows
        in the column, returning a list of erroneous row indices.

        :param col_vals: a list of string values to predict punctuation errors on;

        :return:
        :rtype:
        """
        erroneous_vals = self._predict(col_vals)
        erroneous_indices = []
        for e_val in erroneous_vals:
            erroneous_indices.extend(list(np.where(col_vals == e_val)[0]))

        return erroneous_indices

    # -----------------------------------
    def description(self):
        """
        Returns a description of the error.
        """
        return "PunctuationError: An attribute was found with no alpha numeric characeters."

    # -----------------------------------
    def get_available_types(self):
        """
        Returns a list of data types available for prediction using this error detection module.
        """
        return ["categorical", "string"]
