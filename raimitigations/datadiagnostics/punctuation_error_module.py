import numpy as np
import re
from .error_module import ErrorModule

class PuncErrorModule(ErrorModule):
    """
    This module predicts attributes that are only punctuation, whitespace, etc.
    """
    # -----------------------------------
    def __init__(self):
        self.module_name = "PuncErrorModule"

    # -----------------------------------
    def _predict(self, strings: list) -> set:
        """
        Predicts and returns a set of the subset of a domain that is potentially
        erroneous.

        :param strings: a list of string values to predict punctuation errors on;

        :return: a set of predicted erroneous values;
        :rtype: a set.
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
    def _description(self) -> str:
        """
        Returns a description of the error.
        """
        return "PunctuationError: An attribute was found with no alpha numeric characters."

    # -----------------------------------
    def _get_available_types(self) -> list:
        """
        Returns a list of data types available for prediction using this error prediction module.
        """
        return ["categorical", "string"]
