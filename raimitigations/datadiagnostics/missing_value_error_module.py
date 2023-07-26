from .error_module import ErrorModule


class MissingValueErrorModule(ErrorModule):
    """
    This module predicts missing values in the data.
    """

    # -----------------------------------
    def __init__(self):
        self.module_name = "MissingValueErrorModule"

    # -----------------------------------
    def _predict(self, vals: list) -> set:
        """
        Predicts and returns a set of the subset of a domain that is potentially
        erroneous.

        :param vals: a list of values to predict missing value errors on;

        :return: a set of predicted erroneous values;
        :rtype: a set.
        """
        erroneous_vals = set()
        for val in vals:
            if str(val).lower() in ["nan", "", " "]:
                erroneous_vals.add(str(val).lower())
        return erroneous_vals

    # -----------------------------------
    def _description(self) -> str:
        """
        Returns a description of the error.
        """
        return f"MissingValueError: column contains missing values."

    # -----------------------------------
    def _get_available_types(self) -> list:
        """
        Returns a list of data types available for prediction using this error prediction module.
        """
        return ["numerical", "categorical", "string"]
