from abc import abstractmethod
import numpy as np


class ErrorModule:
    """
    Base class for error modules used by ActiveDetect.
    This error prediction applies to the distinct values of a column.
    """

    # -----------------------------------
    @abstractmethod
    def _predict(self, vals: list) -> set:
        """
        Abstract method. For a given concrete class, it predicts and
        returns a list of the subset of a domain that is potentially erroneous.

        :param vals: a list of values to predict errors on;

        :return: a set of predicted erroneous values;
        :rtype: a set.
        """
        raise NotImplementedError("An error module must implement a _predict() method.")

    # -----------------------------------
    def get_erroneous_rows_in_col(self, col_vals: list) -> list:
        """
        Predicts erroneous values in the input data, maps the errors
        to particular rows in the column and returns a list of
        erroneous row indices.

        :param col_vals: a list of values to predict errors on;

        :return: a list of row indices mapping to predicted errors in the column.
        :rtype: a list.
        """
        erroneous_vals = self._predict(col_vals)
        erroneous_indices = []
        for e_val in erroneous_vals:
            if str(e_val).lower() == "nan":
                erroneous_indices.extend(list(np.where(np.array([str(i).lower() for i in col_vals]) == "nan")[0]))
            else:
                erroneous_indices.extend(list(np.where(col_vals == e_val)[0]))

        return erroneous_indices

    # -----------------------------------
    @abstractmethod
    def _description(self) -> str:
        """
        Abstract method. For a given concrete class, returns
        a description of the error module.
        """
        raise NotImplementedError("An error module must implement a _description() method.")

    # -----------------------------------
    def description(self) -> str:
        """
        Returns a description of the error module.
        """
        return self._description()

    # -----------------------------------
    @abstractmethod
    def _get_available_types(self) -> list:
        """
        Abstract method. For a given concrete class, returns a list
        of data types supported by the error prediction module.
        """
        raise NotImplementedError("An error module must implement a _get_available_types() method.")

    # -----------------------------------
    def get_available_types(self) -> list:
        """
        Returns a list of data types supported by the error prediction module.
        """
        return self._get_available_types()
