from typing import Union
import pandas as pd
import numpy as np

from .data_diagnostics import DataDiagnostics
from .error_module import ErrorModule
from .quantitative_error_module import QuantitativeErrorModule
from .punctuation_error_module import PuncErrorModule
from .semantic_error_module import SemanticErrorModule
from .distribution_error_module import DistributionErrorModule
from .string_similarity_error_module import StringSimilarityErrorModule
from .char_similarity_error_module import CharSimilarityErrorModule


class ActiveDetect(DataDiagnostics):
    """
    Concrete class that implements multiple error modules of error prediction in data.
    Available error_modules:
        - QuantitativeErrorModule
        - PuncErrorModule
        - SemanticErrorModule
        - DistributionErrorModule
        - StringSimilarityErrorModule
        - CharSimilarityErrorModule

    This subclass uses error prediction modules presented in the Sanjay Krishnan et al.'s activedetect repo and paper: [BoostClean: Automated Error Detection and Repair for Machine Learning](https://arxiv.org/abs/1711.01299).

    :param df: pandas dataframe or np.ndarray to predict errors in;

    :param col_predict: a list of column names or indexes that will be subject to error prediction. If None, a list of all columns will be used by default;

    :param mode: a string that can take the values:
        - "column", prediction will be applied to each column.
            An error matrix of the same shape as the data will be returned by predict.
        - "row", prediction will be applied over each row as a whole. A list of
            erroneous row indices will be returned by predict.

    :param error_modules: a list of error module objects to be used for prediction, chosen from the above list. If None, all available modules are used by default.

    :param save_json: a string pointing to a path to save a json log file to when
        calling predict. It defaults to None, in that case, no log file is saved.

    :param verbose: boolean flag indicating whether internal messages should be printed or not.
    """

    # -----------------------------------
    def __init__(
        self,
        df: Union[pd.DataFrame, np.ndarray] = None,
        col_predict: list = None,
        mode: str = "column",
        error_modules: list = [],
        save_json: str = None,
        verbose: bool = True,
    ):
        super().__init__(df, col_predict, None, mode, save_json, verbose)
        self._set_error_modules(error_modules)
        self.module_error_matrix_dict: dict = {}

    # -----------------------------------
    def _check_error_modules(self, error_modules: list):
        """
        Verifies that each object in the input list is an ErrorModule object
        and raises an error otherwise.

        :param error_modules: a list of values to be verified as ErrorModule objects.
        """
        for module in error_modules:
            if not isinstance(module, ErrorModule):
                raise ValueError(
                    f"ERROR: {module} is not an ErrorModule object. Please only use the following options: "
                    + "QuantitativeErrorModule(), PuncErrorModule(), SemanticErrorModule(), "
                    + "DistributionErrorModule(), StringSimilarityErrorModule() or CharSimilarityErrorModule(). "
                    + "You can use these objects with default attributes or customize them as you like."
                )

    # -----------------------------------
    def _set_error_modules(self, error_modules: list):
        """
        Sets up the error_modules attribute. It verifies that each value in the user's
        input list is an ErrorModule object. If no list was passed by the user,
        a default list of all available modules is used.

        :param error_modules: a list of ErrorModule objects.
        """
        if not error_modules:
            default_modules = [
                QuantitativeErrorModule(),
                PuncErrorModule(),
                SemanticErrorModule(),
                DistributionErrorModule(),
                StringSimilarityErrorModule(),
                CharSimilarityErrorModule(),
            ]
            self.error_modules = default_modules
        else:
            self._check_error_modules(error_modules)
            self.error_modules = error_modules

    # -----------------------------------
    def _predictModule(self, df: pd.DataFrame, error_module: ErrorModule) -> np.array:
        """
        Loops over columns of the data, checks if the current error module
        applies to the column's data type and predicts erroneous rows over it.

        :param df: pd.DataFrame containing column data to predict errors on.
        :param error_module: ErrorModule object to use for error prediction on column values;

        :return: an error indicator matrix for the full dataset using the predictions of the current
            error module. Each value's prediction is indicated as follows:
            - -1 indicates an error;
            - +1 indicates no error was predicted or that this error module is not applicable for the column's data type;
            - np.nan for columns not in col_predict.

        :rtype: a 2-dimensional np.array.
        """
        error_matrix = []
        for col in df:
            if col in self.col_predict:
                col_type = self.types[self.col_predict.index(col)]
                indicator_vector = np.full(self.n_rows, 1)
                if col_type in error_module.get_available_types():
                    col_vals = df[col].values
                    erroneous_indices = error_module.get_erroneous_rows_in_col(col_vals)
                    indicator_vector[erroneous_indices] = -1
            else:
                indicator_vector = np.full(self.n_rows, np.nan)
            error_matrix.append(indicator_vector)

        return np.array(error_matrix).T

    # -----------------------------------
    def _get_final_error_matrix(self) -> np.array:
        """
        Combines the error matrices returned by all error modules into a single
        union error matrix. For a given element in the set, if any error module
        predicted that it was an error, then it's set as an error(-1) in the
        final matrix, if no module predicted an error, then it's set as a
        non-error(+1) and if a column was not included in col_predict, all
        values of that column map to np.nan in the final error matrix.

        :return: the final error matrix mapping each value in the set to its error
            indicator value.
        :rtype: a 2-dimensional np.array.
        """
        final_error_matrix = np.full((self.n_rows, self.n_cols), 1.0)
        for matrix in list(self.module_error_matrix_dict.values()):
            final_error_matrix = np.where(matrix == 1.0, final_error_matrix, -1.0)
        for i, col in enumerate(self.df_info.columns):
            if col not in self.col_predict:
                final_error_matrix[:, i] = np.nan
        return final_error_matrix

    # -----------------------------------
    def _get_erroneous_row_indices(self, final_error_matrix: np.array) -> list:
        """
        Get a list of erroneous row indices from the final error
        matrix. If any column in a row contains an error, it's
        considered erroneous.

        :param final_error_matrix: the final error matrix;

        :return: a list of erroneous row indices;
        :rtype: list.
        """
        bool_matrix = final_error_matrix == -1
        error_count_per_row = np.sum(bool_matrix, axis=1)
        erroneous_row_indices = np.where(error_count_per_row > 0)[0]

        return list(erroneous_row_indices)

    # -----------------------------------

    def _fit(self):
        """
        Fit method for this DataDiagnostics class.
        """
        self.n_rows = self.df_info.df.shape[0]
        self.n_cols = self.df_info.df.shape[1]
        return

    # -----------------------------------
    def _predict(self, df: pd.DataFrame) -> Union[np.array, list]:
        """
        Predict method of this class. It loops over error modules to predict
        errors over all applicable columns in the input dataset. It then calculates
        and returns the final union error matrix.

        :param df: a pandas dataframe to predict errors over.

        :return: the final error matrix.
        :rtype: a 2-dimensional np.array.
        """
        self.valid_cols = self.df_info.columns
        self._check_predict_data_structure(df)  # TODO: test this!

        for module in self.error_modules:
            self.module_error_matrix_dict[module.module_name] = self._predictModule(df, module)

        final_error_matrix = self._get_final_error_matrix()
        if self.mode == "column":
            return final_error_matrix
        else:
            indices = self._get_erroneous_row_indices(final_error_matrix)
            return df.index[indices].tolist()

    # ------------------------------------
    def get_error_module_matrix(self, error_module: str) -> np.array:
        """
        Returns the individual error matrix of a certain ErrorModule object post prediction.

        :param error_module: string name of an error module used during prediction.

        :return: an error matrix indicating errors predicted by the input error module.
        :rtype: a 2-dimensional np.array
        """
        self._check_if_predicted()
        module_keys = list(self.module_error_matrix_dict.keys())
        if error_module not in module_keys:
            raise KeyError(
                f"{error_module} is not part of the this error prediction object. Please use the name of an error module present in this detector, you have the following options: {module_keys}"
            )

        return self.module_error_matrix_dict[error_module]

    # -----------------------------------
    def _serialize(self) -> dict:
        """
        Serializes class attributes into a dictionary for logging.
        """
        return {
            "name": "ActiveDetect",
            "col_predict": self.col_predict,
            "mode": self.mode,
            "error_modules": [str(e_mod.module_name) for e_mod in self.error_modules],
            "save_json": self.save_json,
            "verbose": self.verbose,
        }
