from typing import Union
import pandas as pd
import numpy as np

from .error_detector import ErrorDetection
from .error_module import ErrorModule
from .quantitative_error_module import QuantitativeErrorModule
from .punctuation_error_module import PuncErrorModule
from .semantic_error_module import SemanticErrorModule
from .distribution_error_module import DistributionErrorModule
from .string_similarity_error_module import StringSimilarityErrorModule
from .char_similarity_error_module import CharSimilarityErrorModule

class ActiveDetect(ErrorDetection):
    """
    Concrete class that implements multiple error modules of error detection in data.
    Available error_modules:
        - QuantitativeErrorModule
        - PuncErrorModule
        - SemanticErrorModule
        - DistributionErrorModule
        - StringSimilarityErrorModule
        - CharSimilarityErrorModule

    This subclass uses error detection modules presented in the Sanjay Krishnan et al.'s activedetect repo and paper: [BoostClean: Automated Error Detection and Repair for Machine Learning](https://arxiv.org/abs/1711.01299). 

    :param df: pandas dataframe to detect errors in;

    :param col_predict: a list of column names or indexes that will be subject to error detection. If None, a list of all feature columns will be used by default;

    :param error_modules: a list of error detection module objects to be used from the above list of available error modules. If None, all available modules are used by default. 

    :param verbose: boolean flag indicating whether internal messages should be printed or not. #TODO: when could this be used??
    """

    # -----------------------------------
    def __init__(
        self,
        df: Union[pd.DataFrame, np.ndarray] = None,
        col_predict: list = None, 
        error_modules: list = [],
        verbose: bool = True,
    ):
        super().__init__(df, col_predict, verbose)    
        self._set_error_modules(error_modules)
        self.module_error_matrix_dict: dict = {}
    
    # -----------------------------------
    def _check_error_modules(self, error_modules: list):
        for module in error_modules:
            if not isinstance(module, ErrorModule):
                raise ValueError(f"ERROR: {module} is not an ErrorModule object. Please use only the following options: " +
                                 "QuantitativeErrorModule(), PuncErrorModule(), SemanticErrorModule(), " +
                                 "DistributionErrorModule(), StringSimilarityErrorModule() or CharSimilarityErrorModule(). " +
                                 "You can use these objects with default attributes or customize them as you like.")
            
    # -----------------------------------
    def _set_error_modules(self, error_modules):
        """
        set up and check error_modules
        """
        if not error_modules:
            default_modules = [QuantitativeErrorModule(), 
                               PuncErrorModule(),
                               SemanticErrorModule(), 
                               DistributionErrorModule(), 
                               StringSimilarityErrorModule(), 
                               CharSimilarityErrorModule()]
            self.error_modules = default_modules
        else:
            self._check_error_modules(error_modules)
            self.error_modules = error_modules

    # -----------------------------------
    def _predictModule(self, error_module: ErrorModule) -> list:
        """
        Loops over columns in col_predict, checks if the error module detector 
        applies to the columns data type and predicts erroneous rows for each column.

        :param error_module: ErrorModule object to use for error prediction on column values;

        :return: an error indicator matrix for the full dataset using the predictions of the current 
            error module. Each value's prediction is indicated as follows:
            - -1 indicates an error;
            - +1 indicates no error was predicted or that this error module is not applicable for the column's data type;
            - np.nan for columns not in col_predict.

        :rtype: a 2-dimensional array.
        """
        error_matrix = []
        for col in self.df:
            if col in self.col_predict:
                col_type = self.types[self.col_predict.index(col)]
                indicator_vector = np.full(self.n_rows, 1)
                if col_type in error_module.get_available_types():
                    col_vals = self.df[col].values
                    print(col)
                    erroneous_indices = error_module.get_erroneous_rows_in_col(col_vals)
                    indicator_vector[erroneous_indices] = -1
            else:
                indicator_vector = np.full(self.n_rows, np.nan)
            error_matrix.append(indicator_vector)

        return np.array(error_matrix).T
    
    # -----------------------------------
    def _get_final_error_matrix(self):
        """
        """
        final_error_matrix = np.full((self.n_rows, self.n_cols), np.nan)
        for matrix in list(self.module_error_matrix_dict.values()):
            final_error_matrix = np.where(matrix == -1, -1, np.where(np.isnan(matrix), final_error_matrix, 1))
        return final_error_matrix

    # -----------------------------------
    def _fit(self):
        """
        """
        self.n_rows = self.df.shape[0]
        self.n_cols = self.df.shape[1]
        return

    # -----------------------------------
    def _predict(self, df: pd.DataFrame):
        """
        """
        for module in self.error_modules:
            self.module_error_matrix_dict[module.module_name] = self._predictModule(module)
        return self._get_final_error_matrix()
    
    # ------------------------------------
    def get_error_module_matrix(self, error_module: str) -> np.array:
        """
        """
        module_keys = list(self.module_error_matrix_dict.keys())
        if not module_keys:
            raise ValueError(f"No error detection results predicted yet. Make sure to call the fit() and predict() functions first.")
        if error_module not in module_keys:
            raise KeyError(f"{error_module} is not part of the this error detection object. Please use the name of an error module present in this detector, you have the following options: {module_keys}")
        
        return self.module_error_matrix_dict[error_module]
