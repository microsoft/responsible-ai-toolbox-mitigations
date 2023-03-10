from abc import abstractmethod
from typing import Union

import numpy as np
import pandas as pd

from ..dataprocessing import DataProcessing
from ..dataprocessing.data_processing import DataFrameInfo
from ..dataprocessing.imputer.basic_imputer import DataImputer
from .utils import is_cat, is_num


class DataDiagnostics(DataProcessing):
    """
    Base class for all data diagnostics subclasses. Implements basic functionalities
    that can be used for different error detection approaches.

    :param df: pandas data frame to detect errors in;

    :param col_predict: a list of the column names or indexes that will be subject to error detection.
        If None, this parameter will be set automatically as being a list of all feature
        columns;

    :param verbose: indicates whether internal messages should be printed or not.
    """

    # -----------------------------------
    def __init__(self, df: Union[pd.DataFrame, np.ndarray] = None, col_predict: list = None, verbose: bool = True):
        super().__init__(verbose)
        self.df_info = DataFrameInfo()
        self._set_df(df)
        self.n_rows = None
        self.n_cols = None
        self.types = []
        self.col_predict = col_predict
        self.fitted = False
        self.predicted = False

    # -----------------------------------
    def _get_fit_input_type(self):
        return self.FIT_INPUT_DF

    # -----------------------------------
    def _set_column_to_predict(self):
        """
        Sets the columns to detect errors in (col_predict) automatically
        if these columns are not provided, set to be all feature columns.
        """
        if self.col_predict is not None:
            return

        self.col_predict = self.df_info.columns.to_list()
        self.print_message("No columns specified for error detection. Error detection applied to all columns.")

    # -----------------------------------
    def _check_valid_col_predict(self):
        self.col_predict = self._check_error_col_list(self.df_info.columns, self.col_predict, "col_predict")

    # -----------------------------------
    def _set_column_data_types(self, cat_thresh: int = 0.05, num_thresh: float = 0.25) -> list:
        """
        Finds the data type of each column in col_predict. It has 3 data type options:
            - numerical
            - categorical
            - string

        :param cat_thresh:;
        :param num_thresh: ;
        :param addr_thresh: ;
        """
        for col in self.col_predict:
            col_vals = self.df_info.df[col].values
            if is_num(col_vals, num_thresh):
                self.types.append("numerical")
            elif is_cat(col_vals, cat_thresh):
                self.types.append("categorical")
            else:
                self.types.append("string")

    # -----------------------------------
    @abstractmethod
    def _fit(self):
        """
        Abstract method. For a given concrete class, this method must
        create the error detection module implemented and save any
        important information in a set of class-specific attributes to be used for error prediction.
        """
        pass

    # -----------------------------------
    def fit(self, df: Union[pd.DataFrame, np.ndarray] = None):
        """
        Default fit method for all encoders that inherit from the ErrorDetection class. The
        following steps are executed: (i) set the dataset, (ii) set the list of columns that
        will be subject to error detection, (iii) check for any invalid input, (iv) call the fit method of the
        child class.

        :param df: the full dataset to fit the error detection module on;
        """
        self._set_df(df, require_set=True)
        self._set_column_to_predict()
        self._check_valid_col_predict()
        self._set_column_data_types()
        self._fit()
        self.fitted = True
        self.error_matrix = None
        return self

    # -----------------------------------
    @abstractmethod
    def _predict(self, df: pd.DataFrame):
        """
        Abstract method. For a given concrete class, this method must predict errors present
        in col_predict columns of a dataset using the error detection module implemented and
        return the errors in the data.

        :param df: the full dataset to perform error detection on.
        """
        pass

    # -----------------------------------
    def predict(self, df: Union[pd.DataFrame, np.ndarray]) -> dict:
        """
        Predicts errors in a given dataset in all columns in col_predict.
        Returns all data with detected errors.

        :param df: the full dataset to perform error detection on..
        :return: a dictionary mapping each error module to an error matrix. Each value's prediction is indicated as follows:
            - -1 indicates an error;
            - +1 indicates no error was predicted or that this error module is not applicable for the column's data type;
            - np.nan for columns not in col_predict.

        :rtype: a dictionary.
        """
        self._check_if_fitted()
        predict_df = self._fix_col_transform(df)
        self.error_matrix = self._predict(predict_df)
        self.predicted = True
        return self.error_matrix

    # -----------------------------------
    def transform(self, df: Union[pd.DataFrame, np.ndarray], imputer: DataImputer = None) -> pd.DataFrame:
        """
        :param imputer: an imputer object to fit to and impute this dataset. You can use
            imputers present in this library of type dataprocessing.imputer.DataImputer(),
            your options are: BasicImputer(), IterativeDataImputer() or KNNDataImputer().
            If you'd like to leave erroneous values as np.nan, use None.
        """
        self._check_if_predicted()
        transf_df = self._fix_col_transform(df)
        error_mask = np.where(self.error_matrix == -1, False, True)
        mask_df = pd.DataFrame(data=error_mask, columns=transf_df.columns)
        masked_df = transf_df[mask_df]
        if imputer is None:
            return masked_df
        else:
            imputer.fit(masked_df)
            new_df = imputer.transform(masked_df)
            return new_df

    '''
    # -----------------------------------
    @abstractmethod
    def _transform(self, df: pd.DataFrame, imputer: DataImputer = None) -> pd.DataFrame:
        """
        Abstract method. For a given concrete class, this method
        """
        pass
    '''
    # -----------------------------------

    def get_col_predict(self):
        """
        Returns a list with the column names or column indices of
            columns subject to error detection.

        :return: a list with the column names or column indices of
            columns subject to error detection.
        :rtype: list
        """
        return self.col_predict.copy()
