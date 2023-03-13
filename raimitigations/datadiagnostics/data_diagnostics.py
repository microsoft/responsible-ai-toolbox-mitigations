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
    that can be used for different error prediction approaches.

    :param df: pandas dataframe or np.ndarray to predict errors over;

    :param col_predict: a list of the column names or indices that will be subject to error prediction.
        If None, this parameter will be set automatically as being a list of all columns;

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
    def _get_fit_input_type(self) -> int:
        """
        Returns the type of data format this class uses. 

        :return: 0 indicating a pandas dataframe data format;
        :rtype: int.
        """
        return self.FIT_INPUT_DF

    # -----------------------------------
    def _set_column_to_predict(self):
        """
        Sets the col_predict attribute representing columns to predict errors over. 
        If these columns are not provided, it defaults to all columns.
        """
        if self.col_predict is not None:
            return

        self.col_predict = self.df_info.columns.to_list()
        self.print_message("No columns specified for error prediction. Error prediction applied to all columns.")

    # -----------------------------------
    def _check_valid_col_predict(self):
        self.col_predict = self._check_error_col_list(self.df_info.columns, self.col_predict, "col_predict")

    # -----------------------------------
    def _set_column_data_types(self, num_thresh: float = 0.25, cat_thresh: float = 0.05) -> list:
        """
        Detects the data type of each column in col_predict. It has 3 data type options:
            - numerical
            - categorical
            - string

        :param num_thresh: a float threshold of the minimum ratio of float-like values of a numerical column;
        :param cat_thresh: a float threshold of the maximum ratio of unique string data of a categorical column;

        :return: a list of data types mapping to each column in col_predict;
        :rtype: a list.
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
        Abstract method. For a given concrete class, this method must run the
        error prediction steps implemented and save any important information 
        in a set of class-specific attributes. These attributes are then used 
        in the predict and transform methods for error prediction.
        """
        pass

    # -----------------------------------
    def fit(self, df: Union[pd.DataFrame, np.ndarray] = None):
        """
        Default fit method for all error predictors that inherit from the DataDiagnostics class. The
        following steps are executed: (i) set the dataset, (ii) set the list of columns that
        will be subject to error prediction, (iii) check for any invalid input, (iv) call the fit method of the
        child class.

        :param df: the full dataset to fit the error prediction class on.
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
        in col_predict columns of a dataset using the error prediction class implemented and
        return an error matrix.

        :param df: a pandas dataframe to perform error prediction on.
        """
        pass

    # -----------------------------------
    def predict(self, df: Union[pd.DataFrame, np.ndarray]) -> np.array:
        """
        Default predict method for all error predictors that inherit from 
        the DataDiagnostics class. Predicts errors in a given dataset over 
        columns specified in col_predict. Returns an error matrix.

        :param df: the dataset to perform error prediction over.

        :return: an error matrix of the same shape as the input data mapping an 
            error indicator to each value as follows:
            - -1 indicates an error;
            - +1 indicates no error was predicted or that this error module is 
                not applicable for the column's data type;
            - np.nan for columns not in col_predict.
        :rtype: a 2-dimensional np.array.
        """
        self._check_if_fitted()
        predict_df = self._fix_col_transform(df)
        self.error_matrix = self._predict(predict_df)
        self.predicted = True
        return self.error_matrix

    # -----------------------------------
    def transform(self, df: Union[pd.DataFrame, np.ndarray], imputer: DataImputer = None) -> pd.DataFrame:
        """
        The default transform function of all error predictors that inherit 
        from the DataDiagnostics class. Transforms an input dataset using 
        the error matrix predicted by the predict function. It removes erroneous 
        values in the data with the option to apply a fit/transform imputer object afterwards.

        :param df: the dataset to transform using the error matrix;
        :param imputer: an imputer object to fit to and impute this dataset. You can use
            imputers present in this library of type dataprocessing.DataImputer,
            your options are: BasicImputer(), IterativeDataImputer() or KNNDataImputer().
            If you'd like to leave erroneous values as np.nan, use None;

        :return: the transformed dataframe;
        :rtype: pandas dataframe.
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
        Abstract method. For a given concrete class, this method transforms 
        an input dataset using the error matrix predicted by the predict function.
        """
        pass
    '''
    # -----------------------------------

    def get_col_predict(self) -> list:
        """
        Returns a list with the column names or column indices of
            columns subject to error prediction.

        :return: a list with the column names or column indices of
            columns subject to error prediction.
        :rtype: list
        """
        return self.col_predict.copy()
