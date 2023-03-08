from abc import ABC, abstractmethod
from typing import *

import pandas as pd
import numpy as np

class DataDiagnostics(ABC):
    """
    Base class for all classes present in the DataDiagnostics module
    of the RAIMitigation library. Implements basic functionalities
    that can be used throughout different mitigations.

    :param verbose: indicates whether internal messages should be printed or not.
    """
    COL_NAME = 0
    COL_INDEX = 1

    # -----------------------------------
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.column_type = self.COL_NAME
        self.n_rows: int = None
        self.n_cols: int = None

    # -----------------------------------
    def print_message(self, text: str):
        if self.verbose:
            print(text)

    # -----------------------------------
    def _check_error_df(self, df):
        # Check consistency of the df param
        if df is not None and type(df) != pd.DataFrame and type(df) != np.ndarray:
            class_name = type(self).__name__
            raise ValueError(
                f"ERROR: expected parameter 'df' of {class_name} to be of type pandas.DataFrame "
                + f"or numpy.ndarray, but got a parameter of type {type(df)} instead."
            )

    # -----------------------------------
    def _fix_num_col(self, df: pd.DataFrame, label_col: Union[int, str] = None):
        """
        Checks if the column names of the dataset are present or not. If not, create
        valid column names using the index number (converted to string) of each column.
        Also check if the label_col parameter is provided as an index or as a column
        name. In the former case, convert the label_col to a column name. Finally,
        create a dictionary that maps each column index to the column name. This is
        used when a the object uses the transform_pipe parameter. In these cases,
        the dataset might change its column structure, but we need to map these changes
        and guarantee that the indices provided by the user for a given transformation
        (provided before any transformation is applied) are mapped to the correct columns
        even if these columns are changed by other transforms in the transform_pipe.

        :param df: the full dataset;
        :param label_col: the name or index of the label column;
        :return: if label_col is None, returns only the fixed dataset. Otherwise, return
            a tuple (df, label_col) containing the fixed dataset and the fixed label
            column, respectively.
        :rtype: pd.DataFrame or a tuple
        """
        column_type = self.COL_NAME
        if type(df.columns[0]) != str:
            column_type = self.COL_INDEX
            df.columns = [str(i) for i in range(df.shape[1])]
            if label_col is not None:
                label_col = str(label_col)

        self.col_index_to_name = {i: df.columns[i] for i in range(df.shape[1])}
        self.column_type = column_type

        if label_col is not None:
            if self.column_type == self.COL_INDEX:
                label_col = str(label_col)
            else:
                if type(label_col) == int:
                    label_col = self._get_column_from_index(label_col)
            return df, label_col

        return df

    # -----------------------------------
    def _numpy_array_to_df(self, df: Union[pd.DataFrame, np.ndarray]):
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df)
        return df

    # -----------------------------------
    def _fix_col_predict(self, df: Union[pd.DataFrame, np.ndarray]):
        """
        Checks if the column names of the dataset are present or not. If not, create
        valid column names using the index number (converted to string) of each column.

        :param df: the full dataset;
        :return: the fixed dataset.
        :rtype: pd.DataFrame
        """
        if isinstance(df, np.ndarray):
            df = self._numpy_array_to_df(df)
        invalid = False
        if type(df.columns[0]) != str:
            if self.column_type == self.COL_NAME:
                invalid = True
            else:
                df.columns = [str(i) for i in range(df.shape[1])]

        if invalid:
            raise ValueError(
                "ERROR: the columns of the dataset provided to the transform() method from class "
                + f"{type(self).__name__} does not match with the columns provided during the fit() method."
            )
        return df

    # -----------------------------------
    def _get_column_from_index(self, column_index: int):
        """
        Get the column name associated to a given column index.

        :param column_index: the column index.
        :return: the column name associated to the column specified by the index column_index.
        :rtype: str
        """
        if column_index not in self.col_index_to_name.keys():
            raise ValueError(
                f"ERROR: invalid index provided to the class {type(self).__name__}.\n"
                + f"Error caused by the following index: {column_index}"
            )
        return self.col_index_to_name[column_index]

    # -----------------------------------
    def _check_error_col_list(self, df: pd.DataFrame, col_list: list, col_var_name: str):
        """
        For a given dataset df, check if all column names in col_list are present
        in df. col_list can be a list of column names or column indexes. If one of
        the column names or indexes is not present in df, a ValueError is raised. If
        the col_list parameter is made up of integer values (indices) and the dataframe
        has column names, return a new column list using the column names instead.

        :param df: the dataframe that should be checked;
        :param col_list: a list of column names or column indexes;
        :param col_var_name: a name that identifies where the error occurred (if a
            ValueError is raised). This method can be called from many child classes,
            so this parameter shows the name of the parameter from the child class
            that caused the error.
        :return: the col_list parameter. If the col_list parameter is made up of integer
            values (indices) and the dataframe has column names, return a new column list
            using the column names instead.
        :rtype: list
        """
        if type(col_list) != list:
            raise ValueError(
                f"ERROR: the parameter '{col_var_name}' must be a list of column names."
                + f" Each of these columns must be present in the DataFrame 'df'."
            )

        if not col_list:
            self.do_nothing = True
        elif df is not None:
            if type(col_list[0]) != int and type(col_list[0]) != str:
                raise ValueError(f"ERROR: '{col_var_name}' must be a list of strings or a list of integers.")

            if type(col_list[0]) == int:
                if self.column_type == self.COL_NAME:
                    col_list = [self._get_column_from_index(index) for index in col_list]
                else:
                    col_list = [str(val) for val in col_list]

            missing = [value for value in col_list if value not in df.columns]
            if missing:
                err_msg = (
                    f"ERROR: at least one of the columns provided in the '{col_var_name}' param is "
                    f"not present in the 'df' dataframe. The following columns are missing:\n{missing}"
                )
                raise ValueError(err_msg)
        return col_list

    # -----------------------------------
    def _check_if_fitted(self):
        if not self.fitted:
            raise ValueError(
                f"ERROR: trying to call the transform() method from an instance of the {self.__class__.__name__} class "
                + "before calling the fit() method. "
                + "Call the fit() method before using this instance to transform a dataset."
            )

    # -----------------------------------
    def _set_df(self, df: Union[pd.DataFrame, np.ndarray], require_set: bool = False):
        """
        Sets the current dataset self.df using a new dataset df. If both
        self.df and df are None, then a ValueError is raised. df can be None
        if a valid self.df has already been set beforehand.

        :param df: the full dataset;
        :param require_set: a boolean value indicating if the df parameter must
            be a valid dataframe or not. If true and df is None, an error is raised.
        """
        self._check_error_df(df)
        if self.df is None and df is None and require_set:
            raise ValueError(
                "ERROR: dataframe not provided. You need to provide the dataframe "
                + "through the class constructor or through the fit() method."
            )
        if df is not None:
            if isinstance(df, np.ndarray):
                df = self._numpy_array_to_df(df)
            self.df = self._fix_num_col(df)
