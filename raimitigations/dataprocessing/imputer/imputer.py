from abc import abstractmethod
from typing import Union

import numpy as np
import pandas as pd

from ..data_processing import DataProcessing


class DataImputer(DataProcessing):
    """
    Base class for all imputer subclasses. Implements basic functionalities
    that can be used by all imputation approaches.

    :param df: pandas data frame that contains the columns to be encoded;

    :param col_impute: a list of the column names or indexes that will be imputed.
        If None, this parameter will be set automatically to be the list of columns
        with at least one NaN value;

    :param verbose: indicates whether internal messages should be printed or not.
    """

    COL_NAME = 0
    COL_INDEX = 1

    # -----------------------------------
    def __init__(self, df: Union[pd.DataFrame, np.ndarray] = None, col_impute: list = None, verbose: bool = True):
        super().__init__(verbose)
        self.df = None
        self.fitted = False
        self._set_df(df)
        self.col_impute = col_impute
        self.do_nothing = False

    # -----------------------------------
    def _set_column_to_impute(self):
        """
        Sets the columns to impute (col_impute) automatically
        if these columns are not provided. col_impute is set
        to be all columns with at least one NaN value in them.
        """
        if self.col_impute is not None:
            return

        self.col_impute = self.df.columns.to_list()

        col_nan_status = self.df.isna().any()
        col_with_nan = []
        for i, value in enumerate(col_nan_status.index):
            if col_nan_status[value]:
                if type(value) == int:
                    col_with_nan.append(i)
                else:
                    col_with_nan.append(value)

        self.print_message(
            f"No columns specified for imputation. These columns "
            + f"have been automatically identified:\n{col_with_nan}"
        )

    # -----------------------------------
    def _check_valid_input(self):
        self.col_impute = self._check_error_col_list(self.df, self.col_impute, "col_impute")

        if self.do_nothing:
            self.print_message("WARNING: No columns with NaN values identified. Nothing to be done.")

    # -----------------------------------
    def _get_fit_input_type(self):
        return self.FIT_INPUT_DF

    # -----------------------------------
    def fit(self, df: Union[pd.DataFrame, np.ndarray] = None, y: Union[pd.Series, np.ndarray] = None):
        """
        Default fit method for all imputation methods that inherits from the current
        class. The following steps are executed: (i) set the self.df attribute,
        (ii) set the list of columns to impute (or create a default one if needed),
        (iii) check if the dataset provided is valid (contains all columns that
        should be imputed), and (iv) call the concrete class's specific ``_fit`` method.

        :param df: the full dataset;
        :param y: ignored. This exists for compatibility with the sklearn's Pipeline class.
        """
        if self.do_nothing:
            return
        self._set_df(df, require_set=True)
        self._set_column_to_impute()
        self._check_valid_input()
        self._fit()
        self.fitted = True
        return self

    # -----------------------------------
    @abstractmethod
    def _fit(self):
        """
        Abstract method. For a given concrete class, this method must run the
        imputation steps of the imputer method implemented and save any
        important information in a set of class-specific attributes. These
        attributes are then used in the transform method to impute the missing
        data
        """
        pass

    # -----------------------------------
    @abstractmethod
    def _transform(self, df: pd.DataFrame):
        """
        Abstract method. For a given concrete class, this method must execute the
        imputation of a dataset using the imputation method implemented and a new
        dataset with no missing values in the columns in self.col_impute.

        :param df: the full dataset with the columns to be imputed.
        """
        pass

    # -----------------------------------
    def transform(self, df: Union[pd.DataFrame, np.ndarray]):
        """
        Default behavior for transforming the data for the different
        imputation methods.

        :param df: the full dataset with the columns to be imputed.
        :return: the transformed dataset.
        :rtype: pd.DataFrame or np.ndarray
        """
        self._check_if_fitted()
        transf_df = self._fix_col_transform(df)
        if self.do_nothing:
            return df

        transf_df = self._transform(transf_df)

        return transf_df
