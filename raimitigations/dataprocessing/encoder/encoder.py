from abc import abstractmethod
from typing import Union

import numpy as np
import pandas as pd

from ..data_processing import DataProcessing
from ..data_utils import get_cat_cols


class DataEncoding(DataProcessing):
    """
    Base class for all encoding subclasses. Implements basic functionalities
    that can be used by all encoding approaches.

    :param df: pandas data frame that contains the columns to be encoded;

    :param col_encode: a list of the column names or indexes that will be encoded.
        If None, this parameter will be set automatically as being a list of all
        categorical variables in the dataset;

    :param verbose: indicates whether internal messages should be printed or not.
    """

    # -----------------------------------
    def __init__(self, df: Union[pd.DataFrame, np.ndarray] = None, col_encode: list = None, verbose: bool = True):
        super().__init__(verbose)
        self.df = None
        self._set_df(df)
        self.col_encode = col_encode
        self.do_nothing = False
        self.fitted = False

    # -----------------------------------
    def _set_column_to_encode(self):
        """
        Sets the columns to encode (col_encode) automatically
        if these columns are not provided. We consider that
        only categorical columns must be encoded. Therefore, we
        automatically check which columns are possibly categorical.
        """
        if self.col_encode is not None:
            return

        cat_col = get_cat_cols(self.df)
        self.col_encode = cat_col
        self.print_message(
            f"No columns specified for encoding. These columns "
            f"have been automatically identfied as the following:\n{self.col_encode}"
        )

    # -----------------------------------
    def _check_valid_input(self):
        self.col_encode = self._check_error_col_list(self.df, self.col_encode, "col_encode")

    # -----------------------------------
    def _get_fit_input_type(self):
        return self.FIT_INPUT_DF

    # -----------------------------------
    @abstractmethod
    def _fit(self):
        """
        Abstract method. For a given concrete class, this method must
        create and execute the encoding method implemented and save any
        important information in a set of class-specific attributes. These
        attributes are then used to transform a dataset and encode it.
        """
        pass

    # -----------------------------------
    def fit(self, df: Union[pd.DataFrame, np.ndarray] = None, y: Union[pd.Series, np.ndarray] = None):
        """
        Default fit method for all encoders that inherit from the DataEncoding class. The
        following steps are executed: (i) set the dataset, (ii) set the list of columns that
        will be encoded, (iii) check for any invalid input, (iv) call the fit method of the
        child class.

        :param df: the full dataset;
        :param y: ignored. This exists for compatibility with the sklearn's Pipeline class.
        """
        self._set_df(df, require_set=True)
        self._set_column_to_encode()
        self._check_valid_input()
        self._fit()
        self.fitted = True
        return self

    # -----------------------------------
    @abstractmethod
    def _transform(self, df: pd.DataFrame):
        """
        Abstract method. For a given concrete class, this method must execute the
        transformation of a dataset using the encoding method implemented and
        return the encoded dataset.

        :param df: the full dataset with the columns to be encoded.
        """
        pass

    # -----------------------------------
    def transform(self, df: Union[pd.DataFrame, np.ndarray]):
        """
        Transforms a given dataset by encoding all columns specified by the
        col_encode parameter. Returns a dataset with the encoded columns.

        :param df: the full dataset with the columns to be encoded.
        :return: the transformed dataset.
        :rtype: pd.DataFrame or np.ndarray
        """
        self._check_if_fitted()
        transf_df = self._fix_col_transform(df)
        transf_df = self._transform(transf_df)
        return transf_df

    # -----------------------------------
    def get_encoded_columns(self):
        """
        Returns a list with the column names or column indices of the
        encoded columns.

        :return: a list with the column names or column indices of the
            encoded columns.
        :rtype: list
        """
        return self.col_encode.copy()
