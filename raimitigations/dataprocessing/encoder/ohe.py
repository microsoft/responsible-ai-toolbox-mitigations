from typing import Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from .encoder import DataEncoding


class EncoderOHE(DataEncoding):
    """
    Concrete class that applies one-hot encoding over a dataset. The categorical
    features are encoded using the One-Hot encoding class from ``sklearn``. The main
    difference between using the ``sklearn`` implementation directly is that the
    transform method implemented here returns a data frame instead of a numpy array.
    This is useful when it is important to maintain the data frame format without
    losing the name of the columns. The new columns created for the one-hot encoding
    are named according to the original's dataset column name and the value that is
    one-hot encoded.

    :param df: pandas data frame that contains the columns to be encoded;

    :param col_encode: a list of the column names or indexes that will be encoded.
        If None, this parameter will be set automatically as being a list of all
        categorical variables in the dataset;

    :param drop: if True, drop the one-hot encoded column of the first category of a given
        feature. This way, a feature with N different categories will be encoded using
        N-1 one-hot encoded columns. This is useful when using models that does not
        work properly with colinear columns: when using all one-hot columns, each of these
        columns can be expressed as a linear combination of the other columns. By removing
        one of these columns using drop=True, we remove this colinearity. Note however that
        several models can work even with colinear columns;

    :param unknown_err: if True, when an unknwon category is encontered, an error is
        raaised. If False, when an unknown category is found, all encoded columns will be
        set to zero. Note that unknown_err = False does not work with drop = True;

    :param verbose: indicates whether internal messages should be printed or not.
    """

    # -----------------------------------
    def __init__(
        self,
        df: Union[pd.DataFrame, np.ndarray] = None,
        col_encode: list = None,
        drop: bool = True,
        unknown_err: bool = True,
        verbose: bool = True,
    ):
        super().__init__(df, col_encode, verbose)
        self.drop = drop
        self.unk_err = unknown_err
        if drop and not unknown_err:
            raise ValueError("ERROR: If drop = True, then unknown_err should be set to True.")
        if drop:
            drop = "first"
        else:
            drop = None
        if unknown_err:
            unk_err = "error"
        else:
            unk_err = "ignore"
        self.encoder = OneHotEncoder(drop=drop, dtype=np.int32, sparse=False, handle_unknown=unk_err)

    # -----------------------------------
    def _get_new_col_name(self):
        """
        Creates a list of new column names for the one-hot encoding
        columns created during the transform method of OneHotEncoder.
        """
        new_df_col = []
        new_col_dict = {col: [] for col in self.col_encode}
        for i, col in enumerate(self.col_encode):
            values = self.encoder.categories_[i]
            for j, value in enumerate(values):
                if not self.drop or j != self.encoder.drop_idx_[i]:
                    col_name = f"{col}_{value}"
                    new_df_col.append(col_name)
                    new_col_dict[col].append(col_name)

        self.ohe_col_dict = new_col_dict
        return new_df_col

    # -----------------------------------
    def _fit(self):
        """
        Steps for running the fit method for the current class. The following
        steps are executed: (i) obtain a subset of the dataset containing only
        the columns that should be encoded, (ii) call the fit method of the
        OneHotEncoder object, and (iii) create the new column names that will
        be associated with the one-hot encoding columns.
        """
        df_valid = self._get_df_subset(self.df, self.col_encode)
        self.encoder.fit(df_valid)
        self.new_col_names = self._get_new_col_name()

    # -----------------------------------
    def _build_transformed_df(self, org_df: pd.DataFrame, ohe_data: np.array):
        """
        Given the original dataset (with categorical columns) and the resulting
        dataset of the transform method of the OneHotEncoder, this method creates
        a new dataset by creating a name for each of the columns in ohe_data (which
        contains only one-hot encoded columns) using the _get_new_col_name method,
        and then copying the columns from the original dataset that weren't encoded
        to the new dataset.

        :param org_df: the original dataset containing all columns;
        :param ohe_data: dataset containing only the one-hot encoding columns of all
            columns in self.col_encode.
        """
        new_df = pd.DataFrame(ohe_data, columns=self.new_col_names)

        org_df.drop(columns=self.col_encode, inplace=True)
        for col in self.new_col_names:
            org_df[col] = new_df[col].values.tolist()

        return org_df

    # -----------------------------------
    def _transform(self, df: pd.DataFrame):
        """
        Steps for running the transform method for the current class.

        :param df: the full dataset with the columns to be encoded.
        """
        org_df = df.copy()
        ohe_data = self._get_df_subset(org_df, self.col_encode)
        ohe_data = self.encoder.transform(ohe_data)
        new_df = self._build_transformed_df(org_df, ohe_data)
        return new_df

    # -----------------------------------
    def get_encoded_columns(self):
        """
        Returns a list with the column names or column indices of the one-hot
        encoded columns. These are the columns created by the one-hot encoder
        that replaced the original columns.

        :return: a list with the column names or column indices of the one-hot
            encoded columns.
        :rtype: list
        """
        return self.new_col_names.copy()

    # -----------------------------------
    def _inverse_transform(self, df: Union[pd.DataFrame, np.ndarray]):
        self._check_if_fitted()
        org_df = df.copy()
        ohe_data = self._get_df_subset(df, self.new_col_names)
        revert_df = self.encoder.inverse_transform(ohe_data)
        revert_df = pd.DataFrame(revert_df, columns=self.col_encode)
        for col in df.columns:
            if col not in self.col_encode and col not in self.new_col_names:
                revert_df[col] = org_df[col].values.tolist()
        return revert_df
