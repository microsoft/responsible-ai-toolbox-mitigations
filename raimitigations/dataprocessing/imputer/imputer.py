from abc import abstractmethod
from typing import Union, Tuple

import numpy as np
import pandas as pd

from ..data_processing import DataProcessing, DataFrameInfo
from ..encoder.ordinal import EncoderOrdinal
from ...utils.data_utils import (
    get_cat_cols,
    _transform_ordinal_encoder_with_new_values,
    _inverse_transform_ordinal_encoder_with_new_values,
)


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
        self.df_info = DataFrameInfo()
        self.fitted = False
        self._set_df(df)
        self.col_impute = col_impute
        self.do_nothing = False
        self.valid_cols = []
        self.enable_encoder = False
        self.ordinal_encoder = None
        self.none_status = False

    # -----------------------------------
    def _set_columns_to_impute(self):
        """
        Sets the columns to impute (col_impute) automatically
        if these columns are not provided. col_impute is set
        to be all columns.
        """
        if self.col_impute is not None:
            return

        self.none_status = True
        self.col_impute = self.df_info.columns.to_list()

    # -----------------------------------
    def _reset_columns_to_impute(self, df: Union[pd.DataFrame, np.ndarray]):
        """
        Resets the columns to impute (col_impute) automatically if col_impute
        wasn't provided. This occurs at transform time in case the dataframe
        to be transformed has new columns with missing values. col_impute is set
        to be all columns with at least one NaN value in them.

        :param df: pandas dataframe used to identify columns with missing values.
        """
        if not self.none_status:
            return
        col_nan_status = df.isna().any()
        col_with_nan = []
        for i, value in enumerate(col_nan_status.index):
            if col_nan_status[value]:
                if type(value) == int:
                    col_with_nan.append(i)
                else:
                    col_with_nan.append(value)
        self.print_message(
            f"No columns specified for imputation. These columns "
            + f"have been automatically identified at transform time:\n{col_with_nan}"
        )
        self.col_impute = col_with_nan

    # -----------------------------------
    def _check_valid_input(self):
        self.col_impute = self._check_error_col_list(self.df_info.columns, self.col_impute, "col_impute")

        if self.do_nothing:
            self.print_message("WARNING: No columns with NaN values identified. Nothing to be done.")

    # -----------------------------------
    def _get_fit_input_type(self):
        return self.FIT_INPUT_DF

    # -----------------------------------
    def _check_transf_data_structure(self, df: Union[pd.DataFrame, np.ndarray]):
        """
        Checks that all columns seen at fit time are present
        at transform time and vice versa.

        :param df: pandas dataframe used at transform time.
        """
        for col in self.valid_cols:
            if col not in list(df):
                raise KeyError(f"ERROR: Column: {col} seen at fit time, but not present in dataframe.")
        for col in self.col_impute:
            if col not in self.valid_cols:
                raise KeyError(f"ERROR: Column: {col} not seen at fit time.")

    # -----------------------------------
    def _apply_encoding_fit(self) -> Union[pd.DataFrame, np.ndarray]:
        """
        Creates and fits an ordinal encoder to all categorical data before
        fitting the imputer if enable_encoder=True, otherwise it excludes
        all categorical data from the imputation process.

        :return: resulting pandas dataframe or np.ndarray
        :rtype: pd.dataframe or np.ndarray
        """
        all_cat_cols = get_cat_cols(self.df_info.df)
        all_num_cols = [value for value in list(self.df_info.df) if value not in all_cat_cols]

        df_valid = pd.DataFrame()

        if self.enable_encoder is False:
            self.print_message(
                "\nWARNING: Categorical columns will be excluded from the iterative imputation process.\n"
                + "If you'd like to include these columns, you need to set 'enable_encoder'=True.\n"
                + "If you'd like to use a different type of encoding before imputation, consider using the Pipeline "
                + "class and call your own encoder before calling this subclass for imputation.",
            )
            df_valid = self._get_df_subset(self.df_info.df, all_num_cols)

        else:
            self.print_message(
                "\nWARNING: 'enable_encoder'=True and categorical columns will be encoded using ordinal encoding before "
                + "applying the iterative imputation process.\n"
                + "If you'd like to use a different type of encoding before imputation, consider using the Pipeline class "
                + "and call your own encoder before calling this subclass for imputation.",
            )
            self.ordinal_encoder = EncoderOrdinal(df=self.df_info.df, col_encode=all_cat_cols, unknown_value=np.nan)
            self.ordinal_encoder.fit()
            df_valid = self.ordinal_encoder.transform(self.df_info.df)

        self.valid_cols = list(df_valid)

        return df_valid

    # -----------------------------------
    def _apply_encoding_transf(self, df_valid: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Encodes categorical data before applying the imputer's transform function using
        the mapping of the encoder created at fit time while also encoding new values
        on top of the original mapping.

        :param df_valid: pandas dataframe to encode;

        :return: resulting pandas dataframe or np.ndarray
        :rtype: pd.dataframe or np.ndarray
        """
        all_cat_cols = get_cat_cols(df_valid)
        if self.enable_encoder is False:
            if len(all_cat_cols) > 0:
                raise ValueError(
                    "ERROR: Categorical data unseen at fit time and can't be included in the iterative imputation process without encoding.\n"
                    + "If you'd like to ordinal encode and impute these columns, use 'enable_encoder'=True.\n"
                    + "Note that encoded columns are not guaranteed to reverse transform if they have imputed values.\n"
                    + "If you'd like to use a different type of encoding before imputation, consider using the Pipeline "
                    + "class and call your own encoder before calling this subclass."
                )
            df_to_transf = df_valid

        else:
            self.print_message(
                "\nWARNING: Note that encoded columns are not guaranteed to reverse transform if they have imputed values.\n"
                + "If you'd like to use a different type of encoding before imputation, consider using the Pipeline class and "
                + "call your own encoder before calling this subclass."
            )

            df_to_transf, self.inverse_mapping = _transform_ordinal_encoder_with_new_values(
                self.ordinal_encoder, df_valid
            )

        return df_to_transf

    # -----------------------------------
    def _revert_encoding(
        self, transf_df: Union[pd.DataFrame, np.ndarray], missing_value_cols_no_impute: list
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Attempts to revert the encoding of all categorical data in the set,
        whether it was imputed or not.

        :param transf_df: pandas dataframe post possible encoding and imputation;
        :param missing_value_cols_no_impute: list of columns with missing values but not to be imputed;

        :return: resulting pandas dataframe or np.ndarray
        :rtype: pd.dataframe or np.ndarray
        """
        # inverse_transform encoded columns
        if self.enable_encoder is True:
            self.inverse_mapping = dict(
                (k, self.inverse_mapping[k]) for k in self.inverse_mapping if k not in missing_value_cols_no_impute
            )
            transf_df = _inverse_transform_ordinal_encoder_with_new_values(self.inverse_mapping, transf_df)
            self.print_message(f"\nImputed categorical columns' reverse encoding transformation complete.")

        return transf_df

    # -----------------------------------
    def _pre_process_transform(
        self, df: Union[pd.DataFrame, np.ndarray], missing_values_param: any
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], list]:
        """
        Applies pre-processing steps before transform, these include:
        - Resetting col_impute if it was originally None using the df to be transformed.
        - Checking that the structure of the df to be transformed matches that of the dataframe at fit.
        - Encodes categorical data if enable_encoder=True and excludes them otherwise.
        - Saves a list of columns with missing values but aren't set to be imputed in col_impute.

        :param df: pandas dataframe to be transformed by the imputer;
        :param missing_values_param: missing value parameter specified for
            sklearn's imputer object in dict passed by user;

        :return: pre-processed pandas dataframe or np.ndarray, list of columns with missing values but not to be imputed.
        :rtype: pd.dataframe or np.ndarray, list
        """
        self._reset_columns_to_impute(df)
        self._check_transf_data_structure(df)

        df_valid = self._get_df_subset(df, self.valid_cols)
        df_to_transf = self._apply_encoding_transf(df_valid)
        missing_value_cols = [
            col
            for col in list(df_to_transf)
            if (missing_values_param in df_to_transf[col] or df_to_transf[col].isna().any())
        ]
        missing_value_cols_no_impute = list(set(missing_value_cols) - set(self.col_impute))

        return df_to_transf, missing_value_cols_no_impute

    # -----------------------------------
    def _post_process_transform(
        self,
        transf_df: Union[pd.DataFrame, np.ndarray],
        df: Union[pd.DataFrame, np.ndarray],
        missing_value_cols_no_impute: list,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Applies post-processing steps after transform, these include:
        - Reverts possible encoding of categorical data..
        - Reverts columns that had missing values and were automatically imputed
            but weren't listed in col_impute to their original state.
        - Re-adds categorical columns that were possibly excluded at fit time with
            enable_encoder=False.

        :param transf_df: pandas dataframe that was transformed by the imputer;
        :param df: original pandas dataframe passed to the imputer's transform
        :param missing_value_cols_no_impute: list of columns with missing values but not to be imputed;

        :return: post-processed pandas dataframe or np.ndarray
        :rtype: pd.dataframe or np.ndarray
        """
        transf_df = self._revert_encoding(transf_df, missing_value_cols_no_impute)
        transf_df[missing_value_cols_no_impute] = df[missing_value_cols_no_impute]

        non_valid_cols = list(set(list(df)) - set(self.valid_cols))
        for col in non_valid_cols:
            transf_df[col] = df[col]

        return transf_df

    # -----------------------------------
    def fit(self, df: Union[pd.DataFrame, np.ndarray] = None, y: Union[pd.Series, np.ndarray] = None):
        """
        Default fit method for all imputation methods that inherits from the current
        class. The following steps are executed: (i) set the self.df_info attribute,
        (ii) set the list of columns to impute (or create a default one if needed),
        (iii) check if the dataset provided is valid (contains all columns that
        should be imputed), and (iv) call the concrete class's specific ``_fit`` method.

        :param df: the full dataset;
        :param y: ignored. This exists for compatibility with the sklearn's Pipeline class.
        """
        if self.do_nothing:
            return
        self._set_df(df, require_set=True)
        self._set_columns_to_impute()
        self._check_valid_input()
        self._fit()
        self.fitted = True
        self.df_info.clear_df_mem()
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
