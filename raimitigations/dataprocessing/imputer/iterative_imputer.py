from typing import Union
import pandas as pd
import numpy as np
import sys

from sklearn.experimental import enable_iterative_imputer  # noqa # pylint: disable=unused-import
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from ..encoder import EncoderOrdinal
from .imputer import DataImputer
from ...utils.data_utils import get_cat_cols


class IterativeDataImputer(DataImputer):
    """
    Concrete class that imputes missing data of a feature using the other features. It uses a round-robin method
    of modeling each feature with missing values to be imputed as a function of the other features.
    This subclass uses the :class:`~sklearn.impute.IterativeImputer` class from :mod:`sklearn` in the
    background (note that this sklearn class is still in an experimental stage).
    sklearn.impute.IterativeImputer can only handle numerical data, however, this subclass allows for categorical
    input by applying ordinal encoding before calling the sklearn class. In order to use this function,
    use enable_encoder=True. Note that encoded columns are not guaranteed to reverse transform if they have imputed values.
    If you'd like to use a different type of encoding before imputation, consider using the Pipeline class and
    call your own encoder before calling this subclass for imputation.
    For more details see:
    https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#

    :param df: pandas data frame that contains the columns to be imputed;

    :param col_impute: a list of the column names or indexes that will be imputed.
        If None, this parameter will be set automatically as being a list of all
        columns with any NaN value;

    :param enable_encoder: a boolean flag to allow for applying ordinal encoding of categorical data before applying the IterativeImputer since it only accepts numerical values.

    :param iterative_params: a dict indicating the parameters used by
        :class:`~sklearn.impute.IterativeImputer`. The dict has the following structure:

            | {
            |   **'estimator'**:BayesianRidge(),
            |   **'missing_values'**:np.nan,
            |   **'sample_posterior'**:False,
            |   **'max_iter'**:10,
            |   **'tol'**:1e-3,
            |   **'n_nearest_features'**:None,
            |   **'initial_strategy'**:'mean',
            |   **'imputation_order'**:'ascending',
            |   **'skip_complete'**:False,
            |   **'min_value'**:-np.inf,
            |   **'max_value'**:np.inf,
            |   **'random_state'**:None
            | }

        where these are the parameters used by sklearn's IterativeImputer. If None,
        this dict will be auto-filled as the one above.
        ``Note: initial_strategy can take one of these values: ['mean', 'median', 'most_frequent', 'constant']``

    :param verbose: indicates whether internal messages should be printed or not.
    """

    # -----------------------------------
    def __init__(
        self,
        df: Union[pd.DataFrame, np.ndarray] = None,
        col_impute: list = None,
        enable_encoder: bool = False,
        iterative_params: dict = None,
        verbose: bool = True,
    ):
        super().__init__(df, col_impute, verbose)
        self.enable_encoder = enable_encoder
        self.iterative_params = iterative_params
        self._set_dicts()

    # -----------------------------------
    def _set_dicts(self):
        """
        If the 'iterative_params' dictionary that specifies how to impute given data
        is set to None, then create a default dictionary.
        """
        if self.iterative_params is None:
            self.iterative_params = {
                "estimator": BayesianRidge(),
                "missing_values": np.nan,
                "sample_posterior": False,
                "max_iter": 10,
                "tol": 1e-3,
                "n_nearest_features": None,
                "initial_strategy": "mean",
                "imputation_order": "ascending",
                "skip_complete": False,
                "min_value": -np.inf,
                "max_value": np.inf,
                "random_state": None,
            }

    # -----------------------------------
    def _check_valid_dict(self):
        """
        Checks if 'iterative_params' dictionary that specifis how to impute given data
        is appropriately set.
        """
        param_err = type(self.iterative_params) != dict

        if param_err:
            raise ValueError("ERROR: 'iterative_params' is not a dict. Check the documentation for more information.")

        keys = self.iterative_params.keys()
        sklearn_params = [
            "estimator",
            "missing_values",
            "sample_posterior",
            "max_iter",
            "tol",
            "n_nearest_features",
            "initial_strategy",
            "imputation_order",
            "skip_complete",
            "min_value",
            "max_value",
            "random_state",
        ]
        for sklearn_param in sklearn_params:
            if sklearn_param not in keys:
                raise ValueError(
                    "ERROR: expected the key "
                    + f"{sklearn_param}"
                    + " in the dictionary:\n"
                    + f"iterative_params: {self.iterative_params}"
                )

    # -----------------------------------
    def _fit(self):
        """
        Fit method complement used specifically for the current class.
        The following steps are executed: (i) check if 'self.iterative_params'
        dictionary is properly set, (ii) check for categorical columns
        (iii) exclude categorical columns if 'enable_encoder' is false,
        otherwise, apply ordinal encoder, (iv) create and fit the
        IterativeImputer object over df.
        """
        self._check_valid_dict()

        all_cat_cols = get_cat_cols(self.df)
        all_num_cols = [value for value in list(self.df) if value not in all_cat_cols]

        df_valid = pd.DataFrame()

        if not isinstance(self.enable_encoder, bool):
            raise ValueError("ERROR: 'enable_encoder' is a boolean parameter, use True/False.")

        elif self.enable_encoder is False:
            self.print_message(
                "\nWARNING: Categorical columns will be excluded from the iterative imputation process.\n"
                + "If you'd like to include these columns, you need to set 'enable_encoder'=True.\n"
                + "If you'd like to use a different type of encoding before imputation, consider using the Pipeline "
                + "class and call your own encoder before calling this subclass for imputation.",
            )
            df_valid = self._get_df_subset(self.df, all_num_cols)

        else:
            self.print_message(
                "\nWARNING: 'enable_encoder'=True and categorical columns will be encoded using ordinal encoding before "
                + "applying the iterative imputation process.\n"
                + "If you'd like to use a different type of encoding before imputation, consider using the Pipeline class "
                + "and call your own encoder before calling this subclass for imputation.",
            )
            ordinal_encoder = EncoderOrdinal(df=self.df, col_encode=all_cat_cols, unknown_value=np.nan)
            ordinal_encoder.fit()
            df_valid = ordinal_encoder.transform(self.df)

        self.valid_cols = list(df_valid)
        self.imputer = IterativeImputer(
            estimator=self.iterative_params["estimator"],
            missing_values=self.iterative_params["missing_values"],
            sample_posterior=self.iterative_params["sample_posterior"],
            max_iter=self.iterative_params["max_iter"],
            tol=self.iterative_params["tol"],
            n_nearest_features=self.iterative_params["n_nearest_features"],
            initial_strategy=self.iterative_params["initial_strategy"],
            imputation_order=self.iterative_params["imputation_order"],
            skip_complete=self.iterative_params["skip_complete"],
            min_value=self.iterative_params["min_value"],
            max_value=self.iterative_params["max_value"],
            random_state=self.iterative_params["random_state"],
            verbose=int(self.verbose),
        )
        self.imputer.fit(df_valid)

    # -----------------------------------
    def _transform(self, df: pd.DataFrame):
        """
        Transform method complement used specifically for the current class.

        :param df: the full dataset being transformed.
        """
        if self.none_status is True:
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

        self._check_transf_data_structure(df)

        df_valid = self._get_df_subset(df, self.valid_cols)
        non_valid_cols = list(set(list(df)) - set(self.valid_cols))
        all_cat_cols = get_cat_cols(df_valid)
        cat_cols_no_missing_value = [
            value
            for value in list(all_cat_cols)
            if (self.iterative_params["missing_values"] not in df_valid[value] and not df_valid[value].isna().any())
        ]
        cat_cols_missing_value_impute = [
            value for value in list(set(all_cat_cols) - set(cat_cols_no_missing_value)) if (value in self.col_impute)
        ]

        if self.enable_encoder is False:
            if len(all_cat_cols) > 0:
                raise ValueError(
                    "ERROR: Categorical data unseen at fit time and can't be included in the iterative imputation process without encoding.\n"
                    + "If you'd like to ordinal encode and impute these columns, use 'enable_encoder'=True.\n"
                    + "Note that encoded columns are not guaranteed to reverse transform if they have imputed values.\n"
                    + "If you'd like to use a different type of encoding before imputation, consider using the Pipeline class "
                    + "and call your own encoder before calling this subclass."
                )
            df_to_transf = df_valid

        else:
            self.print_message(
                "\nWARNING: Note that encoded columns are not guaranteed to reverse transform if they have imputed values.\n"
                + "If you'd like to use a different type of encoding before imputation, consider using the Pipeline class and "
                + "call your own encoder before calling this subclass.",
            )
            sys.stderr.flush()

            self.ordinal_encoder = EncoderOrdinal(df=df_valid, col_encode=all_cat_cols, unknown_value=np.nan)
            self.ordinal_encoder.fit()
            df_to_transf = self.ordinal_encoder.transform(df_valid)

        missing_value_cols = [
            value
            for value in list(df_to_transf)
            if (self.iterative_params["missing_values"] in df_to_transf[value] or df_to_transf[value].isna().any())
        ]
        missing_value_cols_no_impute = list(set(missing_value_cols) - set(self.col_impute))

        transf_df = pd.DataFrame(self.imputer.transform(df_to_transf), columns=self.valid_cols)

        # attempt inverse_transform encoded imputed columns.
        try:
            if self.enable_encoder is True:
                decoded_transf_df = self.ordinal_encoder.inverse_transform(transf_df)
                transf_df[cat_cols_missing_value_impute] = decoded_transf_df[cat_cols_missing_value_impute]
                self.print_message(f"\nImputed categorical columns' encoding was reverse transformed.")
        except IndexError:
            self.print_message(
                "\nImputed categorical columns' encoding was not reverse transformed."
                + "Note that encoded columns are not guaranteed to reverse transform if they have imputed values.\n"
            )
            pass

        # reverse cols with missing values not in col_impute to their original values.
        transf_df[missing_value_cols_no_impute] = df[missing_value_cols_no_impute]

        # inverse_transform cat cols not in col_impute.
        if self.enable_encoder is True and len(cat_cols_no_missing_value) > 0:
            decoded_df = self.ordinal_encoder.inverse_transform(df_to_transf)
            transf_df[cat_cols_no_missing_value] = decoded_df[cat_cols_no_missing_value]

        # re-add categorical cols possibly excluded at fit time with enable_encoder=False.
        for col in non_valid_cols:
            transf_df[col] = df[col]

        return transf_df
