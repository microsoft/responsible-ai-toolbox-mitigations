from typing import Union
import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer

from .imputer import DataImputer
from ...utils.data_utils import get_cat_cols


class KNNDataImputer(DataImputer):
    """
    Concrete class that imputes missing data of a feature using K-nearest neighbours. A feature's missing values are imputed using
    the mean value from k-nearest neighbors in the dataset. Two samples are close if the features that neither is missing are close.
    This subclass uses the :class:`~sklearn.impute.KNNImputer` class from :mod:`sklearn` in the background.
    sklearn.impute.KNNImputer can only handle numerical data, however, this subclass allows for categorical input by applying ordinal
    encoding before calling the sklearn class. In order to use this function, use enable_encoder=True. Note that encoded columns are
    not guaranteed to reverse transform if they have imputed values.
    If you'd like to use a different type of encoding before imputation, consider using the Pipeline class and call your own encoder
    before calling this subclass for imputation.
    For more details see:
    https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html#

    :param df: pandas data frame that contains the columns to be imputed;

    :param col_impute: a list of the column names or indexes that will be imputed.
        If None, this parameter will be set automatically as being a list of all
        columns with any NaN value;

    :param enable_encoder: a boolean flag to allow for applying ordinal encoding of categorical data before applying the KNNImputer since it only accepts numerical values.

    :param knn_params: a dict indicating the parameters used by
        :class:`~sklearn.impute.KNNImputer`. The dict has the following structure:

            | {
            |   **'missing_values'**:np.nan,
            |   **'n_neighbors'**:5,
            |   **'weights'**:'uniform',
            |   **'metric'**:'nan_euclidean',
            |   **'copy'**:True,
            | }

        where these are the parameters used by sklearn's KNNImputer. If None,
        this dict will be auto-filled as the one above.
        ``Note: 'weights' can take one of these values: ['uniform', 'distance'] or callable``
        ``and 'metric' can take one of these values: ['nan_euclidean'] or callable``

    :param sklearn_obj: an sklearn.impute.KNNImputer object to use directly. If this parameter is used,
        knn_params will be overwritten.

    :param verbose: indicates whether internal messages should be printed or not.
    """

    # -----------------------------------
    def __init__(
        self,
        df: Union[pd.DataFrame, np.ndarray] = None,
        col_impute: list = None,
        enable_encoder: bool = False,
        knn_params: dict = None,
        sklearn_obj: object = None,
        verbose: bool = True,
    ):
        super().__init__(df, col_impute, verbose)
        self.enable_encoder = enable_encoder
        self.knn_params = knn_params
        self.sklearn_obj = sklearn_obj
        self._set_dicts()
        if not isinstance(self.enable_encoder, bool):
            raise ValueError("ERROR: 'enable_encoder' is a boolean parameter, use True/False.")

    # -----------------------------------
    def _set_dicts(self):
        """
        If the 'knn_params' dictionary that specifies how to impute given data
        is set to None, then create a default dictionary.
        """
        if self.sklearn_obj is not None:
            if not isinstance(self.sklearn_obj, KNNImputer):
                raise ValueError("ERROR: 'sklearn_obj' needs to be an sklearn.impute.KNNImputer() object.")
            sklearn_dict = self.sklearn_obj.get_params()
            self.knn_params = {
                "missing_values": sklearn_dict["missing_values"],
                "n_neighbors": sklearn_dict["n_neighbors"],
                "weights": sklearn_dict["weights"],
                "metric": sklearn_dict["metric"],
                "copy": sklearn_dict["copy"],
            }

        elif self.knn_params is None:
            self.knn_params = {
                "missing_values": np.nan,
                "n_neighbors": 5,
                "weights": "uniform",
                "metric": "nan_euclidean",
                "copy": True,
            }

    # -----------------------------------
    def _check_valid_dict(self):
        """
        Checks if 'knn_params' dictionary that specifis how to impute given data
        is appropriately set.
        """
        param_err = type(self.knn_params) != dict

        if param_err:
            raise ValueError("ERROR: 'knn_params' is not a dict. Check the documentation for more information.")

        keys = self.knn_params.keys()
        sklearn_params = [
            "missing_values",
            "n_neighbors",
            "weights",
            "metric",
            "copy",
        ]
        for sklearn_param in sklearn_params:
            if sklearn_param not in keys:
                raise ValueError(
                    "ERROR: expected the key "
                    + f"{sklearn_param}"
                    + " in the dictionary:\n"
                    + f"knn_params: {self.knn_params}"
                )

    # -----------------------------------
    def _fit(self):
        """
        Fit method complement used specifically for the current class.
        The following steps are executed: (i) check if 'self.knn_params'
        dictionary is properly set, (ii) check for categorical columns
        (iii) exclude categorical columns if 'enable_encoder' is false,
        otherwise, apply ordinal encoder, (iv) create and fit the
        KNNImputer object over df.
        """
        self._check_valid_dict()

        df_valid = self._apply_encoding_fit()

        if self.sklearn_obj is None:
            self.imputer = KNNImputer(
                missing_values=self.knn_params["missing_values"],
                n_neighbors=self.knn_params["n_neighbors"],
                weights=self.knn_params["weights"],
                metric=self.knn_params["metric"],
                copy=self.knn_params["copy"],
            )
        else:
            self.imputer = self.sklearn_obj

        self.imputer.fit(df_valid)

    # -----------------------------------
    def _transform(self, df: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform method complement used specifically for the current class.

        :param df: the full dataset being transformed.
        :return: the transformed dataset.
        :rtype: pd.DataFrame or np.ndarray
        """
        self._reset_columns_to_impute(df)
        self._check_transf_data_structure(df)

        df_valid = self._get_df_subset(df, self.valid_cols)
        non_valid_cols = list(set(list(df)) - set(self.valid_cols))
        all_cat_cols = get_cat_cols(df_valid)
        cat_cols_no_missing_value = [
            value
            for value in list(all_cat_cols)
            if (self.knn_params["missing_values"] not in df_valid[value] and not df_valid[value].isna().any())
        ]
        cat_cols_missing_value_impute = [
            value for value in list(set(all_cat_cols) - set(cat_cols_no_missing_value)) if (value in self.col_impute)
        ]

        df_to_transf = self._apply_encoding_transf(all_cat_cols, df_valid)
        missing_value_cols = [
            value
            for value in list(df_to_transf)
            if (self.knn_params["missing_values"] in df_to_transf[value] or df_to_transf[value].isna().any())
        ]
        missing_value_cols_no_impute = list(set(missing_value_cols) - set(self.col_impute))

        transf_df = pd.DataFrame(self.imputer.transform(df_to_transf), columns=self.valid_cols)

        transf_df = self._revert_encoding(
            transf_df,
            df,
            df_to_transf,
            cat_cols_missing_value_impute,
            cat_cols_no_missing_value,
            missing_value_cols_no_impute,
            non_valid_cols,
        )

        return transf_df
