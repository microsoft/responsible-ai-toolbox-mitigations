from typing import Union
import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer  # noqa # pylint: disable=unused-import
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from .imputer import DataImputer


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

    :param enable_encoder: a boolean flag to allow for applying ordinal encoding of categorical data before applying the
        IterativeImputer since it only accepts numerical values.

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

    :param sklearn_obj: an sklearn.impute.IterativeImputer object to use directly. If this parameter is used,
        iterative_params will be overwritten.

    :param verbose: indicates whether internal messages should be printed or not.
    """

    # -----------------------------------
    def __init__(
        self,
        df: Union[pd.DataFrame, np.ndarray] = None,
        col_impute: list = None,
        enable_encoder: bool = False,
        iterative_params: dict = None,
        sklearn_obj: object = None,
        verbose: bool = True,
    ):
        super().__init__(df, col_impute, verbose)
        self.enable_encoder = enable_encoder
        self.iterative_params = iterative_params
        self.sklearn_obj = sklearn_obj
        self._set_dicts()
        if not isinstance(self.enable_encoder, bool):
            raise ValueError("ERROR: 'enable_encoder' is a boolean parameter, use True/False.")

    # -----------------------------------
    def _set_dicts(self):
        """
        If the 'iterative_params' dictionary that specifies how to impute given data
        is set to None, then create a default dictionary.
        """
        if self.sklearn_obj is not None:
            if not isinstance(self.sklearn_obj, IterativeImputer):
                raise ValueError("ERROR: 'sklearn_obj' needs to be an sklearn.impute.IterativeImputer() object.")
            sklearn_dict = self.sklearn_obj.get_params()
            self.iterative_params = {
                "estimator": sklearn_dict["estimator"],
                "missing_values": sklearn_dict["missing_values"],
                "sample_posterior": sklearn_dict["sample_posterior"],
                "max_iter": sklearn_dict["max_iter"],
                "tol": sklearn_dict["tol"],
                "n_nearest_features": sklearn_dict["n_nearest_features"],
                "initial_strategy": sklearn_dict["initial_strategy"],
                "imputation_order": sklearn_dict["imputation_order"],
                "skip_complete": sklearn_dict["skip_complete"],
                "min_value": sklearn_dict["min_value"],
                "max_value": sklearn_dict["max_value"],
                "random_state": sklearn_dict["random_state"],
            }

        elif self.iterative_params is None:
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
        Checks if 'iterative_params' dictionary that specifics how to impute given data
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

        df_valid = self._apply_encoding_fit()

        if self.sklearn_obj is None:
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
        df_to_transf, missing_value_cols_no_impute = self._pre_process_transform(
            df, self.iterative_params["missing_values"]
        )

        transf_df = pd.DataFrame(self.imputer.transform(df_to_transf), columns=self.valid_cols)

        transf_df = self._post_process_transform(transf_df, df, missing_value_cols_no_impute)

        return transf_df
