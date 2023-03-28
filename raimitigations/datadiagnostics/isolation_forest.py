from typing import Union
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.ensemble import IsolationForest

from .data_diagnostics import DataDiagnostics

class IsolationForestDetect(DataDiagnostics):
    """
    Concrete class that performs anomaly detection over data. It uses the IsolationForest 
    algorithm which isolates observations by randomly selecting a feature and then randomly 
    selecting a split value between the maximum and minimum values of the selected feature.
    This subclass uses the :class:`~sklearn.ensemble.IsolationForest` class from :mod:`sklearn` in the
    background.
    sklearn.ensemble.IsolationForest can only handle numerical data, however, this subclass allows for categorical
    input by applying ordinal encoding before calling the sklearn class. In order to use this function,
    use enable_encoder=True. If you'd like to use a different type of encoding, 
    consider using the Pipeline class and call your own encoder before calling this subclass.
    For more details see:
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#

    :param df: pandas dataframe or np.ndarray to predict errors in;

    :param col_predict: a list of column names or indexes that will be subject to error prediction. If None, a list of all columns will be used by default;

    :param mode: a string that can take the values:
        - "column", fit and prediction will be applied to each column independently. 
            An error matrix of the same shape as the data will be returned by predict.
        - "row", fits over the whole data and prediction will be applied over each row. A list of 
            erroneous row indices will be returned by predict.

    :param isf_params: a dict indicating the parameters used by
        :class:`~sklearn.ensemble.IsolationForest`. The dict has the following structure:

            | {
            |   **'n_estimators'**: 100,
            |   **'max_samples'**: "auto",
            |   **'contamination'**: "auto",
            |   **'max_features'**: 1.0,
            |   **'bootstrap'**: False,
            |   **'n_jobs'**: None,
            |   **'random_state'**: None,
            |   **'warm_start'**: False,
            | }

        where these are the parameters used by sklearn's IsolationForest. If None,
        this dict will be auto-filled as the one above.
        ``Note: max_samples can be "auto", int or float;
            contamination can be "auto" or float;
            and max_features can be int or float.``

    :param sklearn_obj: an sklearn.ensemble.IsolationForest object to use directly. If this parameter is used,
        isf_params will be overwritten.

    :param enable_encoder: a boolean flag to allow for applying ordinal encoding of categorical data before applying
        IsolationForestDetect as it only accepts numerical values.

    :param verbose: boolean flag indicating whether internal messages should be printed or not.
    """

    # -----------------------------------
    def __init__(
        self,
        df: Union[pd.DataFrame, np.ndarray] = None,
        col_predict: list = None,
        mode: str = "row",
        isf_params: dict = None,
        sklearn_obj: object = None,
        enable_encoder: bool = False,
        verbose: bool = True,
    ):
        super().__init__(df, col_predict, mode, verbose)
        self.isf_params = isf_params
        self.sklearn_obj = sklearn_obj
        self.enable_encoder = enable_encoder
        self.detectors = {}
        self._set_dict()
        if not isinstance(self.enable_encoder, bool):
            raise ValueError("ERROR: 'enable_encoder' is a boolean parameter, use True/False.")

    # -----------------------------------
    def _set_dict(self):
        """
        If the 'isf_params' dictionary that specifies how to impute given data
        is set to None, then create a default dictionary.
        """
        
        if self.sklearn_obj is not None:
            if not isinstance(self.sklearn_obj, IsolationForest):
                raise ValueError("ERROR: 'sklearn_obj' needs to be an sklearn.ensemble.IsolationForest() object.")
            sklearn_dict = self.sklearn_obj.get_params()
            self.isf_params = {
                "n_estimators": sklearn_dict["n_estimators"],
                "max_samples": sklearn_dict["max_samples"],
                "contamination": sklearn_dict["contamination"],
                "max_features": sklearn_dict["max_features"],
                "bootstrap": sklearn_dict["bootstrap"],
                "n_jobs": sklearn_dict["n_jobs"],
                "random_state": sklearn_dict["random_state"],
                "warm_start": sklearn_dict["warm_start"],
            }

        elif self.isf_params is None:
            self.isf_params = {
                "n_estimators": 100,
                "max_samples": "auto",
                "contamination": "auto",
                "max_features": 1.0,
                "bootstrap": False,
                "n_jobs": None,
                "random_state": None,
                "warm_start": False,
            }

    # -----------------------------------
    def _check_valid_dict(self):
        """
        Checks if 'isf_params' dictionary that specifics how to impute given data
        is appropriately set.
        """
        param_err = type(self.isf_params) != dict

        if param_err:
            raise ValueError("ERROR: 'isf_params' is not a dict. Check the documentation for more information.")

        keys = self.isf_params.keys()
        sklearn_params = [
            "n_estimators",
            "max_samples",
            "contamination",
            "max_features",
            "bootstrap",
            "n_jobs",
            "random_state",
            "warm_start",
        ]
        for sklearn_param in sklearn_params:
            if sklearn_param not in keys:
                raise ValueError(
                    "ERROR: expected the key "
                    + f"{sklearn_param}"
                    + " in the dictionary:\n"
                    + f"isf_params: {self.isf_params}"
                )

    # -----------------------------------
    def _fit(self):
        """
        Fit method for this DataDiagnostics class. This method: 
        (i) verifies input passed by the user, (ii) applies encoding 
        to categorical data if enable_encoder=True, excludes it 
        otherwise and (iii) checks the mode parameter, if mode = "row" 
        it creates and fits an IsolationForest object to the full dataset, 
        otherwise if mode = "column", it creates and fits the IsolationForest 
        object over each column.
        """
        self._check_valid_dict()
        df_valid = self._apply_encoding_fit()
        self.n_rows = df_valid.shape[0]

        if self.mode == "row":
            if self.sklearn_obj is None:
                self.detectors["complete"] = IsolationForest(
                    n_estimators=self.isf_params["n_estimators"],
                    max_samples=self.isf_params["max_samples"],
                    contamination=self.isf_params["contamination"],
                    max_features=self.isf_params["max_features"],
                    bootstrap=self.isf_params["bootstrap"],
                    n_jobs=self.isf_params["n_jobs"],
                    random_state=self.isf_params["random_state"],
                    warm_start=self.isf_params["warm_start"]
                )
            else:
                self.detectors["complete"] = self.sklearn_obj
            self.detectors["complete"].fit(df_valid)
                
        else:
            for col in self.valid_cols:
                if self.sklearn_obj is None:
                    self.detectors[col] = IsolationForest(
                        n_estimators=self.isf_params["n_estimators"],
                        max_samples=self.isf_params["max_samples"],
                        contamination=self.isf_params["contamination"],
                        max_features=self.isf_params["max_features"],
                        bootstrap=self.isf_params["bootstrap"],
                        n_jobs=self.isf_params["n_jobs"],
                        random_state=self.isf_params["random_state"],
                        warm_start=self.isf_params["warm_start"]
                    )
                else:
                    self.detectors[col] = deepcopy(self.sklearn_obj)
                self.detectors[col].fit(df_valid[[col]])

    # -----------------------------------
    def _predict(self, df: pd.DataFrame) -> Union[np.ndarray, list]:
        """
        Predict method complement used specifically for the current class.

        :param df: the full dataset to predict anomalies over;

        :return: if mode = "row", the predicted error matrix dataset otherwise if 
            mode = "column", a list of erroneous row indices;
        :rtype: 2-dimensional np.array or list
        """
        error_matrix = []
        self._check_predict_data_structure(df)
        df_valid = self._get_df_subset(df, self.valid_cols)
        df_to_predict = self._apply_encoding_predict(df_valid)

        if self.mode == "row":
            indicator_vector = self.detectors["complete"].predict(df_to_predict)
            indices = np.where(indicator_vector == -1)[0]
            erroneous_row_indices = df_to_predict.index[indices].tolist()
            return erroneous_row_indices
        else:
            for col in self.df_info.columns:
                if col in self.valid_cols:
                    indicator_vector = self.detectors[col].predict(df_to_predict[[col]])
                else:
                    indicator_vector = np.full(self.n_rows, np.nan)
                error_matrix.append(indicator_vector)

            return np.array(error_matrix).T
    