import pandas as pd
import numpy as np
import copy

from .data_diagnostics import DataDiagnostics
from ..dataprocessing.feat_selection import CorrelatedFeatures


class CorrelatedFeaturesDetect(DataDiagnostics):
    """
    Concrete DataDiagnostics class that utilizes the dataprocessing.CorrelatedFeatures
    class for the detection of correlated features.

    :param df: pandas dataframe or np.ndarray to predict correlation in;

    :param col_predict: a list of column names or indexes that will be subject
        to feature correlation detection. If None, a list of all columns will be used by default;

    :param correlatedfeatures_object: an raimitigations.dataprocessing.CorrelatedFeatures object to be used for correlated features detection.
        See [feat_sel_corr_tutorial.ipynb](../notebooks/dataprocessing/module_tests/feat_sel_corr_tutorial.ipynb) for more details.

    :param save_json: a string pointing to a path to save a json log file to when
        calling predict. It defaults to None, in that case, no log file is saved.

    :param verbose: boolean flag indicating whether internal messages should be printed or not.
    """

    # -----------------------------------
    def __init__(
        self,
        col_predict: list = None,
        correlatedfeatures_object: CorrelatedFeatures = CorrelatedFeatures(),  # TODO: Validate type
        save_json: str = None,
        verbose: bool = True,
    ):
        super().__init__(
            correlatedfeatures_object.df_info.df,
            col_predict,
            correlatedfeatures_object.label_col_name,
            "column",
            save_json,
            verbose,
        )
        self.correlated_features_object = copy.deepcopy(correlatedfeatures_object)
        # TODO: Basically, the label column is being removed and lost by initializing the CF obj, how do we prevent this or save this info before its lost and put the label column back, remove from here too.

    # -----------------------------------
    def _fit(self):
        """
        Fit method for this DataDiagnostics class. It fits the CorrelatedFeatures object.
        """
        self.correlated_features_object.fit(df=self.df_info.df, label_col=self.label_col)

    # -----------------------------------
    def _predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict method complement used specifically for the current class.

        :param df: the full dataset to predict feature correlation over;

        :return: an error matrix of the same shape as the input dataframe.
            - Selected columns within correlated pairs are set to 1;
            - Not selected columns within correlated pairs are set to -1;
            - Columns not in col_predict are set to np.nan.
        :rtype: np.array
        """
        # remove the label_col for correlation detection
        df = df.drop(columns=[self.label_col])

        selected_features = self.correlated_features_object.get_selected_features()

        self.valid_cols = self.df_info.columns
        self._check_predict_data_structure(df)
        error_matrix = np.full(df.shape, np.nan)  # np.empty(df.shape)

        if self.mode == "row":
            self.print_message(
                "The CorrelatedFeaturesDetect class supports only 'column' mode. Mode parameter has been set to 'column'."
            )
            self.mode = "column"

        for i, col in enumerate(list(df)):
            if col in self.col_predict:
                if col in selected_features:
                    error_matrix[:, i] = 1
                else:
                    error_matrix[:, i] = -1
            else:
                error_matrix[:, i] = np.nan

        return error_matrix

    # -----------------------------------

    def _serialize(self) -> dict:
        """
        Serializes class attributes into a dictionary for logging. #TODO: edit to include cf params or cf object
        """
        return {
            "name": "CorrelatedFeaturesDetect",
            "col_predict": self.col_predict,
            "correlatedfeatures_object": self.correlatedfeatures_object,
            "save_json": self.save_json,
            "verbose": self.verbose,
        }
