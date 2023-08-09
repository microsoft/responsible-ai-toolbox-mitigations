import os
from typing import Union
import pandas as pd
import numpy as np
import copy
import json

from .data_diagnostics import DataDiagnostics
from .active_detect import ActiveDetect
from .correlated_features_detect import CorrelatedFeaturesDetect
from .isolation_forest import IsolationForestDetect
from ..dataprocessing import DataProcessing
from ..dataprocessing.data_processing import DataFrameInfo
from ..utils.data_utils import get_cat_cols


class MetaDataDiagnostic(DataProcessing):
    """
    A concrete class that collects diagnostic metadata for a given dataset utilizing the DataDiagnostic classes.

    :param df: pandas dataframe or np.ndarray to predict errors in;

    :param detectors: a list of DataDiagnostic class objects to use for metadata collection;

    :param mode: a string that takes: ['column', 'row', 'full-data'], indicating over what axis to
        evaluate the data, the default value is 'column';

    :param type: a string that takes: ['aggregate', 'absolute'], indicating what type of metrics to
        output, 'aggregate' is the default value;

    :param save_json: a string pointing to a path to save a json log file of the collected metadata;

    :param label_col: a string for the label column, only used for CorrelatedFeaturesDetect, defaults to None;

    :param verbose: indicates whether internal messages should be printed or not.
    """

    # -----------------------------------
    def __init__(
        self,
        df: Union[pd.DataFrame, np.ndarray] = None,
        detectors: list[DataDiagnostics] = None,
        mode: str = "column",
        type: str = "aggregate",
        save_json: str = "./metadata.json",
        label_col: str = None,
        verbose: bool = True,
    ):
        super().__init__(verbose)
        self.df = df
        self.detectors = detectors
        self.mode = self._validate_mode(mode)
        self.type = self._validate_type(type)
        self.save_json = self._check_valid_json(save_json)
        self.label_col = label_col

    # -----------------------------------
    def _check_valid_json(self, json_path: str):
        """
        Verify that the save_json parameter is a valid json path.
        """
        if not json_path:
            raise ValueError(
                f"Please provide a json file path for the output logs."
            )  # TODO: this would only happen if they manually pass None, but can they actually do that?

        abs_path = os.path.abspath(json_path)
        dir_path = os.path.dirname(os.path.abspath(json_path))
        _, ext = os.path.splitext(abs_path)

        if not os.path.exists(dir_path) or ext.lower() != ".json":
            raise ValueError(
                f"The provided path '{json_path}' is not valid, it's not a JSON file or the directory does not exist."
            )
        else:
            return json_path

    # -----------------------------------

    def _get_fit_input_type(self) -> int:
        """
        Returns the type of data format this class uses.

        :return: 0 indicating a pandas dataframe data format;
        :rtype: int.
        """
        return self.FIT_INPUT_DF

    # -----------------------------------
    def _validate_type(self, type: str):
        if type not in ["absolute", "aggregate"]:
            raise ValueError(f"type parameter only takes ['absolute', 'aggregate'], you entered {type}.")
        return type

    # -----------------------------------
    def _validate_mode(self, mode: str):
        if mode not in ["row", "column", "full-data"]:
            raise ValueError(f"type parameter only takes ['row', 'column', 'full-data'], you entered {mode}.")
        return mode

    # -----------------------------------
    def _aggregate_results_column(self, error_matrix: np.array) -> dict:
        """
        Given an error matrix made of -1, 1 and np.nan where -1 indicates
        an error, aggregate the data over each column.
        """
        ratios = (error_matrix == -1).sum(axis=0) / (~np.isnan(error_matrix)).sum(axis=0)
        return {i: ratio for i, ratio in enumerate(ratios)}

    # -----------------------------------
    def _aggregate_results_row(self, indices: list, n_rows: int) -> int:
        """
        Given a list of erroneous row indices, find what ratio they make up
        of the data.
        """
        if len(indices) == 0:
            return 0
        return len(indices) / n_rows

    # -----------------------------------
    def get_num_categorical_features(self):
        """
        Return the number of categorical features (type=aggregate),
        otherwise return the indices of categorical cols (type=absolute)
        """
        cat_cols = get_cat_cols(self.df)
        if self.type == "aggregate":
            return {"Categorical Features": len(cat_cols)}
        else:
            return {"Categorical Features": cat_cols}

    # -----------------------------------
    def get_diagnostics(self):
        """
        Generate and save a json of diagnostic data based on the provided parameters.
        """
        output = {}
        for detector_og in self.detectors:
            detector = copy.deepcopy(detector_og)
            detector.df_info = DataFrameInfo(self.df)
            if self.mode in ["column", "full-data"]:
                # mode="column"
                # type=absolute: returns a mapping of every column index to a list of erroneous rows in that column
                # type=aggregate: return a mapping of every column index to the ratio of erroneous rows in that column
                # mode="full-data"
                # ignores type and returns the count of all erroneous values in full data
                if isinstance(detector, ActiveDetect):
                    detector.mode = "column"  # override detector mode
                    detector.fit(self.df)
                    detector.predict(self.df)
                    for module in detector.error_modules:
                        if module.module_name == "MissingValueErrorModule":
                            e_mat = detector.get_error_module_matrix("MissingValueErrorModule")
                            if self.mode == "full-data":
                                e_mapping = np.sum(e_mat == -1)
                                output["Missing Values"] = e_mapping
                                continue
                            if self.type == "absolute":
                                e_mapping = {
                                    col: np.where(e_mat[:, col] == -1)[0].tolist() for col in range(e_mat.shape[1])
                                }
                            elif self.type == "aggregate":
                                e_mapping = self._aggregate_results_column(e_mat)
                            output["Missing Values"] = e_mapping
                        elif module.module_name == "QuantitativeErrorModule":
                            e_mat = detector.get_error_module_matrix("QuantitativeErrorModule")
                            if self.mode == "full-data":
                                e_mapping = np.sum(e_mat == -1)
                                output["Quantitative Errors"] = e_mapping
                                continue
                            if self.type == "absolute":
                                e_mapping = {
                                    col: np.where(e_mat[:, col] == -1)[0].tolist() for col in range(e_mat.shape[1])
                                }
                            elif self.type == "aggregate":
                                e_mapping = self._aggregate_results_column(e_mat)
                            output["Quantitative Errors"] = e_mapping
                        elif module.module_name == "PuncErrorModule":
                            e_mat = detector.get_error_module_matrix("PuncErrorModule")
                            if self.mode == "full-data":
                                e_mapping = np.sum(e_mat == -1)
                                output["Punctuation Errors"] = e_mapping
                                continue
                            if self.type == "absolute":
                                e_mapping = {
                                    col: np.where(e_mat[:, col] == -1)[0].tolist() for col in range(e_mat.shape[1])
                                }
                            elif self.type == "aggregate":
                                e_mapping = self._aggregate_results_column(e_mat)
                            output["Punctuation Errors"] = e_mapping
                        elif module.module_name == "SemanticErrorModule":
                            e_mat = detector.get_error_module_matrix("SemanticErrorModule")
                            if self.mode == "full-data":
                                e_mapping = np.sum(e_mat == -1)
                                output["Semantic Errors"] = e_mapping
                                continue
                            if self.type == "absolute":
                                e_mapping = {
                                    col: np.where(e_mat[:, col] == -1)[0].tolist() for col in range(e_mat.shape[1])
                                }
                            elif self.type == "aggregate":
                                e_mapping = self._aggregate_results_column(e_mat)
                            output["Semantic Errors"] = e_mapping
                        elif module.module_name == "DistributionErrorModule":
                            e_mat = detector.get_error_module_matrix("DistributionErrorModule")
                            if self.mode == "full-data":
                                e_mapping = np.sum(e_mat == -1)
                                output["Distribution Errors"] = e_mapping
                                continue
                            if self.type == "absolute":
                                e_mapping = {
                                    col: np.where(e_mat[:, col] == -1)[0].tolist() for col in range(e_mat.shape[1])
                                }
                            elif self.type == "aggregate":
                                e_mapping = self._aggregate_results_column(e_mat)
                            output["Distribution Errors"] = e_mapping
                        elif module.module_name == "StringSimilarityErrorModule":
                            e_mat = detector.get_error_module_matrix("StringSimilarityErrorModule")
                            if self.mode == "full-data":
                                e_mapping = np.sum(e_mat == -1)
                                output["String Similarity Errors"] = e_mapping
                                continue
                            if self.type == "absolute":
                                e_mapping = {
                                    col: np.where(e_mat[:, col] == -1)[0].tolist() for col in range(e_mat.shape[1])
                                }
                            elif self.type == "aggregate":
                                e_mapping = self._aggregate_results_column(e_mat)
                            output["String Similarity Errors"] = e_mapping
                        elif module.module_name == "CharSimilarityErrorModule":
                            e_mat = detector.get_error_module_matrix("CharSimilarityErrorModule")
                            if self.mode == "full-data":
                                e_mapping = np.sum(e_mat == -1)
                                output["Char Similarity Errors"] = e_mapping
                                continue
                            if self.type == "absolute":
                                e_mapping = {
                                    col: np.where(e_mat[:, col] == -1)[0].tolist() for col in range(e_mat.shape[1])
                                }
                            elif self.type == "aggregate":
                                e_mapping = self._aggregate_results_column(e_mat)
                            output["Char Similarity Errors"] = e_mapping
                elif isinstance(detector, IsolationForestDetect):
                    detector.mode = "column"  # override detector mode
                    detector.fit(self.df)
                    e_mat = detector.predict(self.df)
                    if self.mode == "full-data":
                        e_mapping = np.sum(e_mat == -1)
                        output["IsolationForest Outliers"] = e_mapping
                        continue
                    if self.type == "absolute":
                        e_mapping = {col: np.where(e_mat[:, col] == -1)[0].tolist() for col in range(e_mat.shape[1])}
                    elif self.type == "aggregate":
                        e_mapping = self._aggregate_results_column(e_mat)
                    output["IsolationForest Outliers"] = e_mapping
                elif isinstance(detector, CorrelatedFeaturesDetect):
                    detector.fit(self.df, self.label_col)
                    e_mat = detector.predict(self.df)
                    if self.mode == "full-data":
                        e_mapping = np.sum(e_mat == -1)
                        output["Correlated Features"] = e_mapping
                        continue
                    if self.type == "absolute":
                        e_mapping = {col: np.where(e_mat[:, col] == -1)[0].tolist() for col in range(e_mat.shape[1])}
                    elif self.type == "aggregate":
                        e_mapping = self._aggregate_results_column(e_mat)
                    output["Correlated Features"] = e_mapping

            elif self.mode == "row":
                if isinstance(detector, ActiveDetect):
                    # type=absolute: appends a list of erroneous rows (for all columns)
                    # type=aggregate: appends ratio of erroneous rows (for all columns)
                    detector.mode = "row"  # override detector mode
                    detector.fit(self.df)
                    indices = detector.predict(self.df)
                    for module in detector.error_modules:
                        if module.module_name == "MissingValueErrorModule":
                            e_mat = detector.get_error_module_matrix("MissingValueErrorModule")
                            indices = detector._get_erroneous_row_indices(e_mat)
                            if self.type == "aggregate":
                                indices = self._aggregate_results_row(indices, e_mat.shape[0])
                            output["Missing Values"] = indices
                        elif module.module_name == "QuantitativeErrorModule":
                            e_mat = detector.get_error_module_matrix("QuantitativeErrorModule")
                            indices = detector._get_erroneous_row_indices(e_mat)
                            if self.type == "aggregate":
                                indices = self._aggregate_results_row(indices, e_mat.shape[0])
                            output["Quantitative Errors"] = indices
                        elif module.module_name == "PuncErrorModule":
                            e_mat = detector.get_error_module_matrix("PuncErrorModule")
                            indices = detector._get_erroneous_row_indices(e_mat)
                            if self.type == "aggregate":
                                indices = self._aggregate_results_row(indices, e_mat.shape[0])
                            output["Punctuation Errors"] = indices
                        elif module.module_name == "SemanticErrorModule":
                            e_mat = detector.get_error_module_matrix("SemanticErrorModule")
                            indices = detector._get_erroneous_row_indices(e_mat)
                            if self.type == "aggregate":
                                indices = self._aggregate_results_row(indices, e_mat.shape[0])
                            output["Semantic Errors"] = indices
                        elif module.module_name == "DistributionErrorModule":
                            e_mat = detector.get_error_module_matrix("DistributionErrorModule")
                            indices = detector._get_erroneous_row_indices(e_mat)
                            if self.type == "aggregate":
                                indices = self._aggregate_results_row(indices, e_mat.shape[0])
                            output["Distribution Errors"] = indices
                        elif module.module_name == "StringSimilarityErrorModule":
                            e_mat = detector.get_error_module_matrix("StringSimilarityErrorModule")
                            indices = detector._get_erroneous_row_indices(e_mat)
                            if self.type == "aggregate":
                                indices = self._aggregate_results_row(indices, e_mat.shape[0])
                            output["String Similarity Errors"] = indices
                        elif module.module_name == "CharSimilarityErrorModule":
                            e_mat = detector.get_error_module_matrix("CharSimilarityErrorModule")
                            indices = detector._get_erroneous_row_indices(e_mat)
                            if self.type == "aggregate":
                                indices = self._aggregate_results_row(indices, e_mat.shape[0])
                            output["Char Similarity Errors"] = indices

                elif isinstance(detector, IsolationForestDetect):
                    detector.mode = "row"  # override detector mode
                    detector.fit(self.df)
                    indices = detector.predict(self.df)
                    if self.type == "aggregate":
                        indices = self._aggregate_results_row(indices, self.df.shape[0])
                    output["IsolationForest Outliers"] = indices

                elif isinstance(detector, CorrelatedFeaturesDetect):
                    self.print_message("ValueError: CorrelatedFeaturesDetect is not available in row mode...skipping.")
        if self.save_json:
            fix_data = (
                lambda f: lambda d: {
                    k: (list(map(int, v)) if isinstance(v, list) else (f(f)(v) if isinstance(v, dict) else int(v)))
                    for k, v in d.items()
                }
            )(
                lambda f: lambda d: {
                    k: (list(map(int, v)) if isinstance(v, list) else (f(f)(v) if isinstance(v, dict) else int(v)))
                    for k, v in d.items()
                }
            )
            output = fix_data(output)
            with open(self.save_json, "w") as json_file:
                json.dump(output, json_file)

        return output
