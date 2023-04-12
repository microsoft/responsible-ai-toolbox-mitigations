import os
from typing import Union
import pandas as pd
import numpy as np

from .data_diagnostics import DataDiagnostics
from ..dataprocessing import DataProcessing

class MetaDataDiagnostic(DataProcessing):
    """
    A concrete class that collects diagnostic metadata for a given dataset utilizing the DataDiagnostic classes.

    :param df: pandas dataframe or np.ndarray to predict errors in;

    :param detectors: a list of DataDiagnostic class objects to use for metadata collection;

    :param save_json: a string pointing to a path to save a json log file of the collected metadata;

    :param verbose: indicates whether internal messages should be printed or not.
    """

    # -----------------------------------
    def __init__(
        self,
        df: Union[pd.DataFrame, np.ndarray] = None,
        detectors: list[DataDiagnostics] = None,
        save_json: str = './metadata.json',
        verbose: bool = True
    ):
        super().__init__(verbose)
        self.df = df
        self.detectors = detectors
        self.save_json = self._check_valid_json(save_json)

    # -----------------------------------
    def _check_valid_json(self, json_path: str):
        """
        Verify that the save_json parameter is a valid json path.
        """
        if not json_path:
            raise ValueError(f"Please provide a json file path for the output logs.")#TODO: this would only happen if they manually pass None, but can they actually do that?
 
        abs_path = os.path.abspath(json_path)
        dir_path = os.path.dirname(os.path.abspath(json_path))
        _, ext = os.path.splitext(abs_path)

        if not os.path.exists(dir_path) or ext.lower() != '.json':
            raise ValueError(f"The provided path '{json_path}' is not valid, it's not a JSON file or the directory does not exist.")
        else:
            return json_path
