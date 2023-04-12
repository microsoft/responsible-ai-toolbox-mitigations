from typing import Union
import pandas as pd
import numpy as np

from .data_diagnostics import DataDiagnostics
from ..dataprocessing.feat_selection import CorrelatedFeatures

class CorrelatedFeaturesDetect(DataDiagnostics):
    """
    Concrete DataDiagnostics class that utilizes the dataprocessing.CorrelatedFeatures 
    class for the detection of correlated features.

    :param df: pandas dataframe or np.ndarray to predict correlation in;

    :param col_predict: a list of column names or indexes that will be subject 
        to feature correlation detection. If None, a list of all columns will be used by default;

    :param mode: a string that can take the values: #TODO: Fix this.
        - "column", fit and prediction will be applied to find correlated columns. 
            An error matrix of the same shape as the data will be returned by predict.
            One of the correlated column pairs will be assigned -1 values.
        - "row", this mode is unavailable for this class.

    :param verbose: boolean flag indicating whether internal messages should be printed or not.
    """

    # -----------------------------------
    def __init__(
        self,
        df: Union[pd.DataFrame, np.ndarray] = None,
        col_predict: list = None,
        mode: str = "column",
        save_json: str = None, #TODO: Fix, duplicate behavior of save_json_corr (Stick with correlatedFeatures logging)
        label_col: str = None,
        X: Union[pd.DataFrame, np.ndarray] = None,
        y: Union[pd.DataFrame, np.ndarray] = None,
        transform_pipe: list = None,
        method_num_num: list = ["spearman"],
        num_corr_th: float = NUM_COR_TH,
        num_pvalue_th: float = NUM_PVALUE_TH,
        method_num_cat: str = "model",
        levene_pvalue: float = PVALUE_LEVENE,
        anova_pvalue: float = PVALUE_ANOVA,
        omega_th: float = OMEGA_ANOVA,
        jensen_n_bins: int = None,
        jensen_th: float = 0.8,
        model_metrics: list = ["f1", "auc"],
        metric_th: float = METRIC_TH,
        method_cat_cat: str = "cramer",
        cat_corr_th: float = CAT_COR_TH,
        cat_pvalue_th: float = CAT_PVALUE_TH,
        tie_method: str = "missing",
        save_json_corr: bool = True,
        json_summary: str = "summary.json",
        json_corr: str = "corr_pairs.json",
        json_uncorr: str = "uncorr_pairs.json",
        compute_exact_matches: bool = True,
        verbose: bool = True,
    ):
        super().__init__(df, col_predict, mode, save_json, verbose)
        self.correlated_features_object = CorrelatedFeatures(label_col, X, y, transform_pipe,
                                    col_predict, method_num_num, num_corr_th, 
                                    num_pvalue_th, method_num_cat, levene_pvalue, 
                                    anova_pvalue, omega_th, jensen_n_bins, 
                                    jensen_th, model_metrics, metric_th, 
                                    method_cat_cat, cat_corr_th, cat_pvalue_th, 
                                    tie_method, save_json_corr, json_summary, 
                                    json_corr, json_uncorr, compute_exact_matches, 
                                    verbose)
    # -----------------------------------
    def _fit(self):
        """
        Fit method for this DataDiagnostics class. It fits the CorrelatedFeatures object.
        """
        self.correlated_features_object.fit(df=self.df, label_col=self.label_col)

    # -----------------------------------
    def _predict(self, df: pd.DataFrame) -> Union[np.ndarray, list]:
        """
        Predict method complement used specifically for the current class.

        :param df: the full dataset to predict feature correlation over;

        :return: if mode = "column", the predicted error matrix dataset otherwise if 
            mode = "column", a list of erroneous row indices;
        :rtype: 2-dimensional np.array or list
        """

