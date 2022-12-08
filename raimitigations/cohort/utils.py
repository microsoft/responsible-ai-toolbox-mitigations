from typing import Union
import numpy as np
import pandas as pd

from ..cohort import CohortManager
from ..utils import MetricNames, get_metrics

# -----------------------------------
def fetch_cohort_results(
    X: Union[pd.DataFrame, np.ndarray],
    y_true: Union[pd.DataFrame, np.ndarray],
    y_pred: Union[pd.DataFrame, np.ndarray],
    cohort_def: Union[dict, list, str] = None,
    cohort_col: list = None,
    regression: bool = False,
    fixed_th: Union[float, dict] = None,
    shared_th: bool = False,
    return_th_dict: bool = False,
):
    """
    Computes several classification or regression metrics for a given array of predictions
    for the entire dataset and for a set of cohorts. The cohorts used to compute
    these metrics are defined by the ``cohort_def`` or ``cohort_col`` parameters
    (but not both). These parameters are the ones used in the constructor method
    of the :class:`~raimitigations.cohort.CohortManager` class. Each metric is computed
    separately for each set of predictions belonging to each of the existing cohorts.

    :param X: the dataset containing the feature columns. This dataset is used to
        filter the instances that belong to each cohort;
    :param y_true: the true labels of the instances in the ``X`` dataset;
    :param y_pred: the predicted labels;
    :param cohort_def: a list of cohort definitions, a dictionary of cohort definitions,
        or the path to a JSON file containing the definition of all cohorts. For more
        details on this parameter, please check the :class:`~raimitigations.cohort.CohortManager`
        class;
    :param cohort_col: a list of column names or indices, from which one cohort is created for each
        unique combination of values for these columns. This parameter is ignored if ``cohort_def``
        is provided;
    :param regression: if True, regression metrics are computed. If False, only classification
        metrics are computed;
    :param fixed_th: if None, the thresholds will be computed using the ROC curve. If a single float
        is provided, then this threshold is used for all cohorts (in this case, ``shared_th`` will
        be ignored). If ``fixed_th`` is a dictionary, it must have the following structure: one key
        equal to each cohort name (including one key named "all" for the entire dataset), and each
        key is associated with the threshold that should be used for that cohort. If ``fixed_th``
        is a dictionary and ``shared_th`` is True, then the only threshold used will be
        ``fixed_th["all"]``. This parameter is ignored if ``regression`` is True;
    :param shared_th: if True, the binarization of the predictions is made using the same threshold
        for all cohorts. The threshold used is the one computed for the entire dataset. If False,
        a different threshold is computed for each cohort;
    :param return_th_dict: if True, return a dictionary that maps the best threshold found for
        each cohort. This parameter is ignored if ``regression`` is True;
    :return: a dataframe containing the metrics for the entire dataset and for all the defined
        cohorts. If 'return_th_dict' is True (and 'regression' is False), return a tuple with
        two values: (``df_metrics``, ``th_dict``), where ``df_metrics`` is the metrics dataframe
        and ``th_dict`` is a dictionary that maps the best threshold found for each cohort;
    :rtype: pd.DataFrame or a tuple (pd.DataFrame, dict)
    """

    def _metric_tuple_to_dict(metric_dict, reg):
        if reg:
            metric_dict = {
                "mse": metric_dict[MetricNames.MSE_KEY],
                "rmse": metric_dict[MetricNames.RMSE_KEY],
                "mae": metric_dict[MetricNames.MAE_KEY],
                "r2": metric_dict[MetricNames.R2_KEY],
            }
        else:
            metric_dict = {
                "roc": metric_dict[MetricNames.AUC_KEY],
                "precision": metric_dict[MetricNames.PREC_KEY],
                "recall": metric_dict[MetricNames.RECALL_KEY],
                "f1": metric_dict[MetricNames.F1_KEY],
                "accuracy": metric_dict[MetricNames.ACC_KEY],
                "threshold": metric_dict[MetricNames.TH],
            }
        return metric_dict

    if cohort_def is None and cohort_col is None:
        raise ValueError("ERROR: one of the two parameters must be provided: 'cohort_col' or 'cohort_def'.")
    if cohort_def is None:
        cht_manager = CohortManager(cohort_col=cohort_col)
    else:
        cht_manager = CohortManager(cohort_def=cohort_def)

    cht_manager.fit(X, y_true)
    subsets = cht_manager.get_subsets(X, y_pred)
    y_pred_dict = {}
    cht_name_list = ["all"]
    for cht_name in subsets.keys():
        y_pred_dict[cht_name] = subsets[cht_name]["y"]
        cht_name_list.append(cht_name)

    metrics_dict = {}
    if fixed_th is None:
        th_dict = {cht_name: None for cht_name in cht_name_list}
    elif isinstance(fixed_th, (float, np.float64)):
        th_dict = {cht_name: fixed_th for cht_name in cht_name_list}
    elif isinstance(fixed_th, dict):
        th_dict = fixed_th
    else:
        raise ValueError(
            f"ERROR: 'fixed_th' must be None, a float value, or a dictionary. Instead, got {type(fixed_th)}."
        )

    results = get_metrics(y_true, y_pred, regression, best_th_auc=True, fixed_th=th_dict["all"])
    metrics_dict["all"] = _metric_tuple_to_dict(results, regression)
    metrics_dict["all"]["cht_size"] = y_true.shape[0]

    subsets = cht_manager.get_subsets(X, y_true)
    for cht_name in subsets.keys():

        if cht_name not in th_dict.keys():
            raise ValueError(
                f"ERROR: the provided 'fixed_th' parameter does not have a threshold value for the cohort {cht_name}"
            )
        th = th_dict[cht_name]
        if shared_th:
            th = th_dict["all"]

        y_subset = subsets[cht_name]["y"]
        y_pred_subset = y_pred_dict[cht_name]
        results = get_metrics(y_subset, y_pred_subset, regression, best_th_auc=True, fixed_th=th)
        metrics_dict[cht_name] = _metric_tuple_to_dict(results, regression)
        metrics_dict[cht_name]["cht_size"] = y_subset.shape[0]

    queries = cht_manager.get_queries()

    df_dict = {"cohort": [], "cht_query": []}
    for key in metrics_dict["all"].keys():
        df_dict[key] = []

    for key in metrics_dict.keys():
        df_dict["cohort"].append(key)
        if key == "all":
            df_dict["cht_query"].append("all")
        else:
            df_dict["cht_query"].append(queries[key])

        for sub_key in df_dict.keys():
            if sub_key in ["cht_query", "cohort"]:
                continue
            df_dict[sub_key].append(metrics_dict[key][sub_key])

    df = pd.DataFrame(df_dict)

    if not regression and return_th_dict:
        final_th_dict = {df_dict["cohort"][i]: df_dict["threshold"][i] for i in range(len(df_dict["cohort"]))}
        return df, final_th_dict

    return df