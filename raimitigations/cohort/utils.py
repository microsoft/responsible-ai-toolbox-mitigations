from typing import Union, List
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, StratifiedKFold

from ..dataprocessing import DataProcessing
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


# -----------------------------------
def _get_k_value(cohort_size: int, min_size_fold: int, valid_k_folds: List[int], cohort_y: pd.DataFrame):
    """
    Returns the first viable value for K within the possible values found in
    valid_k_folds (K here represents the number of folds used during cross-validation).
    A viable value of K means that a dataset with a size of cohort_size will be
    divided into K folds, where each fold is larger or equal to min_size_fold.

    :param cohort_size: the size of the cohort being split into folds;
    :param min_size_fold: the minimum size allowed for a fold. If
        cohort_size / K < min_size_fold, the value of K is considered invalid;
    :param valid_k_folds: a list with possible values for K. The first value of
        K found in this list (checked in reversed order) that results in valid
        folds is returned. We recommend filling this list with increasing values of K.
        This way, the largest valid value of K will be selected;
    :param cohort_y: data frame containing the labels of the cohort.
    """
    valid_cohort = False
    min_class_instances = None
    if cohort_y is not None:
        min_class_instances = min(cohort_y.value_counts().values)
    for k in reversed(valid_k_folds):
        if min_class_instances is not None and min_class_instances < k:
            continue
        fold_size = int(float(cohort_size) / float(k))
        if fold_size >= min_size_fold:
            valid_cohort = True
            break
    if not valid_cohort:
        return None
    return k


# -----------------------------------
def _get_fold_subsets(
    cohort_x: pd.DataFrame,
    cohort_y: pd.DataFrame,
    out_data_x: pd.DataFrame,
    out_data_y: pd.DataFrame,
    train_index: List[int],
    test_index: List[int],
    transform_pipe: List[DataProcessing],
    weight_out: float,
):
    """
    Returns the train and test subsets used for cross-validation. The test set is
    comprised of one fold from the cohort being analyzed. The train set is comprised
    of the remaining folds from the cohort + all of the out data provided. Also,
    return a list of weights for each training instance, where a weight of 1 is
    assigned to training instances from the cohort and a weight given by the
    weight_out parameter is assigned to the instances from the outside data.

    :param cohort_x: data frame containing the features of the cohort;
    :param cohort_x: data frame containing the labels of the cohort;
    :param out_data_x: data frame containing the features of the outside data;
    :param out_data_y: data frame containing the labels of the outside data;
    :param train_index: a list with the indices of all instances from the cohort
        data frame that should be placed inside the train set;
    :param test_index: a list with the indices of all instances from the cohort
        data frame that should be placed inside the test set;
    :param transform_pipe: a list of transformations that should be used for the
        train and test sets. Must be a list of mitigations from the dataprocessing
        package, found in the current library;
    :param weight_out: the weight assigned to the data outside the cohort (used
        for transfer learning).
    """
    cohort_train_x = cohort_x.filter(items=train_index, axis=0)
    cohort_train_y = cohort_y.filter(items=train_index, axis=0)
    train_x = pd.concat([cohort_train_x, out_data_x], axis=0)
    train_y = pd.concat([cohort_train_y, out_data_y], axis=0)
    test_x = cohort_x.filter(items=test_index, axis=0)
    test_y = cohort_y.filter(items=test_index, axis=0)

    transform_copy = [deepcopy(tf) for tf in transform_pipe]
    for tf in transform_copy:
        if tf._get_fit_input_type() == tf.FIT_INPUT_DF:
            tf.fit(train_x)
        else:
            tf.fit(train_x, train_y)
    for tf in transform_copy:
        train_x = tf.transform(train_x)
        test_x = tf.transform(test_x)

    weights = [1.0 for _ in range(cohort_train_x.shape[0])]
    weights += [weight_out for _ in range(out_data_x.shape[0])]

    return train_x, train_y, test_x, test_y, weights


# -----------------------------------
def _get_cross_validation_results(
    cohort_x: pd.DataFrame,
    cohort_y: pd.DataFrame,
    out_data_x: pd.DataFrame,
    out_data_y: pd.DataFrame,
    transform_pipe: List[DataProcessing],
    weight_out: float,
    min_size_fold: int,
    valid_k_folds: List[int],
    estimator: BaseEstimator,
    regression: bool,
):
    """
    Trains a model (estimator parameter) over different training and test
    sets, and for each split, fetch the AUC score and add it to a list.
    Returns the mean AUC score obtained for all splits. The data split
    occurs only on the cohort data:

        * split the cohort into K folds;
        * create a test set using fold 'i';
        * create a training dataset using all folds != 'i' + the outside data;
        * train the model using the train data;
        * evaluate it over the test set;
        * increment the value of 'i' and repeat until all folds have been used as a test set.

    :param cohort_x: data frame containing the features of the cohort;
    :param cohort_y: data frame containing the labels of the cohort;
    :param out_data_x: data frame containing the features of the outside data;
    :param out_data_y: data frame containing the labels of the outside data;
    :param train_index: a list with the indices of all instances from the cohort
        data frame that should be placed inside the train set;
    :param test_index: a list with the indices of all instances from the cohort
        data frame that should be placed inside the test set;
    :param transform_pipe: a list of transformations that should be used for the
        train and test sets. Must be a list of mitigations from the dataprocessing
        package, found in the current library;
    :param weight_out: the weight assigned to the data outside the cohort (used
        for transfer learning);
    :param min_size_fold: the minimum size allowed for a fold. If
        cohort_size / K < min_size_fold, the value of K is considered invalid;
    :param valid_k_folds: a list with possible values for K. The first value of
        K found in this list (checked in reversed order) that results in valid
        folds is returned. We recommend filling this list with increasing values of K.
        This way, the largest valid value of K will be selected;
    :param estimator: the estimator used for each data split. This estimator must
        accept a list of weights for each instance and must also have the
        .predict_proba() method implemented;
    :param regression: indicates if the estimator is a classifier or a regressor.
    """
    if regression:
        k = _get_k_value(cohort_x.shape[0], min_size_fold, valid_k_folds, None)
    else:
        k = _get_k_value(cohort_x.shape[0], min_size_fold, valid_k_folds, cohort_y)
    if k is None:
        return None
    if regression:
        k_fold_obj = KFold(n_splits=k)
    else:
        k_fold_obj = StratifiedKFold(n_splits=k)
    results = []
    for train_index, test_index in k_fold_obj.split(cohort_x, cohort_y):
        train_x, train_y, test_x, test_y, weights = _get_fold_subsets(
            cohort_x, cohort_y, out_data_x, out_data_y, train_index, test_index, transform_pipe, weight_out
        )
        estimator.fit(train_x, train_y, sample_weight=weights)
        if regression:
            y_pred = estimator.predict(test_x)
            result_dict = get_metrics(test_y, y_pred, regression=regression)
            results.append(result_dict[MetricNames.MSE_KEY])
        else:
            y_pred = estimator.predict_proba(test_x)
            result_dict = get_metrics(test_y, y_pred, regression=regression)
            results.append(result_dict[MetricNames.AUC_KEY])

    return np.mean(results)
