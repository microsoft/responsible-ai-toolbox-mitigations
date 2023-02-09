from typing import Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
    log_loss,
    roc_curve,
    precision_recall_curve,
)

from .data_utils import err_float_01


# ----------------------------------------
class MetricNames:
    """
    Defines the keys used in the dictionary returned by the
    ``get_metric()`` function.
    """

    RMSE_KEY = "rmse"
    MSE_KEY = "mse"
    MAE_KEY = "mae"
    R2_KEY = "r2"
    ACC_KEY = "acc"
    AUC_KEY = "roc"
    PREC_KEY = "precision"
    RECALL_KEY = "recall"
    F1_KEY = "f1"
    LOG_LOSS_KEY = "log_loss"
    TH_ROC = "th_roc"
    TH_PR = "th_pr_rc"
    TH = "th"
    FINAL_PRED = "y_pred_final"
    TH_LIST = "th_tested"
    PROBLEM_TYPE = "problem_type"

    BIN_CLASS = "bin"
    MULTI_CLASS = "multi"
    REGRESSION = "reg"


# ----------------------------------------
def _pred_to_numpy(pred: Union[np.ndarray, list, pd.DataFrame]):
    """
    Converts an array to a numpy array (only if it's not a numpy
    array).

    :param pred: the array to be converted. This array could be a pandas
        dataframe, a pandas series, a list, or a numpy array.
    :return: the array provided in the parameter converted to a numpy array.
    :rtype: np.ndarray
    """
    if type(pred) in [pd.DataFrame, pd.Series]:
        pred = pred.to_numpy()
    elif type(pred) == list:
        pred = np.array(pred)
    elif type(pred) != np.ndarray:
        raise ValueError(
            "ERROR: The y_pred parameter passed to the get_metrics() "
            + "function must be a numpy array, a list, or a pandas dataframe. Instead, "
            + f"got a value from type {type(pred)}."
        )

    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, 1)

    return pred


# -----------------------------------
def _roc_evaluation(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes the AUC ROC metric for classification problems, as well as the optimal threshold
    identified using the ROC curve to be used (in case it is a binary classification problem)
    to binarize the predictions. This function works for binary and multiclass problems.

    :param y: an array with the true labels;
    :param y_pred: an arry with the predicted probabilities for each class, with shape = (N, C),
        where N is the number of rows and C is the number of classes;
    :return: a tuple (roc, th), where 'roc' is the AUC ROC metric, and 'th' is the optimal
        threshold found using the ROC curve for binary problems. In case it is a multiclass
        problem, 'th' will be None;
    :rtype: tuple
    """
    th = None
    # Binary classification
    if y_pred.shape[1] <= 2:
        if y_pred.shape[1] == 1:
            y_pred_temp = y_pred[:, 0]
        else:
            y_pred_temp = y_pred[:, 1]
        roc_auc = roc_auc_score(y, y_pred_temp, average="weighted")
        fpr, tpr, th = roc_curve(y, y_pred_temp, drop_intermediate=True)
        target = tpr - fpr
        index = np.argmax(target)
        best_th = th[index]
        # scikit-learn uses 1 th > 1.0 for technical reasons.
        # If this th is selected, move to the next th value
        if best_th > 1.0:
            best_th = th[index + 1]
    # Multi-class
    else:
        y_temp = np.squeeze(y)
        roc_auc = roc_auc_score(y_temp, y_pred, average="weighted", multi_class="ovo")
        best_th = None

    return roc_auc, best_th, th


# -----------------------------------
def _probability_to_class_binary(prediction: np.ndarray, th: float):
    """
    Converts an array with the predicted probablities for a binary classification
    problem into an array with the predicted labels (0 or 1). The 'prediction'
    parameter is expected to have the shape (N, 2), where N is the number of
    predictions and 2 is the number of classes. Only the probabilities of class
    1 is used (that is, prediction[:, 1]). If the probability of an instance
    is >= 'th', then that instance is classified as 1. Otherwise, it is classified
    as 0.

    :param prediction: an array with the probabilities for each of the 2 classes,
        with shape = (N, 2), where N is the number of rows and 2 is the number of
        classes;
    :param th: a value between [0, 1] that represents the threshold used to determine
        if a probability value is converted to class 0 or class 1;
    :return: a list with the predicted classes based on the probability values (given
        by the 'prediction' parameter) and the threshold to be used ('th' parameter).
        The list has shape = (N).
    :rtype: list
    """
    classes = []
    if prediction.shape[1] == 1:
        prediction = prediction[:, 0]
    else:
        prediction = prediction[:, 1]

    for p in prediction:
        c = 0
        if p >= th:
            c = 1
        classes.append(c)
    return classes


# -----------------------------------
def _probability_to_class_multi(prediction: np.ndarray):
    """
    Converts an array with the predicted probablities for a multiclass classification
    problem into an array with the predicted labels. The 'prediction' parameter is
    expected to have the shape (N, C), where N is the number of predictions and C is
    the number of classes. The class of each instance is chosen as being the class
    with the highest probability value among the possible classes.

    :param prediction: an array with the probabilities for each of the 2 classes,
        with shape = (N, 2), where N is the number of rows and 2 is the number of
        classes;
    :return: an array with shape = (N) with the predicted classes.
    :rtype: np.ndarray
    """
    new_pred = prediction.argmax(axis=1)
    return new_pred


# -----------------------------------
def probability_to_class(prediction: np.ndarray, th: float):
    if prediction.shape[1] > 2:
        return _probability_to_class_multi(prediction)
    return _probability_to_class_binary(prediction, th)


# -----------------------------------
def _get_precision_recall_fscore(y: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]):
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, warn_for=tuple())
    precision = np.mean(precision)
    recall = np.mean(recall)
    f1 = np.mean(f1)
    return precision, recall, f1


# -----------------------------------
def _get_precision_recall_th(y, y_pred):
    if y_pred.shape[1] > 2:
        return None, None
    if y_pred.shape[1] == 1:
        y_pred_temp = y_pred[:, 0]
    else:
        y_pred_temp = y_pred[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, y_pred_temp)

    np.seterr(invalid="ignore")

    fscore = (2 * precision * recall) / (precision + recall)
    index = np.argmax(fscore)
    best_th = thresholds[index]
    return best_th, thresholds


# ----------------------------------------
def _check_if_probability_pred(y_pred: np.ndarray):
    # Check if any of the predicted values are between 0 and 1
    y_bool = (0.0 < y_pred) & (y_pred < 1.0)
    # Check if y_pred has a shape similar to (N, 2) or (N, C),
    # where C is the number of classes
    shape_prob = len(y_pred.shape) > 1 and y_pred.shape[1] > 1
    is_prob = y_bool.all() or shape_prob
    return is_prob


# ----------------------------------------
def _get_classification_metrics(
    y: np.ndarray,
    y_pred: np.ndarray,
    best_th_auc: bool = True,
    fixed_th: float = None,
    return_th_list: bool = False,
):
    """
    Given a set of true labels (y) and predicted labels (y_pred), compute a series
    of metrics and values to measure the performance of the predictions provided
    (restricted to classification problems). The metrics computed are:

        * ROC AUC
        * Best threshold found to binarize the results. The predictions are transformed
          from probabilities to class predictions using this threshold before computing
          the precision, recall, F,1 and accuracy. This threshold is ignored if the
          predictions are for a multiclass problem. In this case, the transformation
          from probabilities to class will be made by choosing the highest probability
          for each instance
        * optimal threshold using the ROC curve (only returned if the best threshold
          is returned)
        * optimal threshold using the precision and recall curve (only returned if the
          best threshold is returned)
        * Precision
        * Recall
        * F1 score
        * Accuracy
        * Class predictions (only if the threshold was used)
        * log loss

    :param y: an array with the true labels;
    :param y_pred: an arry with the predicted probabilities for each class, with shape = (N, C),
        where N is the number of rows and C is the number of classes;
    :param best_th_auc: if True, the best threshold is computed using ROC graph. If False,
        the threshold is computed using the precision x recall graph. This parameter is ignored
        if 'fixed_th' is a value different from None;
    :param fixed_th: a value between [0, 1] that should be used as the threshold for classification
        tasks;
    :param return_th_list: returns the list of thresholds tested (be it using the ROC curve, or the
        precision and recall curve).
    :return: a dictionary with multiple classification metrics (check the description above
        for more details);
    :param rtype: dict
    """
    probability = _check_if_probability_pred(y_pred)
    problem_type = MetricNames.BIN_CLASS
    if probability:
        roc_auc, th_roc, th_list_roc = _roc_evaluation(y, y_pred)
        y_float = y.astype(np.float64)
        y_pred_float = y_pred.astype(np.float64)
        loss = log_loss(y_float, y_pred_float)
        th_pr_rc, th_list_prrc = _get_precision_recall_th(y, y_pred_float)
        th = th_roc
        th_list = th_list_roc
        if not best_th_auc:
            th = th_pr_rc
            th_list = th_list_prrc
        if fixed_th is not None:
            err_float_01(fixed_th, "fixed_th")
            th = fixed_th
        y_pred = probability_to_class(y_pred, th)

        if th is None:
            problem_type = MetricNames.MULTI_CLASS

    precision_sup, recall_sup, f1_sup = _get_precision_recall_fscore(y, y_pred)
    acc = accuracy_score(y, y_pred)

    results = {
        MetricNames.PROBLEM_TYPE: problem_type,
        MetricNames.ACC_KEY: acc,
        MetricNames.PREC_KEY: precision_sup,
        MetricNames.RECALL_KEY: recall_sup,
        MetricNames.F1_KEY: f1_sup,
    }

    if probability:
        results[MetricNames.AUC_KEY] = roc_auc
        results[MetricNames.LOG_LOSS_KEY] = loss
        results[MetricNames.TH_ROC] = th_roc
        results[MetricNames.TH_PR] = th_pr_rc
        results[MetricNames.TH] = th
        results[MetricNames.FINAL_PRED] = y_pred

    if return_th_list:
        results[MetricNames.TH_LIST] = th_list

    return results


# ----------------------------------------
def _get_regression_metrics(y: np.ndarray, y_pred: np.ndarray):
    """
    Given a set of true labels (y) and predicted labels (y_pred), compute a series
    of metrics and values to measure the performance of the predictions provided
    (restricted to regression problems). The metrics computed are:

        * MSE
        * RMSE
        * MAE
        * R2

    :param y: an array with the true labels;
    :param y_pred: an arry with the predicted values for each instance;
    :return: a dictionary with multiple regression metrics (check the description above
        for more details);
    :param rtype: dict
    """
    mse = mean_squared_error(y, y_pred, multioutput="uniform_average")
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred, multioutput="uniform_average")
    r2 = r2_score(y, y_pred, multioutput="uniform_average")

    results = {
        MetricNames.MSE_KEY: mse,
        MetricNames.RMSE_KEY: rmse,
        MetricNames.MAE_KEY: mae,
        MetricNames.R2_KEY: r2,
        MetricNames.PROBLEM_TYPE: MetricNames.REGRESSION,
    }

    return results


# ----------------------------------------
def get_metrics(
    y: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list, pd.DataFrame],
    regression: bool = False,
    best_th_auc: bool = True,
    fixed_th: float = None,
    return_th_list: bool = False,
):
    """
    Evaluates the performance of a prediction array based on its true values. This function computes
    a set of metrics and return a dictionary with the following metrics (depends if it is a regression
    or classification problem):

        * **Classification:** ROC AUC, Precision, Recall, F1, acuracy, log loss, threshold used, and
          final classes predicted (using the probabilities with the threshold);
        * **Regression:** MSE, RMSE, MAE, and R2

    :param y: an array with the true labels;
    :param y_pred: an arry with the predicted values. For classification problems, this array must contain
        the probabilities for each class, with shape = (N, C), where N is the number of rows and C is the
        number of classes;
    :param regression: if True, regression metrics are computed. If False, only classification
        metrics are computed;
    :param best_th_auc: if True, the best threshold is computed using ROC graph. If False,
        the threshold is computed using the precision x recall graph;
    :param fixed_th: a value between [0, 1] that should be used as the threshold for classification
        tasks;
    :param return_th_list: returns the list of thresholds tested (be it using the ROC curve, or the
        precision and recall curve).
    :return: a dictionary with the following metrics:

        * **Classification:** ROC AUC, Precision, Recall, F1, acuracy, log loss, threshold used, and
          final classes predicted (using the probabilities with the threshold);
        * **Regression:** MSE, RMSE, MAE, and R2

    :rtype: dict
    """
    y = _pred_to_numpy(y)
    y_pred = _pred_to_numpy(y_pred)

    if regression:
        results = _get_regression_metrics(y, y_pred)
    else:
        results = _get_classification_metrics(y, y_pred, best_th_auc, fixed_th, return_th_list)
    return results
