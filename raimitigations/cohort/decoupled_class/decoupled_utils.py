from typing import List
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, StratifiedKFold

from ...dataprocessing import DataProcessing
from ...utils import MetricNames, get_metrics

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
def get_cross_validation_results(
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
    random_state: int = None,
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
    :param regression: indicates if the estimator is a classifier or a regressor;
    :param random_state: controls the randomness of how the folds are divided.
    """
    if regression:
        k = _get_k_value(cohort_x.shape[0], min_size_fold, valid_k_folds, None)
    else:
        k = _get_k_value(cohort_x.shape[0], min_size_fold, valid_k_folds, cohort_y)
    if k is None:
        return None
    if regression:
        k_fold_obj = KFold(n_splits=k, random_state=random_state)
    else:
        k_fold_obj = StratifiedKFold(n_splits=k, random_state=random_state)
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
