from typing import List

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.spatial import distance

from ..dataprocessing import DataProcessing
from ..dataprocessing.data_utils import err_float_01, freedman_diaconis
from ..dataprocessing.model_utils import fetch_results, probability_to_class


# -----------------------------------
class CohortHandler:
    """
    Class responsible for managing all data associated with a specific
    cohort. Also, provides the functionalities needed for training a
    model using the data from the cohort. This allows a clear separation
    between the classes that work with cohorts and the cohort
    functionalities.

    :param name: the name associated with the cohort created;

    :param label_col: the label column of the full dataset. This column is
        used to save some information about the label column that is later
        used in other funtions;

    :param regression: indicates if the cohort is used for a classification
        or a regression task.
    """

    VALID = 0
    INVALID_SIZE = 1
    INVALID_DISTRIBUTION = 2

    # -----------------------------------
    def __init__(self, name: str, label_col: pd.Series, regression: bool = False):
        self.name = name
        self.regression = regression
        self._check_valid_inputs(label_col)
        self._save_label_col_info(label_col)
        self.index_list = []
        self.out_group_index = []
        self.out_group_cohorts = []
        self.out_group_weight = 1.0
        self.conditions = []
        self.estimator = None
        self.query = None
        self.train_result = None
        self.test_result = None

    # -----------------------------------

    def _check_valid_inputs(self, label_col: pd.Series):
        """

        :param label_col: the label column of the full dataset. This column is
        used to save some information about the label column that is later
        used in other funtions;
        """
        if type(self.regression) != bool:
            raise ValueError("ERROR: the 'regression' parameter must a boolean value.")
        if type(label_col) != pd.Series:
            raise ValueError(
                "ERROR: the 'label_col' parameter must be a pd.Series containing the "
                + "full label column of the dataset."
            )

    # -----------------------------------

    def _save_label_col_info(self, label_col: pd.Series):
        """

        :param label_col: the label column of the full dataset. This column is
        used to save some information about the label column that is later
        used in other funtions;
        """
        self.label_info = {
            "optimal_bins": freedman_diaconis(label_col),
            "min": label_col.min(),
            "max": label_col.max(),
            "unique": label_col.unique(),
        }

    # -----------------------------------

    def _compute_value_counts_pct(self, df_y: pd.Series = None):
        """
        Creates a dictionary that stores the distribution of labels within
        the cohort, that is, the fraction of instances associated with each
        label (a value between [0, 1]). This is only used for classification
        tasks.

        :param df_y: data frame containing the labels of the cohort.
        """
        if df_y is not None:
            self.value_counts = df_y.value_counts()

        self.value_counts_pct = {}
        total = self.value_counts.values.sum()
        for item in self.value_counts.items():
            self.value_counts_pct[item[0]] = float(item[1]) / float(total)
        missing = [label for label in self.label_info["unique"] if label not in self.value_counts_pct.keys()]
        for label in missing:
            self.value_counts_pct[label] = 0.0

    # -----------------------------------

    def _compute_label_dist(self, df_y: pd.Series):
        """
        Saves the label column. This is later used to compare the
        distribution between two cohorts (check the function
        _jensen_distance_regression() for more details). This is only
        used for regression tasks.

        :param df_y: data frame containing the labels of the cohort.
        """
        self.label_dist = df_y.to_list()

    # -----------------------------------
    def _compute_label_statistics(self, df_y: pd.Series):
        """
        Computes and saves the statistics associated with the label column.
        These stats are then used to determine if the current cohort is
        compatible or not with another cohort. Different stats are computed
        if the current task is a classification or a regression.

        :param df_y: data frame containing the labels of the cohort.
        """
        if self.regression:
            self._compute_label_dist(df_y)
        else:
            self._compute_value_counts_pct(df_y)

    # -----------------------------------
    # CHANGE
    def set_info_df(self, df_x: pd.DataFrame, df_y: pd.Series):
        """
        Save the indices of the instances that comprise the current cohort,
        and then compute the label distribution of the cohort.

        :param df_x: data frame containing the features of the cohort;
        :param df_y: data frame containing the labels of the cohort.
        """
        new_indices = list(df_x.index)
        self.index_list = new_indices
        self._compute_label_statistics(df_y)

    # -----------------------------------
    # CHANGE: compute this internally
    def set_conditions(self, conditions: list):
        self.conditions = conditions

    # -----------------------------------
    def set_estimator(self, estimator: BaseEstimator, regression: bool):
        self.estimator = estimator
        self.regression = regression

    # -----------------------------------
    def get_size(self):
        return len(self.index_list)

    # -----------------------------------
    def get_index_list(self):
        return self.index_list

    # -----------------------------------
    def get_out_group_index_list(self):
        return self.out_group_index

    # -----------------------------------
    def get_conditions(self):
        return self.conditions

    # -----------------------------------
    def add_out_group_instances(self, outside_cohort: "CohortHandler", out_group_weight: float):
        """
        Adds training data from outside the current cohort (used for transfer learning).

        :param outside_cohort: a CohortHandler object representing the cohort containing the
            outside data. The indices from this cohort are added to the training data used
            by the current cohort;
        :param out_group_weight: the weight assigned to the outside instances (used for
            the transfer learning part).
        """
        self.out_group_weight = out_group_weight
        out_group_index = outside_cohort.get_index_list()
        self.out_group_index += out_group_index
        self.out_group_cohorts.append(outside_cohort.name)

    # -----------------------------------
    def _compute_best_tl_weight(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        transform_pipe: List[DataProcessing],
        min_size_fold: int,
        valid_k_folds: List[int],
        default_theta: float,
    ):
        """
        Compute the best value for the $\theta$ parameter, which represents the
        weight assigned to data outside the cohort when doing transfer learning.
        This method receives a list of possible $\theta$ values that should be
        tested and the full dataset (from which the current cohort was extracted
        from). For each $\theta$ value, assess the performance of the model trained
        using this value of $\theta$ using cross-validation. After cycling through
        all possible values, select the $\theta$ value associated with the best
        performance metric.

        :param X: a data frame containing the features of the full dataset;
        :param y: a data frame containing the labels of the full dataset;
        :param transform_pipe: a list of transformations that should be used for the
            train and test sets. Must be a list of mitigations from the dataprocessing
            package, found in the current library;
        :param min_size_fold: the minimum size allowed for a fold. If
            cohort_size / K < min_size_fold, the value of K is considered invalid;
        :param valid_k_folds: a list with possible values for K. The first value of
            K found in this list (checked in reversed order) that results in valid
            folds is returned. We recommend filling this list with increasing values of K.
            This way, the largest valid value of K will be selected;
        :param default_theta: a default value used when a cohort is too small to use
            cross-validation to determine the best value for $\theta$. In these cases,
            a default value is used. If default_theta is None, an error is raised when
            a cohort is too small to use cross-validation.
        """
        index_cohort = self.get_index_list()
        cohort_x = X.filter(items=index_cohort, axis=0)
        cohort_y = y.filter(items=index_cohort, axis=0)
        cohort_x.reset_index(inplace=True, drop=True)
        cohort_y.reset_index(inplace=True, drop=True)

        index_out_data = self.get_out_group_index_list()
        out_data_x = X.filter(items=index_out_data, axis=0)
        out_data_y = y.filter(items=index_out_data, axis=0)

        best_score = 0
        best_theta = -1
        for theta in self.out_group_weight:
            score = _get_cross_validation_results(
                cohort_x,
                cohort_y,
                out_data_x,
                out_data_y,
                transform_pipe,
                theta,
                min_size_fold,
                valid_k_folds,
                deepcopy(self.estimator),
                self.regression,
            )
            if score is None:
                if default_theta is None:
                    raise ValueError(
                        f"ERROR: one of the cohorts used is too small, or with a skewed label distribution, making "
                        + f"it unable to use cross validation with folds larger than {min_size_fold}. Select a "
                        + f"different cohort separation and try again, or provide a default value for theta "
                        + f"using the default_theta parameter, which is used in these situations."
                    )
                self.out_group_weight = default_theta
                return

            if best_theta < 0 or score > best_score:
                best_score = score
                best_theta = theta
        self.out_group_weight = best_theta

    # -----------------------------------
    def _get_train_cohort(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        transform_pipe: List[DataProcessing],
        min_size_fold: int,
        valid_k_folds: List[int],
        default_theta: float,
    ):
        """
        Returns a dataset with the features and a dataset with the labels of
        the cohort, and a list with the weights assigned to each instance of
        the train set (used with the sample_weight parameter of the model).
        If using transfer learning and the $\theta$ parameter provided is a
        list of possible values instead of a single value, then call the
        _compute_best_tl_weight() method to compute the optimal $\theta$ value
        among the possible values.

        :param X: a data frame containing the features of the full dataset;
        :param y: a data frame containing the labels of the full dataset;
        :param transform_pipe: a list of transformations that should be used for the
            train and test sets. Must be a list of mitigations from the dataprocessing
            package, found in the current library;
        :param min_size_fold: the minimum size allowed for a fold. If
            cohort_size / K < min_size_fold, the value of K is considered invalid;
        :param valid_k_folds: a list with possible values for K. The first value of
            K found in this list (checked in reversed order) that results in valid
            folds is returned. We recommend filling this list with increasing values of K.
            This way, the largest valid value of K will be selected;
        :param default_theta: a default value used when a cohort is too small to use
            cross-validation to determine the best value for $\theta$. In these cases,
            a default value is used. If default_theta is None, an error is raised when
            a cohort is too small to use cross-validation.
        """
        if type(self.out_group_weight) == list:
            self._compute_best_tl_weight(X, y, transform_pipe, min_size_fold, valid_k_folds, default_theta)
            # print(f"Best theta = {self.out_group_weight}")

        index = self.get_index_list() + self.get_out_group_index_list()
        subset_x = X.filter(items=index, axis=0)
        subset_y = y.filter(items=index, axis=0)
        weights = None
        if len(self.out_group_cohorts) > 0:
            weights = [1.0 for _ in range(len(self.get_index_list()))]
            weights += [self.out_group_weight for _ in range(len(self.get_out_group_index_list()))]
        return subset_x, subset_y, weights

    # -----------------------------------
    def _merge_cohorts_label_value_counts(self, cohort: "CohortHandler"):
        # merge value_counts of both cohorts
        for key in self.value_counts.keys():
            if key in cohort.value_counts.keys():
                self.value_counts[key] += cohort.value_counts[key]

        # add the keys of the value_counts variable that are in 'cohort' but not in self
        for key in cohort.value_counts.keys():
            if key not in self.value_counts.keys():
                self.value_counts[key] = cohort.value_counts[key]

        self._compute_value_counts_pct()

    # -----------------------------------
    def _merge_cohorts_label_dist(self, cohort: "CohortHandler"):
        self.label_dist += cohort.label_dist.copy()

    # -----------------------------------
    def merge_cohorts(self, cohort: "CohortHandler"):
        """
        Merges the cohort provided as a parameter with the current cohort (self).
        Merging two cohorts involves updating the index list of the cohort to
        include the index list of the new cohort, updating the set of conditions
        in a way that the new set of conditions are true for all instances of both
        cohorts, and finally, updating the label statistics of the current cohort,
        where, for classification tasks, this means updating the value_counts
        attribute to account for the label distribution of the new cohort, and
        for regression tasks means updating the distribution of values found in the
        label column.

        :param cohort: an object of the current class (ChortHandler) representing
            the cohort to be merged into the current cohort (self).
        """
        self.index_list += cohort.index_list
        for cond in cohort.conditions:
            self.conditions.append(cond)

        if self.regression:
            self._merge_cohorts_label_dist(cohort)
        else:
            self._merge_cohorts_label_value_counts(cohort)

    # -----------------------------------
    def _jensen_distance_regression(self, outside_cohort: "CohortHandler"):
        bins = self.label_info["optimal_bins"]
        hist_1, _ = np.histogram(
            self.label_dist, bins=bins, range=(self.label_info["min"], self.label_info["max"]), density=True
        )
        hist_2, _ = np.histogram(
            outside_cohort.label_dist, bins=bins, range=(self.label_info["min"], self.label_info["max"]), density=True
        )
        jensen_dist = distance.jensenshannon(hist_1, hist_2, base=2.0)
        return jensen_dist

    # -----------------------------------
    def _jensen_distance_class(self, outside_cohort: "CohortHandler"):
        p = []
        q = []
        for label in self.label_info["unique"]:
            if label not in outside_cohort.value_counts_pct.keys():
                raise ValueError(
                    "ERROR: called the is_outside_cohort_compatible() method using an outside cohort with different or missing labels."
                )
            p.append(self.value_counts_pct[label])
            q.append(outside_cohort.value_counts_pct[label])

        jensen_dist = distance.jensenshannon(p, q, base=2.0)
        return jensen_dist

    # -----------------------------------
    def is_outside_cohort_compatible(self, outside_cohort: "CohortHandler", dist_th: float):
        """
        Check if another cohort is compatible with the current cohort for the transfer
        learning task. Compatible cohorts must have a similar label distribution. The
        ​similarity of these distributions is computed using the Jensen-Shanon distance,
        which computes the distance between two distributions. This distance returns a
        value between [0, 1], where values close to 0 mean that two distributions being
        compared are similar, while values close to 1 mean that these distributions are
        considerably different. If the distance between the label distribution of both
        cohorts is smaller than a provided threshold (dist_th), then the outside cohort
        (outside_cohort) is considered compatible with the current cohort. This
        compatibility test is important to avoid using outside data during the transfer
        learning task that has a considerably different data distribution, which could
        harm the performance of the transfer learning model trained.

        :param outside_cohort: the outside cohort being tested for compatibility for
            being used as the outside data in the transfer learning task by the current
            cohort;
        :param dist_th: a value between [0, 1] that represents the threshold used to determine
            if the label distribution between two cohorts are similar or not. If the distance
            between these two distributions is <= dist_th, then the cohorts are considered
            compatible, and considered incompatible otherwise.
        """
        compatible = False

        if self.regression:
            jensen_dist = self._jensen_distance_regression(outside_cohort)
        else:
            jensen_dist = self._jensen_distance_class(outside_cohort)

        if jensen_dist <= dist_th:
            compatible = True

        return compatible

    # -----------------------------------
    def check_min_class_rate(self, min_rate: float):
        """
        Returns true if all label values ​have an ​occurrence rate greater
        than min_rate. If one of the label values has an ​occurrence rate
        smaller than min_rate, return False. This is used to determine
        if the minority class has a rate greater than the minimum rate
        allowed.

        :param min_rate: a value between [0, 1] that determines the
            minimum ​occurrence rate allowed for any class in the label
            column.
        """
        if self.regression:
            return True
        err_float_01(min_rate, "min_rate")
        for rate in self.value_counts_pct.values():
            if rate < min_rate:
                return False
        return True

    # -----------------------------------
    def is_valid(self, min_size: int, min_rate: float):
        """
        Check if the current cohort (self) is valid. A valid cohort
        is larger than the minimum size allowed (min_size) and has
        an ​occurrence rate for all classes in the label column larger
        than the minimum ​occurrence rate allowed (min_rate).

        :param min_size: the minimum number of instances that a cohort
            is allowed to have in order to it be considered valid;
        :param min_rate: the minimum ​occurrence rate for the minority
            class (from the label column) that a cohort is allowed
            to have. If the minority class of the cohort has an ​occurrence
            rate lower than min_rate, the cohort is considered invalid.
        """
        if not self.check_min_class_rate(min_rate):
            return self.INVALID_DISTRIBUTION
        if self.get_size() < min_size:
            return self.INVALID_SIZE
        return self.VALID

    # -----------------------------------
    def _get_best_prob_th(self, X: pd.DataFrame, y: pd.DataFrame, best_th_auc: bool = True, train_set: bool = True):
        """
        Computes the best probability ​threshold used for the binary label (doesn't do anything
        for regression tasks). The best threshold is computed using either the precision x recall
        curve or the AUC curve.

        :param X: the features of the instances used to get the best threshold;
        :param y: the labels of the instances used to get the best threshold;
        :param best_th_auc: if True, use the AUC curve to get the best threshold. If False, use
            the precision x recall curve to get the best threshold;
        :param train_set: indicates if the X and y provided ​represent the train set or a test set.
        """
        if self.regression:
            return
        pred = self.estimator.predict_proba(X)
        roc, best_th, best_pr, best_rc, best_f1, best_acc, best_y_pred = fetch_results(y, pred, best_th_auc)
        result = {
            "roc": roc,
            "th": best_th,
            "pr": best_pr.mean(),
            "rc": best_rc.mean(),
            "f1": best_f1.mean(),
            "accuracy": best_acc,
        }
        if train_set:
            self.train_result = result
        else:
            self.test_result = result

    # -----------------------------------
    def get_cohort_and_fit_estimator(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        transform_pipe: List[DataProcessing],
        min_size_fold: int,
        valid_k_folds: List[int],
        default_theta: float,
        best_th_auc: bool = False,
    ):
        """
        Runs the steps to train a model for a given cohort and get the best probability
        threshold using the training set. First, fetch the instances used for training
        the cohort (_get_train_cohort) and their respective weights. Second, fit the
        estimator using the data fetched. Finally, compute the best threshold using the
        _get_best_prob_th method (only for classification models).

        :param X: a data frame containing the features of the full dataset;
        :param y: a data frame containing the labels of the full dataset;
        :param transform_pipe: a list of transformations that should be used for the
            train and test sets. Must be a list of mitigations from the dataprocessing
            package, found in the current library;
        :param min_size_fold: the minimum size allowed for a fold. If
            cohort_size / K < min_size_fold, the value of K is considered invalid;
        :param valid_k_folds: a list with possible values for K (used only to get the
            best value for $\theta$ when using transfer learning). The first value of
            K found in this list (checked in reversed order) that results in valid folds
            is returned. We recommend filling this list with increasing values of K.
            This way, the largest valid value of K will be selected;
        :param default_theta: a default value used when a cohort is too small to use
            cross-validation to determine the best value for $\theta$. In these cases,
            a default value is used. If default_theta is None, an error is raised when
            a cohort is too small to use cross-validation.
        :param best_th_auc: if True, use the AUC curve to get the best threshold. If False, use
            the precision x recall curve to get the best threshold;
        """
        subset_x, subset_y, weights = self._get_train_cohort(
            X, y, transform_pipe, min_size_fold, valid_k_folds, default_theta
        )

        for tf in transform_pipe:
            subset_x = tf.transform(subset_x)
        if weights is None:
            self.estimator.fit(subset_x, subset_y)
        else:
            try:
                self.estimator.fit(subset_x, subset_y, sample_weight=weights)
            except Exception as err_msg:
                raise ValueError(
                    (
                        "ERROR: an error occured while fitting the estimator. The "
                        f"following error message was returned: {err_msg}"
                    )
                )
        self._get_best_prob_th(subset_x, subset_y, best_th_auc)

    # -----------------------------------
    def get_cohort_subset(self, df: pd.DataFrame, index_used: list = None):
        """
        Filters a dataset to fetch only the instances that follow the
        conditions of the current cohort. Use the _set_cohort_query()
        to build the query that filters the dataset, and then apply
        this query. If the current cohort doesn't have any conditions,
        this means that this cohort is comprised of all instances that
        don't belong to any other cohort (condition = None in the
        cohort_dict parameter of the DecoupledClass class). In this
        case, the cohort subset will be all instances whose index does
        not belong to any other cohort. The list of indices used in
        other cohorts is given by the index_used parameter. Finally,
        return the filtered dataset.

        :param df: a data frame containing the features of a dataset
            that should be filtered;
        :param index_used: a list of all indices of the dataset df that
            already belongs to some other cohort.
        """
        if self.conditions == []:
            missing_index = [index for index in df.index if index not in index_used]
            subset = df.filter(items=missing_index, axis=0)
        else:
            self._set_cohort_query()
            subset = df.query(self.query)
        return subset

    # -----------------------------------
    def find_instances_cohort_and_predict(
        self, df: pd.DataFrame, transform_pipe: List[DataProcessing], index_used: list = None, probability: bool = False
    ):
        """
        Implements the main prediction flow for a given cohort. Given a dataset, this
        method returns a list of ​probabilities for each class for each instance or a
        list of class predictions for each instance. The following steps are executed:
        (i) filter the dataset so that the subset contains only the instances that
        follows the cohort's conditions, (ii) apply the transformations from the
        trainsform_list over the filtered subset, (iii) compute the probabilities
        of each instance belonging to each class using the cohort's trained estimator,
        (iv) transform each probability to a class value using the probability threshold
        computed using the _get_best_prob_th() method (but only if the 'probability'
        parameter is True and if the current problem is a classification task). Return
        the list of classes, probabilities, or predictions, along with the list of indices
        assigned to these predictions.

        :param df: a data frame containing the features of a given dataset;
        :param transform_pipe: a list of transformations that should be used for the
            train and test sets. Must be a list of mitigations from the dataprocessing
            package, found in the current library;
        :param index_used: a list of all indices of the dataset df that
            already belongs to some other cohort;
        :param probability: if True, return the probability values (predict_proba). If
            False, transform the probabilities to class values using the threshold
            computed by the _get_best_prob_th() method. This parameter is ignored if
            the current task is a regression task.
        """
        subset = self.get_cohort_subset(df, index_used)
        if subset.empty:
            return [], []
        for tf in transform_pipe:
            subset = tf.transform(subset)
        if self.regression:
            pred = self.estimator.predict(subset)
        else:
            pred = self.estimator.predict_proba(subset)
            if not self.regression and not probability:
                pred_bin = probability_to_class(pred, self.train_result["th"])
                pred = pred_bin
        return list(subset.index), pred

    # -----------------------------------
    def print(self):
        """
        Prints all the relevant information of the current cohort.
        """
        print(f"{self.name}:")
        print(f"\tSize: {self.get_size()}")

        print(f"\tConditions:")
        if self.conditions == []:
            print(f"\t\tRemaining instances")
        else:
            for cond in self.conditions:
                print(f"\t\t{cond}")

        if not self.regression:
            print(f"\tValue Counts:")
            for key in self.value_counts.keys():
                print(f"\t\t{key}: {self.value_counts[key]} ({self.value_counts_pct[key]*100:.2f}%)")

        if self.get_out_group_index_list() == []:
            print("\tInvalid: False")
        else:
            print("\tInvalid: True")
            print(f"\t\tCohorts used as outside data: {self.out_group_cohorts}")
            print(f"\t\tTheta = {self.out_group_weight}")