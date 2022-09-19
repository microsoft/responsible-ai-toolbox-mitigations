from typing import List
from copy import deepcopy
from typing import Union
import inspect

import numpy as np
import pandas as pd
import itertools
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..estimator import Estimator
from .cohorts import CohortHandler
from ...data_utils import err_float_01


"""
TODO:

    GENERAL:
        - This class currently only works for binary classification problems. In the
        future, expand it to work with multiclass problems as well (not multilabel, at
        least not for now);

    TRANSFER LEARNING:
        - Create an option for saving information of all cohorts used in a JSON file + save
        the cohorts in separate csv files.
"""


class DecoupledClass(Estimator):
    """
    Concrete class that trains different models over different subsets of data (cohorts).
    This is useful when a given cohort behaves differently from the rest of the dataset,
    or when a cohort represents a minority group that is underrepresented. For small cohorts,
    it is possible to train a model using the data of other cohorts (outside data) with a
    smaller weight $\theta$ (only works with models that accept instance weights). This process
    is called here Transfer Learning. Instead of using transfer learning, it is also possible
    to merge small cohorts together, resulting in a set of sufficiently large cohorts.

    :param df: the data frame to be used during the fit method.
        This data frame must contain all the features, including the label
        column (specified in the 'label_col' parameter). This parameter is
        mandatory if 'label_col' is also provided. The user can also provide
        this dataset (along with the 'label_col') when calling the fit()
        method. If df is provided during the class instantiation, it is not
        necessary to provide it again when calling fit(). It is also possible
        to use the 'X' and 'y' instead of 'df' and 'label_col', although it is
        mandatory to pass the pair of parameters (X,y) or (df, label_col) either
        during the class instantiation or during the fit() method;

    :param label_col: the name or index of the label column. This parameter is
        mandatory if 'df' is provided;

    :param X: contains only the features of the original dataset, that is, does not
        contain the label column. This is useful if the user has already separated
        the features from the label column prior to calling this class. This parameter
        is mandatory if 'y' is provided;

    :param y: contains only the label column of the original dataset.
        This parameter is mandatory if 'X' is provided;

    :param transform_pipe: a list of transformations to be used as a pre-processing
        pipeline. Each transformation in this list must be a valid subclass of the
        current library (EncoderOrdinal, BasicImputer, etc.). Some feature selection
        methods require a dataset with no categorical features or with no missing
        values (depending on the approach). If no transformations are provided, a set
        of default transformations will be used, which depends on the feature selection
        approach (subclass dependent);

    :param regression: if True and no estimator is provided, then create a default
        regression model. If False, a classifier is created instead. This parameter
        is ignored if an estimator is provided using the 'estimator' parameter;

    :param cohort_dict: a dictionary that defines each cohort. This dictionary must have
        one key for each cohort, where the key represents the cohort's name. Each of these
        keys is assigned to a list of secondary dictionaries, where each of these secondary
        dictionaries represents a set of conditions. This set of conditions defines which
        instances will belong to each cohort. The last cohort is allowed to be assigned
        a None value, which means that all instances not yet assigned to any of the previous
        cohorts will be assigned to the last cohort. Below is an example of a valid
        cohort_dict:
            * cohort_dict = {
                            "cohort_1": [{'column1': 'c1value1', 'column2': 'c2value5'}],
                            "cohort_2": [{'column1': 'c1value3', 'column2': 'c2value4'},
                                         {'column1': ['c1value2', 'c1value4']}],
                            "cohort_3": None
                        }

    :param cohort_cols: a list of column names that indicates which columns should be used
        to create a cohort. For example, if cohort_cols = ["C1", "C2"], then we first identify
        all possible values for column "C1" and "C2". Suppose that the unique values in "C1"
        are: [0, 1, 2], and the unique values in "C2" are: ['a', 'b']. Then, we create one
        cohort for each combination between these two sets of unique values. This way, the
        first cohort will be conditioned to instances where ("C1" == 0 and "C2" == 'a'),
        cohort 2 will be conditioned to ("C1" == 0 and "C2" == 'b'), and so on. There are
        called the baseline cohorts. We then check if there are any of the baseline cohorts
        that are invalid, where an invalid cohort is considered as being a cohort with
        size < max(min_cohort_size, df.shape[0] * min_cohort_pct) or a cohort with a
        minority class (the label value with least ocurrences) with an occurrence rate <
        minority_min_rate. Every time an invalid cohort is found, we merge this cohort to
        the current smallest cohort. This is simply a heuristic, as identifying the best
        way to merge these cohorts in a way that results in a list of valid cohorts is a
        complex problem that we do not try to solve here;

    :param theta: the $\theta$ parameter is used in the transfer learning step of the
        decoupled classifier, and it represents the weight assigned to the instances from
        the outside data (data not from the current cohort) when fitting an estimator for
        a given cohort. The $\theta$ parameter must be a value between [0, 1], or a list
        of floats, or a boolean value (more information on each type ahead). This
        parameter is associated to how the $\theta$ is set. When a cohort with a size
        smaller than the minimum size allowed is found, transfer learning is used to fix
        this issue. Here, transfer learning occurs when a set of data not belonging to a
        given cohort is used when fitting that cohort's estimator, but the instances from
        the outside data are assigned a smaller weight equal to $\theta$. This weight can
        be fixed for all cohorts (when theta is a simple float value) or it can be identified
        automatically for each cohort separately (only for those cohorts that require
        transfer learning)(when theta is a list of floats or a boolean value). The theta
        parameter can be a float, a list of floats, or a boolean value. Each of the possible
        values is explained as follows:
            * float: the exact value assigned to the $\theta$ parameter for all cohorts.
                Must be a value between [0, 1];
            * list of float: a list of possible values for $\theta$. All values in this
                list must be values between [0, 1]. When a cohort uses transfer learning,
                Cross-Validation is used with the cohort data plus the outside data using
                different values of $\theta$ (the values within the list of floats), and
                the final $\theta$ is selected as being the one associated with the highest
                performance in the Cross-Validation process. The Cross-Validation (CV) here
                splits the cohort data into K folds (the best K value is identified
                according to the possible values in valid_k_folds_theta), and then proceeds
                to use one of the folds as the test set, and the remaining folds plus the
                outside data as the train set. A model is fitted for the train set and then
                evaluated in the test set. The ROC AUC metric is obtained for each CV run
                until all folds have been used as a test set. We then compute the average
                ROC AUC score for the K runs and that gives the CV score for a given $\theta$
                value. This is repeated for all possible $\theta$ values (the theta list),
                and the $\theta$ with the best score is selected for that cohort. This
                process is repeated for each cohort that requires transfer learning;
            * boolean: similar to when theta is a list of floats, but here, instead of
                receiving the list of possible $\theta$ from the user, a default list of
                possible values is used (self.THETA_VALUES). If True, uses transfer
                learning over small cohorts, and for each small cohort, select the best
                $\theta$ among the values in THETA_VALUES. If False, don't use transfer
                learning;

    :param default_theta: the default value for $\theta$ when a given cohort is too small
        to use Cross-Validation to find the best $\theta$ value among a list of possible
        values. This parameter is only used when the 'theta' parameter is True or a list
        of float values. When splitting a cohort into K folds, each fold must have a
        minimum size according to the min_fold_size_theta parameter. When that is not
        possible, we reduce the value of K (according to the possible values of K specified
        in the valid_k_folds_theta parameter) and test if now we can split the cohort into
        K folds, each fold being larger than min_fold_size_theta. We do this until all K
        values are tested, and if none of these results in large enough folds, a default
        value of $\theta$ is used to avoid raising an error. This default value is given
        by this parameter. If None, then don't use any default value in these cases. Instead,
        raise an error;

    :param cohort_dist_th: a value between [0, 1] that represents the threshold used to
            determine if the label distribution of two cohorts are similar or not. If the
            distance between these two distributions is <= cohort_dist_th, then the cohorts
            are considered compatible, and considered incompatible otherwise. This is used
            to determine how to build the outside data used for transfer learning: when
            a cohort uses transfer learning (check the parameter 'theta' for more information
            on that process), the outside data used for it must be comprised of data from
            other cohorts that follow a somehow similar label distribution. Otherwise, the
            use of outside data could be more harmful than useful. Therefore, for each cohort
            using transfer learing, we check which other cohorts have a similar label
            distribution. The similarity of these distributions is computed using the
            Jensen-Shanon distance, which computes the distance between two distributions.
            This distance returns a value between [0, 1], where values close to 0 mean that
            two distributions being compared are similar, while values close to 1 mean that
            these distributions are considerably different. If the distance between the label
            distribution of both cohorts is smaller than a provided threshold (cohort_dist_th),
            then the outside cohort is considered compatible with the current cohort;

    :param min_fold_size_theta: the minimum size allowed for each fold when doing
        Cross-Validation to determine the best value for $\theta$. For more information, check
        the parameter default_theta;

    :param valid_k_folds_theta: a list of possible values for K, which represents the number
        of splits used over a cohort data when doing Cross-Validation (to determine the best
        $\theta$ value). The last value in this list is used, and if this K value results in
        invalid folds, the second-to-last value in the list is. This process goes on until
        a valid K value is found in the list. We recommend filling this list with increasing
        values of K. This way, the largest valid value of K will be selected; For more
        information, check the parameter default_theta;

    :param estimator: the estimator used for each cohort. Each cohort will have their own
        copy of this estimator, which means that different instances of the same estimator
        is used for each cohort;

    :param min_cohort_size: the minimum size a cohort is allowed to have to be considered
        valid. Check the cohort_cols parameter for more information;

    :param min_cohort_pct: a value between [0, 1] that determines the minimum size allowed
        for a cohort. The minimum size is given by the size of the full dataset (df.shape[0])
        multiplied by min_cohort_pct. The maximum value between min_cohort_size and
        (df.shape[0] * min_cohort_pct) is used to determine the minimum size allowed for a
        cohort. Check the cohort_cols parameter for more information;

    :param minority_min_rate: the minimum occurrence rate for the minority class (from the label
        column) that a cohort is allowed to have. If the minority class of the cohort has an
        occurrence rate lower than min_rate, the cohort is considered invalid. Check the
        cohort_cols parameter for more information;

    :param inplace: indicates if the original dataset will be saved internally
        (df_org) or not. If True, then the feature selection transformation is saved
        over the original dataset. If False, the original dataset is saved separately
        (default value);

    :param verbose: indicates whether internal messages should be printed or not.
    """

    MIN_COHORT_SIZE_PCT = 0.1
    MIN_COHORT_SIZE = 50
    MINORITY_MIN_RATE = 0.1

    VALID_NAME = "valid"
    INVALID_SIZE_NAME = "invalid_size"
    INVALID_DIST_NAME = "invalid_dist"

    THETA_VALUES = [0.2, 0.4, 0.6, 0.8]
    MIN_FOLD_SIZE_THETA = 20
    VALID_K_FOLDS = [3, 4, 5]

    BASE_CLASSIFIER = DecisionTreeClassifier(max_features="sqrt")
    BASE_REGRESSOR = DecisionTreeRegressor()

    # -----------------------------------
    def __init__(
        self,
        df: pd.DataFrame = None,
        label_col: str = None,
        X: pd.DataFrame = None,
        y: pd.DataFrame = None,
        transform_pipe: list = None,
        regression: bool = None,
        cohort_dict: dict = None,
        cohort_cols: list = None,
        theta: Union[float, List[float], bool] = False,
        default_theta: float = None,
        cohort_dist_th: float = 0.8,
        min_fold_size_theta: int = MIN_FOLD_SIZE_THETA,
        valid_k_folds_theta: List[int] = VALID_K_FOLDS,
        estimator: BaseEstimator = None,
        min_cohort_size: int = MIN_COHORT_SIZE,
        min_cohort_pct: float = MIN_COHORT_SIZE_PCT,
        minority_min_rate: float = MINORITY_MIN_RATE,
        inplace: bool = False,
        verbose: bool = True,
    ):
        self.cohort_list = None
        super().__init__(df, label_col, X, y, estimator, transform_pipe, regression, inplace, verbose)
        self._set_cohorts(cohort_dict, cohort_cols)
        self._set_theta_param(theta, min_fold_size_theta, valid_k_folds_theta, default_theta, cohort_dist_th)
        self._set_cohorts_min_size(min_cohort_size, min_cohort_pct, minority_min_rate)
        self._get_cohorts()

    # -----------------------------------
    def _get_base_estimator(self):
        """
        Returns the default estimator that should be used for each cohort.
        For regression tasks, BASE_REGRESSOR (internal variable) is returned.
        For classification tasks, BASE_CLASSIFIER. These base estimators are
        only used if the user doesn't provide any estimator through the estimator
        parameter.
        """
        if self.regression:
            return self.BASE_REGRESSOR
        return self.BASE_CLASSIFIER

    # -----------------------------------
    def _set_cohorts(self, cohort_dict: dict, cohort_cols: list):
        """
        Sets the attributes cohort_dict and cohort_cols. Also
        checks for any errors with the parameters provided.

        :param cohort_dict: the cohort_dict parameter provided
            to the constructor of this class;
        :param cohort_cols: the cohort_cols parameter provided
            to the constructor of this class;
        """
        if cohort_dict is None and cohort_cols is None:
            raise ValueError(
                "ERROR: at least one of the following parameters must be provided: cohorts or cohort_cols."
            )

        self.cohort_cols = None
        self.cohort_dict = None

        if cohort_cols is not None:
            if type(cohort_cols) != list or cohort_cols == []:
                raise ValueError(
                    "ERROR: cohort_cols must be a list of column names that indicates "
                    + "the columns used to build the cohorts."
                )
            self.cohort_cols = cohort_cols

        if cohort_dict is not None:
            cohort_err_msg = (
                "ERROR: cohort_dict must be a dictionary that defines each cohort. This dictionary must have "
                "one key for each cohort, where the key represents the cohort's name. Each of these keys "
                "is assigned to a list of ​secondary dictionaries, where each of these ​secondary  dictionaries "
                "represents a set of conditions. Check the documentation for more details."
            )
            if type(cohort_dict) != dict:
                raise ValueError(cohort_err_msg)
            if len(cohort_dict.keys()) < 2:
                raise ValueError("ERROR: provide at least two cohorts to the cohort_dict parameter.")
            for (name, conditions) in cohort_dict.items():
                if conditions is None:
                    continue
                if type(conditions) != list:
                    raise ValueError(cohort_err_msg)
                for cond in conditions:
                    if type(cond) != dict:
                        raise ValueError(cohort_err_msg)
            self.cohort_dict = cohort_dict

    # -----------------------------------
    def _set_cohorts_min_size(self, min_cohort_size: int, min_cohort_pct: float, minority_min_rate: float):
        """
        Sets the attributes min_cohort_size, min_cohort_pct, and minority_min_rate.
        Also, check for any errors with the parameters provided.

        :param min_cohort_size: the min_cohort_size parameter provided to the
            constructor of this class;
        :param min_cohort_pct: the min_cohort_pct parameter provided to the
            constructor of this class;
        :param minority_min_rate: the minority_min_rate parameter provided to the
            constructor of this class;
        """
        err_float_01(min_cohort_pct, "min_cohort_pct")
        err_float_01(minority_min_rate, "minority_min_rate")
        if type(min_cohort_size) != int:
            raise ValueError(
                "ERROR: the 'min_cohort_size' parameter must be an integer "
                + "representing the minimum size of each cohort."
            )
        self.min_cohort_size = min_cohort_size
        self.min_cohort_pct = min_cohort_pct
        self.minority_min_rate = minority_min_rate

    # -----------------------------------
    def _set_theta_param(
        self,
        theta: Union[float, List[float], bool],
        min_size_fold: int,
        valid_k_folds: List[int],
        default_theta: float,
        cohort_dist_th: float,
    ):
        """
        Sets the attributes associated with the theta parameter. Also, check for any
        errors with the parameters provided.

        :param theta: the theta parameter provided to the constructor of this class;
        :param min_size_fold: the min_fold_size_theta parameter provided to the
            constructor of this class;
        :param valid_k_folds: the valid_k_folds_theta parameter provided to the
            constructor of this class;
        :param default_theta: the default_theta parameter provided to the constructor
            of this class;
        :param cohort_dist_th: the cohort_dist_th parameter provided to the constructor
            of this class;
        """
        # check for errors and set the theta parameter
        self.theta = theta
        if type(theta) == bool:
            if theta:
                self.theta = self.THETA_VALUES
        elif type(theta) == float:
            err_float_01(theta, "theta")
        elif type(theta) == list:
            for i in range(len(theta)):
                err_float_01(theta[i], f"theta[{i}]")
        else:
            raise ValueError(
                "ERROR: Invalid value for the 'theta' parameter. Theta must be a float value between [0, 1], "
                + "a list of possible values (the best value being selected using cross-validation), of a boolean "
                + "value that indicates if transfer learning should be used or not (when True, a default list of "
                + "values is used)."
            )

        # check for errors and set the min_size_fold parameter
        if type(min_size_fold) != int or min_size_fold <= 0:
            raise ValueError(
                "ERROR: Invalid parameter 'min_fold_size_theta'. This parameter must be an integer value greater than 0."
            )
        self.min_size_fold = min_size_fold

        # check for errors and set the valid_k_folds parameter
        error = False
        if type(valid_k_folds) != list or valid_k_folds == []:
            error = True
        else:
            for k in valid_k_folds:
                if type(k) != int or k <= 1:
                    error = True
                    break
        if error:
            raise ValueError(
                "ERROR: Invalid parameter 'valid_k_folds_theta'. This parameter must be a list of valid values for k, "
                + "where k represents the number of folds in a cross validation scenario. Each k in this list must be a "
                + "value greater than 1."
            )
        self.valid_k_folds = valid_k_folds

        # check for errors and set the default_theta parameter
        if default_theta is not None:
            err_float_01(default_theta, "default_theta")
        self.default_theta = default_theta

        # check for errors and set the cohort_dist_th parameter
        err_float_01(cohort_dist_th, "cohort_dist_th")
        self.cohort_dist_th = cohort_dist_th

        # check if the estimator accepts the 'sample_weight' parameter to its fit() method
        if self.estimator is not None:
            signature = inspect.signature(self.estimator.fit)
            has_sample_weight = False
            for param in signature.parameters.values():
                if param.name == "sample_weight":
                    has_sample_weight = True
            if not has_sample_weight:
                raise ValueError(
                    (
                        "ERROR: invalid estimator provided. When using transfer learning, only "
                        "estimators capable of working with sample weights are valid. The fit "
                        "method of the estimator must have the 'sample_weight' parameter."
                    )
                )

    # -----------------------------------
    def _set_transforms(self, transform_pipe: list):
        """
        Overwrites the _set_transforms() method from the BaseClass. This method
        first calls the _set_transforms() method from the BaseClass to check for
        errors in the list of transformations (transform_pipe) and then set the
        transform_pipe attribute. Following this call, it is created a copy of
        the transform_pipe attribute to each cohort. This way, each cohort will
        have its own list of transformations, which should be applied
        independently.

        :param transform_pipe: the transform_pipe parameter provided to the
            constructor of this class;
        """
        super()._set_transforms(transform_pipe)
        self.cohort_transf_list = []
        if self.cohort_list is not None:
            cohort_transforms = []
            for i in range(len(self.cohort_list)):
                cohort_transforms.append([])
                for transf in self.transform_pipe:
                    cohort_transforms[i].append(deepcopy(transf))
            self.cohort_transf_list = cohort_transforms

    # -----------------------------------
    def _fit_transforms(self, df: pd.DataFrame, y: pd.DataFrame = None):
        """
        Overwrites the _fit_transforms() method from the BaseClass. This method
        follows the same idea as the _fit_transforms() method from the BaseClass,
        but instead of calling the fit and transform methods for each of the
        transformations in the transformations list for the entire dataset,
        it calls the fit and ​transform methods for each cohort separately. It
        cycles through all cohorts, and for each cohort: (i) fetch the subset
        that belongs to the cohort, and (ii) call the fit and transform methods
        for each of the transforms in the cohort's transformations list using
        the subset dataset. This way, the transformations are applied independently
        for each cohort.

        :param df: data frame containing the features of the dataset;
        :param y: data frame containing the labels of the dataset.
        """
        for i in range(len(self.cohort_list)):
            if self.cohort_transf_list[i] != []:
                index_list = self.cohort_list[i].get_index_list().copy()
                out_group = self.cohort_list[i].get_out_group_index_list().copy()
                index_list += out_group
                subset_x = df.filter(items=index_list, axis=0)
                if y is not None:
                    subset_y = y.filter(items=index_list, axis=0)
                for tf in self.cohort_transf_list[i]:
                    fit_params = tf._get_fit_input_type()
                    if fit_params == self.FIT_INPUT_DF:
                        tf.fit(subset_x)
                        subset_x = tf.transform(subset_x)
                    elif fit_params == self.FIT_INPUT_XY:
                        if y is None:
                            raise ValueError(
                                f"ERROR: using the tranformation class {type(tf).__name__} "
                                + "that requires an X and Y datasets as a preprocessing step "
                                + "inside another class that does not require the separation "
                                + "a Y dataset (whcich contains only the labels)."
                            )
                        tf.fit(subset_x, subset_y)
                        subset_x = tf.transform(subset_x)
                    else:
                        raise NotImplementedError("ERROR: Unknown fit input order.")

    # -----------------------------------
    def _get_cohort_limits_and_query(self, cohort: tuple):
        """
        Builds a condition dictionary that controls which instances
        should be included in a cohort or not. The condition dictionary
        is created based on the tuple provided as a parameter (cohort).
        The values in this tuple are associated to the columns in
        self.cohort_cols (also in the same order), and each value
        represents the condition for an instance to be considered from
        the cohort or not. This method also creates a query that
        manages to filter a data frame to include only the instances
        from the cohort defined by the tuples 'cohort' using the query
        method from the Pandas library.

        :param cohort: a tuple of values that defines a new cohort. The
            values in this tuple are associated ​with the columns in
            self.cohort_cols (also in the same order), and each value
            represents the condition for an instance to be considered
            from the cohort or not.
        """

        def is_number(value):
            try:
                _ = float(value)
                return True
            except:
                return False

        condition = {}
        query = ""
        for j, col in enumerate(self.cohort_cols):
            condition[col] = cohort[j]
            if query != "":
                query += " and "
            if is_number(cohort[j]):
                if np.isnan(cohort[j]):
                    query += f"`{col}`.isnull()"
                else:
                    query += f"`{col}` == {cohort[j]}"
            else:
                query += f"`{col}` == '{cohort[j]}'"

        return condition, query

    # -----------------------------------
    def _get_baseline_cohorts(self):
        """
        Creates a list of baseline cohorts. This method is only used
        when the cohort_cols parameter is used in the constructor method
        (instead of using the cohort_dict). To create these cohorts:
        (i) get the unique values for each column in cohort_cols and save
        each one as a set, (ii) create a combination of all unique values
        among the different columns (a product of sets), which represents
        the possible cohorts to be created, and finally (iii) for each of
        these combinations of values across columns, create a cohort conditioned
        to these values. For more details, check the documentation of the
        cohort_cols parameter for the constructor method.
        """
        self.baseline_cohorts = []
        sets = []
        for i, col in enumerate(self.cohort_cols):
            subset = self._get_df_subset(self.df, [col])
            values = subset.iloc[:, 0].unique()
            sets.append(values)

        cohort_list = list(itertools.product(*sets))

        for i, cht in enumerate(cohort_list):
            name = f"Cohort {i}"
            cohort = CohortHandler(name, self.y, self.regression)
            condition, query = self._get_cohort_limits_and_query(cht)
            subset_x = self.df.query(query)
            subset_y = self.y.filter(items=subset_x.index, axis=0)
            cohort.set_info_df(subset_x, subset_y)
            cohort.set_conditions([condition])
            if cohort.get_size() > 0:
                self.baseline_cohorts.append(cohort)

        if self.baseline_cohorts == []:
            raise ValueError(
                "ERROR: Unexpected behavior when computing the baseline set of cohorts. "
                + "No valid cohorts were found."
            )

    # -----------------------------------
    def _get_validity_status_dict(self, cohort_list: List[CohortHandler]):
        """
        Returns a dictionary with three keys: (i) key 1 assigned to a list of
        all valid cohorts, (ii) key 2 assigned to a list of cohorts with
        invalid size, and (iii) key 3 assigned to a list of cohorts with
        invalid minority class ​occurrence rate. For more details on the validity
        of cohorts, check the documentation of the cohort_cols parameter for the
        constructor method.

        :param cohort_list: a list of cohorts.
        """
        min_cohort_size = max(self.min_cohort_size, self.df.shape[0] * self.min_cohort_pct)
        status_dict = {self.VALID_NAME: [], self.INVALID_SIZE_NAME: [], self.INVALID_DIST_NAME: []}
        for i, cohort in enumerate(cohort_list):
            valid_status = cohort.is_valid(min_cohort_size, self.minority_min_rate)
            if valid_status == CohortHandler.INVALID_SIZE:
                status_dict[self.INVALID_SIZE_NAME].append(i)
            elif valid_status == CohortHandler.INVALID_DISTRIBUTION:
                status_dict[self.INVALID_DIST_NAME].append(i)
            else:
                status_dict[self.VALID_NAME].append(i)
        return status_dict

    # -----------------------------------
    def _get_smallest_cohort(self, cohort_list: List[CohortHandler], invalid_index: int):
        """
        Return the index (according to the cohort_list) of the
        smallest cohort that has an index different than
        invalid_index.

        :param cohort_list: a list of cohorts;
        :param invalid_index: an invalid index value. The
            smallest cohort must not have this index.
        """
        min_size = -1
        min_index = -1
        for i, cohort in enumerate(cohort_list):
            size = cohort.get_size()
            if (i != invalid_index) and (size < min_size or min_size < 0):
                min_size = size
                min_index = i
        return min_index

    # -----------------------------------
    def _merge_baseline_cohorts(self):
        """
        Merges all invalid cohorts into another cohort. The aim is to
        produce a list containing only valid cohorts. The following
        steps are executed: (i) identify all invalid cohorts using the
        _get_validity_status_dict() method, and (ii) get the first
        cohort with invalid minority class distribution or the first
        cohort with invalid size (in that order ​of priority) and merge
        this cohort with the smallest cohort. Repeat these steps until
        all cohorts are valid. If in the end, only one cohort remains,
        then it could not find a set of valid cohorts with more than
        one cohort. Therefore, return an error.
        """
        cohort_list = []
        for cohort in self.baseline_cohorts:
            cohort_list.append(deepcopy(cohort))

        all_valid = False
        while not all_valid:
            status_dict = self._get_validity_status_dict(cohort_list)
            if len(status_dict[self.INVALID_DIST_NAME]) > 0:
                invalid_index = status_dict[self.INVALID_DIST_NAME][0]
            elif len(status_dict[self.INVALID_SIZE_NAME]) > 0:
                invalid_index = status_dict[self.INVALID_SIZE_NAME][0]
            else:
                all_valid = True
                break
            smallest_cohort_index = self._get_smallest_cohort(cohort_list, invalid_index)
            cohort_list[invalid_index].merge_cohorts(cohort_list[smallest_cohort_index])
            # remove the smallest cohort from the list of cohorts
            cohort_list.pop(smallest_cohort_index)
            if len(cohort_list) <= 1:
                break

        if len(cohort_list) <= 1:
            raise ValueError(
                (
                    f"ERROR: Could not create more than 1 cohort with the following restrinctions:\n"
                    f"\tMinimum instances per cohort = {self.min_cohort_size}\n"
                    f"\tMinimum percentage of instances per cohort in relation to the dataset size = {self.min_cohort_pct*100}%\n"
                    f"\tMinimum rate of the minority class per cohort = {self.minority_min_rate}"
                )
            )

        self.cohort_list = cohort_list

    # -----------------------------------
    def _fix_cohort_dict_col_name(self):
        # cycle through the cohorts
        for i, cohort_name in enumerate(self.cohort_dict.keys()):
            if self.cohort_dict[cohort_name] is None:
                if i < len(self.cohort_dict.keys()) - 1:
                    raise ValueError(
                        "ERROR: when providing a cohort dictionary through the cohort_dict parameter, "
                        + "only the last condition is allowed to be None."
                    )
                continue
            # for each cohort, cycle through the cohort's conditions
            for i in range(len(self.cohort_dict[cohort_name])):
                # cycle through the different columns in each condition
                switch_keys = []
                for col_name in self.cohort_dict[cohort_name][i].keys():
                    if type(col_name) == int:
                        new_col_name = self._get_column_from_index(col_name)
                        switch_keys.append([col_name, new_col_name])
                    elif col_name not in self.df.columns:
                        raise ValueError(
                            (
                                "ERROR: Invalid column name provided in the cohort_dict parameter. The following "
                                "condition contains a column that doesn't exist in the dataframe used:"
                                f"\n{self.cohort_dict[cohort_name][i]}"
                            )
                        )
                for pair in switch_keys:
                    old_key = pair[0]
                    new_key = pair[1]
                    value = self.cohort_dict[cohort_name][i].pop(old_key)
                    self.cohort_dict[cohort_name][i][new_key] = value

    # -----------------------------------
    def _conditions_from_list(self, conditions: list):
        """
        Reestructures the list of conditions provided by the user for a
        given cohort through the cohort_dict parameter.
        """
        final_conditions = []
        for condition_dict in conditions:
            col_list = []
            sets = []
            for col, values in condition_dict.items():
                col_list.append(col)
                if type(values) != list:
                    values = [values]
                sets.append(values)

            combination_list = list(itertools.product(*sets))
            for combination in combination_list:
                condition = {}
                for i in range(len(combination)):
                    condition[col_list[i]] = combination[i]
                final_conditions.append(condition)
        return final_conditions

    # -----------------------------------
    def _get_cohorts_from_dict(self):
        """
        Creates the cohort list from the cohort dictionary provided
        as a parameter in the constructor class (cohort_dict). Given
        the cohort dictionary with the conditions for all cohorts,
        this method executes the following instructions: (i) cycle
        through each key (one for each cohort) in the cohort_dict,
        (ii) find the subset that belongs to the cohort being analyzed,
        (iii) find the query capable of filtering this subset according
        to the conditions specified by the cohort_dict parameter, (iv)
        create a cohort (an object of the cohort_dict class) with all
        this data, and finally (v) check if there are any instances
        in the full dataset that belongs to multiple cohorts or to
        no cohort at all (in both cases, an error is raised, since
        each instance must belong to a single cohort only).
        """
        cohort_list = []
        index_cohort_count = {index: 0 for index in self.df.index}
        for i, (name, conditions) in enumerate(self.cohort_dict.items()):
            cohort = CohortHandler(name, self.y, self.regression)
            if conditions is None:
                cohorts_index = []
                for cht in cohort_list:
                    cohorts_index += cht.get_index_list()
                missing_index = [index for index in self.df.index if index not in cohorts_index]
                subset_x = self.df.filter(items=missing_index, axis=0)
                subset_y = self.y.filter(items=missing_index, axis=0)
            else:
                final_conditions = self._conditions_from_list(conditions)
                cohort.set_conditions(final_conditions)
                subset_x = cohort.get_cohort_subset(self.df)
                subset_y = self.y.filter(items=subset_x.index, axis=0)
            cohort.set_info_df(subset_x, subset_y)
            if cohort.get_size() > 0:
                cohort_list.append(cohort)
                for index in subset_x.index:
                    index_cohort_count[index] += 1

        # check if more than 1 cohort was created
        if len(cohort_list) <= 1:
            raise ValueError(
                f"ERROR: Could not create more than 1 valid cohort using the cohorts specification provided."
            )
        # check if there are any instances that don't belong to any cohort
        # or if there is any instance that belongs to multiple cohorts
        for index in index_cohort_count.keys():
            n_cohorts_belong = index_cohort_count[index]
            # instance doesn't belong to any cohort
            if n_cohorts_belong == 0:
                raise ValueError(
                    "ERROR: the cohorts created doesn't include all instances of the dataset. Add a final cohort "
                    + "to the cohort_dict parameter with a None value. This will create a cohort with all the instances "
                    + "not included in any of the other cohorts."
                )
            # instance belongs to multiple cohorts
            if n_cohorts_belong > 1:
                raise ValueError(
                    "ERROR: the cohorts created have intersecting conditions. There are instances in the dataset assigned "
                    + "to multiple cohorts."
                )

        self.cohort_list = cohort_list

    # -----------------------------------
    def _error_if_invalid_cohort(self):
        """
        Check if any of the existing cohorts are invalid.
        If at least one cohort is invalid, raise an error.
        """
        min_cohort_size = max(self.min_cohort_size, self.df.shape[0] * self.min_cohort_pct)
        for cohort in self.cohort_list:
            if cohort.is_valid(min_cohort_size, self.minority_min_rate) != CohortHandler.VALID:
                cohort.print()
                raise ValueError(
                    f"ERROR: Found an invalid cohort when trying to build the cohorts specified.\n"
                    + f"The last cohort printed shows more details of the invalid cohort."
                )

    # -----------------------------------
    def _fix_invalid_cohort_transfer_learning(self):
        """
        Fix all cohorts with invalid sizes by using Transfer Learning. Here, transfer learning
        occurs when a set of data not belonging to a given cohort is used when fitting that
        cohort's estimator, but the instances from the outside data are assigned a smaller weight
        equal to $\theta$ (check the theta parameter for the constructor method for more details).
        This method executes the following instructions: (i) identify all the invalid cohorts
        using the _get_validity_status_dict() method, (ii) throw an error if there are any cohorts
        with invalid distribution (transfer learning doesn't work in these cases), (iii) for all
        cohorts i with invalid size, do the following:
            (a) find all other cohorts j (i != j, that is, different ​than the invalid cohort being
                cycled) that have a similar label distribution (check the documentation of the
                cohort_dist_th parameter for the constructor ​method for more information),
            (b) add the data of cohort j to the outside data of cohort i (the data outside of
                cohort i that will be used when fitting cohort i's model, but this outside data
                is assigned a lower weight equal to $\theta$),
            (c) raise an error if no other cohort j was considered ​compatible with the invalid
                cohort i.
        """
        if self.cohort_dict is not None:
            cohort_list = self.cohort_list
        else:
            cohort_list = self.baseline_cohorts

        status_dict = self._get_validity_status_dict(cohort_list)

        # throw an error if there are any cohorts with invalid distribution
        if len(status_dict[self.INVALID_DIST_NAME]) > 0:
            print("INVALID COHORTS:")
            for index in status_dict[self.INVALID_DIST_NAME]:
                cohort_list[index].print()
            raise ValueError("ERROR: Cannot use transfer learning over cohorts with skewed distributions.")

        # use transfer learning over cohorts with invalid sizes. To do this, add
        # data from outside each cohort into the invalid cohort's training data
        # (this outside data will be weighed down later by the self.theta parameter).
        # Only add outside data from cohorts with a similar label distribution to
        # avoid using outside data that could harm the cohort's model performance
        for index in status_dict[self.INVALID_SIZE_NAME]:
            added_outgroup_success = False
            for outside_cohort_index in range(len(cohort_list)):
                if outside_cohort_index == index:
                    continue
                out_cohort = cohort_list[outside_cohort_index]
                if cohort_list[index].is_outside_cohort_compatible(out_cohort, self.cohort_dist_th):
                    cohort_list[index].add_out_group_instances(out_cohort, self.theta)
                    added_outgroup_success = True

            if not added_outgroup_success:
                self.cohort_list = cohort_list
                self.print_cohorts()
                raise ValueError(
                    f"ERROR: could not find any outside cohort with a compatible label distribution to be used "
                    + f"as the out data for cohort {cohort_list[index].name} (used during transfer learning). "
                    + f"Change the cohort creation parameters or lower the value of the cohort_dist_th parameter."
                )

        self.cohort_list = cohort_list

    # -----------------------------------
    def _get_cohorts(self):
        """
        Build the cohorts. The approach used to build the cohorts
        depends if a dictionary of cohorts was provided (cohort_dict)
        or if a list of cohort columns was provided (cohort_cols).
        If transfer learning is used (if a theta parameter is provided),
        then also fix all invalid cohorts to use transfer learning.
        Otherwise, fix the invalid cohorts by merging them with other
        cohorts.
        """
        if self.df is None:
            return
        # if the cohorts are specified
        if self.cohort_dict is not None:
            self._fix_cohort_dict_col_name()
            self._get_cohorts_from_dict()
            # if transfer learning is not used
            if not self.theta:
                self._error_if_invalid_cohort()
            # if using transfer learning
            else:
                self._fix_invalid_cohort_transfer_learning()
        # if cohorts are build automatically based on a set of features
        else:
            self.cohort_cols = self._check_error_col_list(self.df, self.cohort_cols, "cohort_cols")
            self._get_baseline_cohorts()
            # if transfer learning is not used
            if not self.theta:
                self._merge_baseline_cohorts()
            # if using transfer learning
            else:
                self._fix_invalid_cohort_transfer_learning()

    # -----------------------------------
    def _fit_estimators_cohorts(self):
        """
        Fits the estimator of each cohort using the cohort's data.
        Each cohort is assigned a different copy of the base estimator
        provided in the constructor method. This way, the estimator of
        each cohort is independent of the estimator of the other
        cohorts.
        """
        for i, cohort in enumerate(self.cohort_list):
            cohort.set_estimator(deepcopy(self.estimator), self.regression)
            cohort.get_cohort_and_fit_estimator(
                self.df, self.y, self.cohort_transf_list[i], self.min_size_fold, self.valid_k_folds, self.default_theta
            )

    # -----------------------------------
    def _fit(self):
        """
        Steps for running the fit method for the current class.
        """
        self._get_cohorts()
        self._set_transforms(self.transform_pipe)
        self._fit_transforms(self.df, self.y)
        self._fit_estimators_cohorts()

    # -----------------------------------
    def fit(self, df: pd.DataFrame = None, label_col: str = None, X: pd.DataFrame = None, y: pd.DataFrame = None):
        """
        Overwrites the fit() method of the base Estimator class. Implements
        the steps for running the fit method for the current class.

        :param df: the full dataset;
        :param label_col: the name or index of the label column;
        :param X: contains only the features of the original dataset, that is, does not
            contain the label column;
        :param y: contains only the label column of the original dataset;
        """
        self._set_df_mult(df, label_col, X, y, require_set=True)
        self._check_regression()
        self._set_estimator()
        self._fit()
        self.classes_ = np.sort(self.y.unique())

    # -----------------------------------
    def _raise_missing_inference_error(self, df: pd.DataFrame, index_list: list):
        """
        Raises an error when there are instances in an inference dataset
        that didn't have any predictions. This means that these instances
        didn't fit into any of the existing cohorts.

        :param df: the dataset used for inference;
        :param index_list: the indices from df that have a valid prediction.
            From this, we can obtain the indices that don't have a valid
            prediction.
        """
        set_difference = set(df.index) - set(index_list)
        missing = list(set_difference)
        missing_subset = df.filter(items=missing, axis=0)
        raise ValueError(
            "ERROR: a subset of the instances passed for prediction doesn't fit into any of the existing cohorts.\n"
            + f"The subset is given as follows:\n{missing_subset}"
        )

    # -----------------------------------
    def _predict(self, df: pd.DataFrame, probability: bool = False):
        """
        Predicts the labels of the instances in the provided dataframe. Can return the
        predicted probabilities for each instance belonging to each class or the
        predicted classes (the probabilities are binarized).

        :param df: the dataset used for inference;
        :param probability: if True, return the probability values (predict_proba). If
            False, transform the probabilities to class values using the threshold
            computed by the _get_best_prob_th() method.
        """
        df_copy = df.reset_index(drop=True)
        index_list = []
        pred_list = None
        for i in range(len(self.cohort_list)):
            cohort_index, cohort_pred = self.cohort_list[i].find_instances_cohort_and_predict(
                df_copy, self.cohort_transf_list[i], index_list, probability
            )
            index_list += cohort_index
            if pred_list is None:
                pred_list = cohort_pred
            else:
                pred_list = np.concatenate((pred_list, cohort_pred), axis=0)

        index_sort = np.argsort(index_list)
        final_pred = []
        for i in index_sort:
            final_pred.append(pred_list[i])

        if len(index_sort) != df_copy.shape[0]:
            self._raise_missing_inference_error(df_copy, index_list)

        return np.array(final_pred)

    # -----------------------------------
    def predict(self, df: pd.DataFrame):
        """
        Predicts the labels of the instances in the provided dataframe.
        This is the public method that should be called externally.

        :param df: the dataset used for inference;
        """
        predict = self._predict(df)
        return predict

    # -----------------------------------
    def predict_proba(self, df: pd.DataFrame):
        """
        Predicts the probabilities for each instance (of the provided
        dataframe) belonging to each class. This is the public method
        that should be called externally.

        :param df: the dataset used for inference;
        """
        if self.regression:
            raise ValueError("ERROR: the predict_proba() method doesn't work for regression tasks.")
        predict = self._predict(df, probability=True)
        return predict

    # -----------------------------------
    def print_cohorts(self):
        """
        Print the information of all cohorts created.
        """
        if self.cohort_list is None:
            print(
                "No cohorts built yet. To build the cohorts, pass the dataset to the constructor or call the fit method."
            )
            return
        print("FINAL COHORTS")
        for cohort in self.cohort_list:
            cohort.print()
            print("\n")
