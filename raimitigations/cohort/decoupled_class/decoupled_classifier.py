from typing import List
from copy import deepcopy
from typing import Union
import inspect

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..cohort_handler import CohortHandler
from .decoupled_cohort import _DecoupledCohort
from ...utils.data_utils import err_float_01


class DecoupledClass(CohortHandler):
    """
    Concrete class that trains different models over different subsets of data (cohorts).
    Based on the work presented in the following paper: `Decoupled classifiers for group-fair
    and efficient machine learning
    <https://www.microsoft.com/en-us/research/publication/decoupled-classifiers-for-group-fair-and-efficient-machine-learning/>`_.
    This is useful when a given cohort behaves differently from the rest of the dataset,
    or when a cohort represents a minority group that is underrepresented. For small cohorts,
    it is possible to train a model using the data of other cohorts (outside data) with a
    smaller weight $\theta$ (only works with models that accept instance weights). This process
    is herein called Transfer Learning. Instead of using transfer learning, it is also possible
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

    :param cohort_def: a list of cohort definitions or a dictionary of cohort definitions. A
        cohort condition is the same variable received by the ``cohort_definition`` parameter
        of the ``CohortDefinition`` class. When using a list of cohort definitions, the cohorts
        will be named automatically. For the dictionary of cohort definitions, the key used represents
        the cohort's name, and the value assigned to each key is given by that cohort's conditions.
        This parameter can't be used together with the ``cohort_col`` parameter. Only one these two
        parameters must be used at a time;

    :param cohort_col: a list of column names that indicates which columns should be used
        to create a cohort. For example, if ``cohort_col`` = ["C1", "C2"], then we first identify
        all possible values for column "C1" and "C2". Suppose that the unique values in "C1"
        are: [0, 1, 2], and the unique values in "C2" are: ['a', 'b']. Then, we create one
        cohort for each combination between these two sets of unique values. This way, the
        first cohort will be conditioned to instances where ("C1" == 0 and "C2" == 'a'),
        cohort 2 will be conditioned to ("C1" == 0 and "C2" == 'b'), and so on. They are
        called the baseline cohorts. We then check if there are any of the baseline cohorts
        that are invalid, where an invalid cohort is considered as being a cohort with
        size < ``max(min_cohort_size, df.shape[0] * min_cohort_pct)`` or a cohort with a
        minority class (the label value with least ocurrences) with an occurrence rate <
        ``minority_min_rate``. Every time an invalid cohort is found, we merge this cohort to
        the current smallest cohort. This is simply a heuristic, as identifying the best
        way to merge these cohorts in a way that results in a list of valid cohorts is a
        complex problem that we do not try to solve here. Note that if using transfer
        learing (check the ``theta`` parameter for more details), then the baseline
        cohorts are not merged if they are found invalid. Instead, we use transfer learning
        over the invalid cohorts;

    :param cohort_json_files: a list with the name of the JSON files that contains the definition
        of each cohort. Each cohort is saved in a single JSON file, so the length of the
        ``cohort_json_files`` should be equal to the number of cohorts to be used.

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
        values are tested, and if none of these results in folds large enough, a default
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
        valid. Check the ``cohort_col`` parameter for more information;

    :param min_cohort_pct: a value between [0, 1] that determines the minimum size allowed
        for a cohort. The minimum size is given by the size of the full dataset (df.shape[0])
        multiplied by min_cohort_pct. The maximum value between min_cohort_size and
        (df.shape[0] * min_cohort_pct) is used to determine the minimum size allowed for a
        cohort. Check the ``cohort_col`` parameter for more information;

    :param minority_min_rate: the minimum occurrence rate for the minority class (from the label
        column) that a cohort is allowed to have. If the minority class of the cohort has an
        occurrence rate lower than min_rate, the cohort is considered invalid. Check the
        ``cohort_col`` parameter for more information;

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
        cohort_def: dict = None,
        cohort_col: list = None,
        cohort_json_files: list = None,
        theta: Union[float, List[float], bool] = False,
        default_theta: float = None,
        cohort_dist_th: float = 0.8,
        min_fold_size_theta: int = MIN_FOLD_SIZE_THETA,
        valid_k_folds_theta: List[int] = VALID_K_FOLDS,
        estimator: BaseEstimator = None,
        min_cohort_size: int = MIN_COHORT_SIZE,
        min_cohort_pct: float = MIN_COHORT_SIZE_PCT,
        minority_min_rate: float = MINORITY_MIN_RATE,
        verbose: bool = True,
    ):
        self.regression = regression
        self.estimator = estimator
        super().__init__(cohort_def, cohort_col, cohort_json_files, df, label_col, X, y, verbose)
        self._set_transforms(transform_pipe)
        self._set_theta_param(theta, min_fold_size_theta, valid_k_folds_theta, default_theta, cohort_dist_th)
        self._set_cohorts_min_size(min_cohort_size, min_cohort_pct, minority_min_rate)
        self._check_regression()
        self._build_cohorts()
        self._check_and_update_cohorts()

    # -----------------------------------
    def _instantiate_cohort(self, cohort_definition: Union[list, str], name: str = "cohort"):
        """
        Create a cohort object from the ``_DecoupledCohort`` based on the specifications
        provided in the parameters.

        :param cohort_definition: a list of conditions or a string containing the path
            to a JSON file that has the list condition. Check the description of this parameter
            in the constructor method of the ``CohortDefinition`` class for more info.
        :param name: a string indicating the name of the cohort. This parameter may be accessed
            later using the ``name`` attribute.
        :return: an object from the ``_DecoupledCohort`` class.
        :rtype: _DecoupledCohort
        """
        return _DecoupledCohort(self.y, cohort_definition, name, self.regression)

    # -----------------------------------
    def _build_cohorts(self):
        """
        Overwrites the ``_build_cohorts()`` method from the parent class ``CohortHandler``.
        The current class (``DecoupledClass``) uses a different class when creating its
        cohorts: it uses the ``_DecoupledCohort`` instead of the base ``CohortDefinition``
        class. And the former class requires a few extra steps after it is created,
        which are implemented here.
        """
        if self.df is None or self.cohorts is not None:
            return

        if self.cohort_def is None:
            self._use_baseline_cohorts = True
            self.cohort_col = self._check_error_col_list(self.df, self.cohort_col, "cohort_col")
            self._cohort_col_to_def()
            if self.cohort_def is None:
                return

        self.cohorts = []
        index_used = []
        for i, cohort_def in enumerate(self.cohort_def):
            if cohort_def is None and i < len(self.cohort_def) - 1:
                raise ValueError("ERROR: only the last cohort is allowed to have a condition list assigned to 'None'.")
            cohort = self._instantiate_cohort(cohort_def, self._cohort_names[i])
            subset_x, subset_y, subset_index = cohort.get_cohort_subset(
                self.df, self.y, index_used=index_used, return_index_list=True
            )
            index_used += subset_index
            cohort.set_info_df(subset_x, subset_y)
            self.cohorts.append(cohort)

    # -----------------------------------
    def _get_base_estimator(self):
        """
        Returns the default estimator that should be used for each cohort.
        For regression tasks, BASE_REGRESSOR (internal variable) is returned.
        For classification tasks, BASE_CLASSIFIER. These base estimators are
        only used if the user doesn't provide any estimator through the estimator
        parameter.

        :return: the base estimator to be used. It could be a base classifier
            or a base regression model;
        :rtype: BaseEstimator
        """
        if self.regression:
            return self.BASE_REGRESSOR
        return self.BASE_CLASSIFIER

    # -----------------------------------
    def _set_transforms(self, transform_pipe: list):
        """
        Overwrites the _set_transforms() method from the DataProcessing. Thismethod
        first calls the _set_transforms() method from the DataProcessing to check for
        errors in the list of transformations (transform_pipe) and then set the
        transform_pipe attribute. Following this call, it is created a copy of
        the transform_pipe attribute to each cohort. This way, each cohort will
        have its own list of transformations, which should be applied
        independently.

        :param transform_pipe: the transform_pipe parameter provided to the
            constructor of this class;
        """
        super()._set_transforms(transform_pipe)
        self._cohort_pipe = []
        if self.cohorts is not None:
            cohort_transforms = []
            for i in range(len(self.cohorts)):
                cohort_transforms.append([])
                for transf in self.transform_pipe:
                    cohort_transforms[i].append(deepcopy(transf))
            self._cohort_pipe = cohort_transforms

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
                    "ERROR: invalid estimator provided. When using transfer learning, only "
                    + "estimators capable of working with sample weights are valid. The fit "
                    + "method of the estimator must have the 'sample_weight' parameter."
                )

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
    def _get_validity_status_dict(self, cohort_list: List[_DecoupledCohort]):
        """
        Returns a dictionary with three keys: (i) key 1 assigned to a list of
        all valid cohorts, (ii) key 2 assigned to a list of cohorts with
        invalid size, and (iii) key 3 assigned to a list of cohorts with
        invalid minority class occurrence rate. For more details on the validity
        of cohorts, check the documentation of the cohort_col parameter for the
        constructor method.

        :param cohort_list: a list of cohorts.
        :return: a dictionary with three keys:

            * **Key 1:** a list of all valid cohorts
            * **Key 2:** a list of cohorts with invalid size
            * **Key 3:** a list of cohorts with invalid minority class occurrence
              rate.
        :rtype: dict
        """
        min_cohort_size = max(self.min_cohort_size, self.df.shape[0] * self.min_cohort_pct)
        status_dict = {self.VALID_NAME: [], self.INVALID_SIZE_NAME: [], self.INVALID_DIST_NAME: []}
        for i, cohort in enumerate(cohort_list):
            valid_status = cohort.is_valid(min_cohort_size, self.minority_min_rate)
            if valid_status == _DecoupledCohort.INVALID_SIZE:
                status_dict[self.INVALID_SIZE_NAME].append(i)
            elif valid_status == _DecoupledCohort.INVALID_DISTRIBUTION:
                status_dict[self.INVALID_DIST_NAME].append(i)
            else:
                status_dict[self.VALID_NAME].append(i)
        return status_dict

    # -----------------------------------
    def _get_smallest_cohort(self, cohort_list: List[_DecoupledCohort], invalid_index: int):
        """
        Return the index (according to the cohort_list) of the
        smallest cohort that has an index different than
        invalid_index.

        :param cohort_list: a list of cohorts;
        :param invalid_index: an invalid index value. The
            smallest cohort must not have this index.
        :return: the index of ``cohort_list`` associated with the
            smallest cohort.
        :rtype: int
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
        cohort with invalid size (in that order of priority) and merge
        this cohort with the smallest cohort. Repeat these steps until
        all cohorts are valid. If in the end, only one cohort remains,
        then it could not find a set of valid cohorts with more than
        one cohort. Therefore, return an error.
        """
        cohort_list = []
        for cohort in self.cohorts:
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
                f"ERROR: Could not create more than 1 cohort with the following restrinctions:\n"
                + f"\tMinimum instances per cohort = {self.min_cohort_size}\n"
                + f"\tMinimum percentage of instances per cohort in relation to the dataset size = {self.min_cohort_pct*100}%\n"
                + f"\tMinimum rate of the minority class per cohort = {self.minority_min_rate}"
            )

        self.cohorts = cohort_list

    # -----------------------------------
    def _error_if_invalid_cohort(self):
        """
        Check if any of the existing cohorts are invalid.
        If at least one cohort is invalid, raise an error.
        """
        min_cohort_size = max(self.min_cohort_size, self.df.shape[0] * self.min_cohort_pct)
        for cohort in self.cohorts:
            if cohort.is_valid(min_cohort_size, self.minority_min_rate) != _DecoupledCohort.VALID:
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
        cohort_list = self.cohorts
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
                self.cohorts = cohort_list
                self.print_cohorts()
                raise ValueError(
                    f"ERROR: could not find any outside cohort with a compatible label distribution to be used "
                    + f"as the out data for cohort {cohort_list[index].name} (used during transfer learning). "
                    + f"Change the cohort creation parameters or lower the value of the cohort_dist_th parameter."
                )

        self.cohorts = cohort_list

    # -----------------------------------
    def _check_and_update_cohorts(self):
        """
        Check if the generated cohorts are according to the conditions
        tested in the _get_validity_status_dict() function.
        If transfer learning is used (if a theta parameter is provided),
        then fix all invalid cohorts to use transfer learning.
        Otherwise, either fix the invalid cohorts by merging them with other
        cohorts, or give an error.
        """
        if self.df is None:
            return
        # if the cohorts are specified
        if not self._use_baseline_cohorts:
            # if transfer learning is not used
            if not self.theta:
                self._error_if_invalid_cohort()
            # if using transfer learning
            else:
                self._fix_invalid_cohort_transfer_learning()
        # if cohorts are build automatically based on a set of features
        else:
            # if transfer learning is not used
            if not self.theta:
                self._merge_baseline_cohorts()
            # if using transfer learning
            else:
                self._fix_invalid_cohort_transfer_learning()

    # -----------------------------------
    def _fit_cohorts(self):
        """
        Fits the estimator of each cohort using the cohort's data.
        Each cohort is assigned a different copy of the base estimator
        provided in the constructor method. This way, the estimator of
        each cohort is independent of the estimator of the other
        cohorts.
        """
        index_used = []
        for i, cohort in enumerate(self.cohorts):
            cohort.set_estimator(deepcopy(self.estimator), self.regression)
            index_list = cohort.get_cohort_and_fit_estimator(
                self.df, self.y, self._cohort_pipe[i], self.min_size_fold, self.valid_k_folds, self.default_theta
            )
            index_used += index_list

        self._check_intersection_cohorts(index_used)
        self._raise_missing_instances_error(self.df, index_used)

    # -----------------------------------
    def _fit(self):
        """
        Steps for running the fit method for the current class.
        """
        self._build_cohorts()
        self._check_and_update_cohorts()
        self._set_transforms(self.transform_pipe)
        self._fit_cohorts()

    # -----------------------------------
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray] = None,
        y: Union[pd.Series, np.ndarray] = None,
        df: pd.DataFrame = None,
        label_col: str = None,
    ):
        """
        Overwrites the fit() method of the base CohortHandler class. Implements
        the steps for running the fit method for the current class.

        :param X: contains only the features of the original dataset, that is, does not
            contain the label column;
        :param y: contains only the label column of the original dataset;
        :param df: the full dataset;
        :param label_col: the name or index of the label column;

        Check the documentation of the _set_df_mult method (DataProcessing class)
        for more information on how these parameters work.
        """
        self._set_df_mult(df, label_col, X, y, require_set=True)
        self._check_regression()
        self._set_estimator()
        self._fit()
        self.classes_ = np.sort(self.y.unique())

        self.fitted = True
        return self

    # -----------------------------------
    def _predict(self, X: pd.DataFrame, probability: bool = False, split_pred: bool = False):
        """
        Predicts the labels of the instances in the provided dataframe. Can return the
        predicted probabilities for each instance belonging to each class or the
        predicted classes (the probabilities are binarized).

        :param X: the dataset used for inference;
        :param probability: if True, return the probability values (``predict_proba``). If
            False, and if the estimator has the ``predict_proba`` method, then transform the
            probabilities to class values using the threshold computed by the _get_best_prob_th()
            method. Otherwise, just invoke the ``predict()`` method from each estimator;
        :param split_pred: if True, return a dictionary with the predictions
            for each cohort. If False, return a single predictions array;
        :return: an array with the predictions of all instances of the dataset, built from the
            predictions of each cohort, or a dictionary with the predictions for each cohort;
        :rtype: np.ndarray or dict
        """
        self._check_if_fitted()
        df = self._fix_col_transform(X)
        df = df.reset_index(drop=True)

        index_used = []
        pred_dict = {}
        for i, cohort in enumerate(self.cohorts):
            cohort_index, cohort_pred = self.cohorts[i].find_instances_cohort_and_predict(
                df, self._cohort_pipe[i], index_used, probability
            )
            index_used += cohort_index
            pred_dict[cohort.name] = cohort_pred

        self._check_intersection_cohorts(index_used)
        self._raise_missing_instances_error(df, index_used)
        final_pred = self._merge_cohort_predictions(pred_dict, index_used, split_pred)

        return final_pred

    # -----------------------------------
    def predict(self, X: Union[pd.DataFrame, np.ndarray], split_pred: bool = False):
        """
        Calls the ``transform()`` method of all transformers in all pipelines, followed
        by the ``predict()`` method for the estimator.

        :param X: contains only the features of the dataset to be transformed;
        :param split_pred: if True, return a dictionary with the predictions
            for each cohort. If False, return a single predictions array;
        :return: an array with the predictions of all instances of the dataset, built from the
            predictions of each cohort, or a dictionary with the predictions for each cohort;
        :rtype: np.ndarray or dict
        """
        final_pred = self._predict(X, probability=False, split_pred=split_pred)
        return final_pred

    # -----------------------------------
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray], split_pred: bool = False):
        """
        Calls the ``transform()`` method of all transformers in all pipelines, followed
        by the ``predict_proba()`` method for the estimator.

        :param X: contains only the features of the dataset to be transformed;
        :param split_pred: if True, return a dictionary with the predictions
            for each cohort. If False, return a single predictions array;
        :return: an array with the predictions of all instances of the dataset, built from the
            predictions of each cohort, or a dictionary with the predictions for each cohort;
        :rtype: np.ndarray or dict
        """
        final_pred = self._predict(X, probability=True, split_pred=split_pred)
        return final_pred

    # -----------------------------------
    def print_cohorts(self):
        """
        Print the information of all cohorts created.
        """
        if self.cohorts is None:
            print(
                "No cohorts built yet. To build the cohorts, pass the dataset to the constructor "
                + "or call the fit method."
            )
            return
        print("FINAL COHORTS")
        for cohort in self.cohorts:
            cohort.print()
            print("\n")
