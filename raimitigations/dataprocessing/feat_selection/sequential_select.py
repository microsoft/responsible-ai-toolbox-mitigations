from typing import Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, is_classifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import json
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from .selector import FeatureSelection


class SeqFeatSelection(FeatureSelection):
    """
    Concrete class that uses ``SequentialFeatureSelector`` over a dataset.
    Implements the sequential feature selection method using the``mlextend``
    library. This approach uses a classifier and sequentially changes the
    set of features used for training the model. There are two ways to
    perform this: (i) forward feature selection or (ii) backward feature
    selection. The former starts with an empty set of features and tests
    the performance of the model when inserting each of the non-selected
    features. The feature with the best score in the test set is added to
    the selected features. It then restarts the process until the number
    of desired features is reached. The backward feature selection is the
    opposite: it starts with all the features and removes them one by one.
    The feature that has the least impact on the score of the test set is
    selected to be removed. This is repeated until the number of remaining
    features is the desired number of features.

    :param df: the data frame to be used during the fit method.
        This data frame must contain all the features, including the label
        column (specified in the ``label_col`` parameter). This parameter is
        mandatory if ``label_col`` is also provided. The user can also provide
        this dataset (along with the ``label_col``) when calling the :meth:`fit`
        method. If ``df`` is provided during the class instantiation, it is not
        necessary to provide it again when calling :meth:`fit`. It is also possible
        to use the  ``X`` and  ``y`` instead of  ``df`` and ``label_col``, although it is
        mandatory to pass the pair or parameters (X,y) or (df, label_col) either
        during the class instantiation or during the :meth:`fit` method;

    :param label_col: the name or index of the label column. This parameter is
        mandatory if  ``df`` is provided;

    :param X: contains only the features of the original dataset, that
        is, does not contain the label column. This is useful if the user has
        already separated the features from the label column prior to calling this
        class. This parameter is mandatory if  ``y`` is provided;

    :param y: contains only the label column of the original dataset.
        This parameter is mandatory if  ``X`` is provided;

    :param transform_pipe: a list of transformations to be used as a pre-processing
        pipeline. Each transformation in this list must be a valid subclass of the
        current library (:class:`~raimitigations.dataprocessing.EncoderOrdinal`, :class:`~raimitigations.dataprocessing.BasicImputer`, etc.). Some feature selection
        methods require a dataset with no categorical features or with no missing values
        (depending on the approach). If no transformations are provided, a set of default
        transformations will be used, which depends on the feature selection approach
        (subclass dependent);

    :param in_place: indicates if the original dataset will be saved internally (df_org)
        or not. If True, then the feature selection transformation is saved over the
        original dataset. If False, the original dataset is saved separately (default
        value);

    :param regression: if True and no estimator is provided, then create a default
        CatBoostRegressor. If False, a CatBoostClassifier is created instead. This parameter
        is ignored if an estimator is provided using the 'estimator' parameter;

    :param estimator: a sklearn estimator to be used during the sequential
        feature selection process. If no estimator is provided, a default classifier or
        regressor is used (BASE_CLASSIFIER and BASE_REGRESSOR, respectively);

    :param n_feat: the number of features to be selected. Can be an
        integer, string, or tuple:

            * **int**: a number between 1 and df.shape[1] (number of features);
            * **string:** the only value accepted in this case is the "best" string, which
              selects the number of features with the best score using cross-validation;
            * **tuple:** a tuple with only 2 values: (min, max), where min and max
              are the minimum and maximum number of features to be selected. The
              number of features selected the number of features that achieved
              the best score in the cross-validation and that is between min and max;

    :param fixed_cols: a list of column names or indices that should always be included in the
        set of selected features. Note that the number of columns included here must be smaller
        than n_feat, otherwise there is nothing for the class to do (that is:
        len(fixed_cols) < n_feat);

    :param cv: the number of folds used for the cross-validation;

    :param scoring: the score used to indicate which set of features is better. The set of valid
        values for this parameter depends on the task being solved: regression or classification.
        The valid values are:

            * **Regression:** "neg_mean_squared_error", "r2", "neg_median_absolute_error";
            * **Classification:** "accuracy", "f1", "precision", "recall", "roc_auc".

        If None, "roc_auc" is used for classification tasks, and "r2" is used for regression tasks;

    :param forward: if True, a forward sequential feature selection approach is used.
        If False, a backward sequential feature selection approach is used;

    :param save_json: if True, the summary json will be saved in the path specified by the
        json_summary parameter after calling the fit() method. If False, this json file is
        not saved;

    :param json_summary: the path where the summary with the results obtained by the feature
        selection process should be saved. This summary is saved after the fit() method is
        called. Note that this summary is only saved if save_json is set to True;

    :param n_jobs: the number of workers used to run the sequential feature selection method;

    :param verbose: indicates whether internal messages should be printed or not.
    """

    PCT_FEAT_SEL = 0.5
    VALID_NFEAT_STR = ["best"]
    VALID_SCORING_CLASS = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    VALID_SCORING_REG = ["neg_mean_squared_error", "r2", "neg_median_absolute_error"]

    BASE_CLASSIFIER = DecisionTreeClassifier(max_features="sqrt")
    BASE_REGRESSOR = DecisionTreeRegressor(max_features="sqrt")

    # -----------------------------------
    def __init__(
        self,
        df: Union[pd.DataFrame, np.ndarray] = None,
        label_col: str = None,
        X: Union[pd.DataFrame, np.ndarray] = None,
        y: Union[pd.DataFrame, np.ndarray] = None,
        transform_pipe: list = None,
        in_place: bool = False,
        regression: bool = None,
        estimator: BaseEstimator = None,
        n_feat: Union[int, str, tuple] = "best",
        fixed_cols: list = None,
        cv: int = 3,
        scoring: str = None,
        forward: bool = True,
        save_json: bool = False,
        json_summary: str = "seq_feat_summary.json",
        n_jobs: int = 1,
        verbose: bool = True,
    ):
        super().__init__(df, label_col, X, y, transform_pipe, in_place, verbose)
        self.cv = cv
        self.scoring = scoring
        self.forward = forward
        self.njobs = n_jobs
        self.estimator = estimator
        self.n_feat = n_feat
        self.fixed_cols = fixed_cols
        self.save_json = save_json
        self.json_summary = json_summary
        self.regression = regression
        self._check_n_feat()
        self._check_scoring()
        self._set_estimator()

    # -----------------------------------
    def _check_n_feat(self):
        """
        Checks if the value provided to the n_feat parameter is valid.
        If it is not valid, an appropriate ValueError is raised informing
        the user of a possible explanation for the error.
        """
        if self.n_feat is None:
            raise ValueError(
                "ERROR: 'n_feat' must be an integer value between 1 and the "
                + "number of columns of the targeted dataframe. Instead, got None."
            )
        if type(self.n_feat) == int:
            n_feat_invalid = False
            if self.df is not None and self.n_feat >= self.df.shape[1]:
                n_feat_invalid = True
            if self.n_feat <= 0 or n_feat_invalid:
                raise ValueError(
                    "ERROR: 'n_feat' must be an integer value between 1 and the "
                    + "number of columns of the targeted dataframe."
                )
        elif type(self.n_feat) == str:
            if self.n_feat not in self.VALID_NFEAT_STR:
                raise ValueError(
                    "ERROR: invalid 'n_feat' string value. 'n_feat' must be one "
                    + f"of the following strings: {self.VALID_NFEAT_STR}"
                )
        elif type(self.n_feat) == tuple:
            error = False
            if self.df is not None:
                if len(self.n_feat) != 2:
                    error = True
                elif type(self.n_feat[0]) != int or type(self.n_feat[1]) != int:
                    error = True
                elif self.n_feat[0] >= self.n_feat[1]:
                    error = True
                elif self.n_feat[0] <= 0 or self.n_feat[0] >= self.df.shape[1]:
                    error = True
                elif self.n_feat[1] <= 0 or self.n_feat[1] >= self.df.shape[1]:
                    error = True
                if error:
                    raise ValueError(
                        "ERROR: invalid 'n_feat' tuple. Expected a tuple with two integer values, "
                        + "where n_feat[0] and n_feat[1] specifies the minimum and maximum number of "
                        + "features to be selected, respectively."
                    )
        else:
            raise ValueError(
                f"ERROR: unexpected 'n_feat' parameter format. Expected 'n_feat' to be a string in {self.VALID_NFEAT_STR}, "
                + "an integer value between 1 and the number of columns of the targeted dataframe, or a tuple with two "
                + "integer values, where n_feat[0] and n_feat[1] specifies the minimum and maximum number of "
                + "features to be selected, respectively."
            )

    # -----------------------------------
    def _check_fixed_columns(self):
        """
        Checks for any errors or inconsistencies in the fixed_cols parameter.
        If any errors are encountered, an error is raised.
        """
        if self.fixed_cols is None:
            return

        if type(self.fixed_cols) != list:
            raise ValueError(
                "ERROR: 'fixed_cols' must be a list. It should contain a list of column names or indices that "
                + "should be present in the set of selected features."
            )

        self.fixed_cols = self._check_error_col_list(self.df, self.fixed_cols, "fixed_cols")

        if type(self.n_feat) == int:
            if len(self.fixed_cols) >= self.n_feat:
                raise ValueError(
                    "ERROR: the number features to be selected (n_feat) must be greater than the number of "
                    + f"fixed columns (fixed_cols). Instead, got n_feat = {self.n_feat} <= "
                    + f"fixed_cols = {self.fixed_cols}."
                )
        elif type(self.n_feat) == tuple:
            if len(self.fixed_cols) >= self.n_feat[0]:
                raise ValueError(
                    "ERROR: the number features to be selected (n_feat) must be greater than the number of "
                    + f"fixed columns (fixed_cols). Instead, got: "
                    + f"n_feat = ({self.n_feat[0]}, {self.n_feat[1]}) <= fixed_cols = {self.fixed_cols}."
                )

    # -----------------------------------
    def _check_scoring(self):
        """
        Checks if the value provided to the scoring parameter is valid.
        If it is not valid, an appropriate ValueError is raised informing
        the user of a possible explanation for the error. If scoring is set
        to None, a default value is used based on the task being solved:
        classification or regression.
        """
        if self.scoring is None:
            if self.regression is not None:
                self.scoring = "roc_auc"
                if self.regression:
                    self.scoring = "r2"
            return

        if self.scoring not in self.VALID_SCORING_CLASS and self.scoring not in self.VALID_SCORING_REG:
            raise ValueError(
                f"ERROR: expected 'scoring' to be one of the following values:\n"
                + f" - CLASSIFICATION: {self.VALID_SCORING_CLASS}\n"
                + f" - REGRESSION: {self.VALID_SCORING_REG}\n"
            )
        if self.regression is not None:
            if self.regression:
                if self.scoring not in self.VALID_SCORING_REG:
                    raise ValueError(
                        f"ERROR: the value passed to the 'scoring' parameter is not valid for a regression task. "
                        + f"Expected 'scoring' to be one of the following values (regression): {self.VALID_SCORING_REG}"
                    )
            elif self.scoring not in self.VALID_SCORING_CLASS:
                raise ValueError(
                    f"ERROR: the value passed to the 'scoring' parameter is not valid for a classification task. "
                    + f"Expected 'scoring' to be one of the following values (classification): "
                    + f"{self.VALID_SCORING_CLASS}"
                )

    # -----------------------------------
    def _get_base_estimator(self):
        """
        Returns the default estimator that should be used for the Sequential
        Feature Selection method. The default estimator is defined by the
        internal variable BASE_CLASSIFIER and BASE_REGRESSOR for a default
        classifier and a regressor, respectively. This base estimator is only
        used if the user doesn't provide any estimator through the estimator
        parameter.
        """
        if self.regression:
            return self.BASE_REGRESSOR
        return self.BASE_CLASSIFIER

    # -----------------------------------
    def _set_estimator(self):
        """
        Sets the self.estimator attribute based on the estimator passed by
        the user through the estimator parameter. If estimator is None, then
        a default estimator is used (defined by BASE_CLASSIFIER and BASE_REGRESSOR).
        Otherwise, the estimator is checked to see if it is an estimator from the
        sklearn library. If not, an error is raised, since only sklearn estimators
        are allowed.
        """
        if self.estimator is None:
            if self.regression is not None:
                self.estimator = self._get_base_estimator()
        else:
            if not isinstance(self.estimator, BaseEstimator):
                raise ValueError("ERROR: Expected 'estimator' to be a SKLearn classifier or regressor.")
            self.regression = True
            if is_classifier(self.estimator):
                self.regression = False

    # -----------------------------------
    def _check_regression(self):
        if self.regression is not None:
            return

        self.regression = False
        if "float" in self.y.dtype.name:
            self.regression = True

    # -----------------------------------
    def _run_feat_selection(self):
        """
        Runs the Sequential Feature Selection method using the
        SequentialFeatureSelector class from the mlxtend.feature_selection
        library.
        """
        verbose = 0
        if self.verbose:
            verbose = 2
        self.selector = SFS(
            self.estimator,
            k_features=self.n_feat,
            forward=self.forward,
            floating=False,
            verbose=verbose,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.njobs,
            fixed_features=self.fixed_cols,
        )
        self.selector.fit(self.df, self.y)

    # -----------------------------------
    def _save_json(self):
        """
        Saves the summary dictionary returned by the get_summary()
        method into a JSON file specified by the self.json_summary
        attribute.
        """
        if not self.save_json:
            return

        def default(obj):
            if type(obj).__module__ == np.__name__:
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj.item()
            raise TypeError("Unknown type:", type(obj))

        if self.json_summary is not None:
            dict_format = json.dumps(self.selector.subsets_, default=default)
            dict_format = json.loads(dict_format)
            with open(self.json_summary, "w") as json_file:
                json.dump(dict_format, json_file)

    # -----------------------------------
    def _fit(self):
        """
        Steps for running the fit method for the current class.
        """
        self._check_regression()
        self._set_estimator()
        self._check_scoring()
        self._check_n_feat()
        self._check_fixed_columns()
        self._run_feat_selection()
        self._save_json()

    # -----------------------------------
    def _get_selected_features(self):
        """
        Returns the features selected by the SequentialFeatureSelector
        class (from the mlxtend.feature_selection library).
        """
        return list(self.selector.k_feature_names_)

    # -----------------------------------
    def get_summary(self):
        """
        Public method that returns the summary generated by the
        SequentialFeatureSelector class. This summary is a dictionary
        where each key represents a different run, which is associated
        with a secondary dictionary with all the relevant data regarding
        that particular run.

        :return: a dictionary where each key represents a different run,
            which is associated with a secondary dictionary with all the
            relevant data regarding that particular run.
        :rtype: dict
        """
        return self.selector.subsets_.copy()
