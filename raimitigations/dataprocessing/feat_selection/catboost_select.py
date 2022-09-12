from typing import Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, CatBoostRegressor, EShapCalcType, EFeaturesSelectionAlgorithm, Pool
import json

from .selector import FeatureSelection
from ..imputer import DataImputer, BasicImputer
from ..data_utils import get_cat_cols, err_float_01


class CatBoostSelection(FeatureSelection):
    """
    Concrete class that uses the ``CatBoost`` model and its feature's
    importance values to select the most important features. ``CatBoost``
    is a tree boosting method capable of handling categorical features.
    This method creates internally an importance score for each feature.
    This way, these scores can be used to perform feature selection. The
    ``CatBoost`` implementation (from the ``catboost`` lib) has already prepared
    a functionality for this purpose. The subclass :class:`~raimitigations.dataprocessing.CatBoostSelection` simply
    encapsulates all the complexities associated with this functionality and
    makes it easier for the user feature selection over a dataset.

    :param df: the data frame to be used during the fit method.
        This data frame must contain all the features, including the label
        column (specified in the  ``label_col`` parameter). This parameter is
        mandatory if  ``label_col`` is also provided. The user can also provide
        this dataset (along with the  ``label_col``) when calling the :meth:`fit`
        method. If df is provided during the class instantiation, it is not
        necessary to provide it again when calling :meth:`fit`. It is also possible
        to use the  ``X`` and  ``y`` instead of  ``df`` and  ``label_col``, although it is
        mandatory to pass the pair of parameters (X,y) or (df, label_col) either
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

    :param regression: if True and no estimator is provided, then create a default
        CatBoostRegressor. If False, a CatBoostClassifier is created instead. This parameter
        is ignored if an estimator is provided using the  ``estimator`` parameter;

    :param estimator: a :class:`~catboost.CatBoostClassifier` or :class:`~catboost.CatBoostRegressor` object that will be used
        for filtering the most important features. If no estimator is provided, a default
        :class:`~catboost.CatBoostClassifier` or :class:`~catboost.CatBoostRegressor` is used, where the latter is used if
        regression is set to True or if the type of the label column is float, while the
        former is used otherwise;

    :param in_place: indicates if the original dataset will be saved internally (``df_org``)
        or not. If True, then the feature selection transformation is saved over the
        original dataset. If False, the original dataset is saved separately (default
        value);

    :param catboost_log: if True, the default estimator will print logging
        messages during its training phase. If False, no log will be printed. This
        parameter is only used when  ``estimator`` is None, since this parameter is only used
        when creating the default classifier, which is not the case when the user specifies
        the classifier to be used. This parameter is automatically set to False if 'verbose'
        is False;

    :param catboost_plot: if True, uses ``CatBoost``'s plot feature: if running on a python
        notebook environment, an interactive plot will be created, which shows the loss
        function for both training and test sets, as well as the error obtained when removing
        an increasing number of features. If running on a script environment, a web interface
        will be opened showing this plot. If False, no plot will be generated.  This
        parameter is only used when  ``estimator`` is None, since this parameter is only used
        when creating the default classifier, which is not the case when the user specifies
        the classifier to be used;

    :param test_size: the size of the test set used to train the ``CatBoost`` method, which
        is used to create the importance score of each feature;

    :param cat_col: a list with the name or index of all categorical columns. These columns
        do not need to be encoded, since ``CatBoost`` does this encoding internally. If None, this
        list will be automatically set as a list with all categorical features found in the
        dataset;

    :param n_feat: the number of features to be selected. If None, then the following procedure
        is followed: (i) half of the existing features will be selected, (ii) after the feature
        selection method from ``CatBoost`` is executed, it generates a 'loss_graph', that indicates
        the loss function for each feature removed: the loss when 0 features were removed, the
        loss when 1 feature was removed, etc., up until half of the features were removed. With
        this loss graph, we get the number of features removed that resulted in the best loss
        function. We then set the features to be selected as the ones selected up until that
        point;

    :param fixed_cols: a list of column names or indices that should always be included in the
        set of selected features. Note that the number of columns included here must be smaller
        than n_feat, otherwise there is nothing for the class to do (that is:
        len(fixed_cols) < n_feat);

    :param algorithm: the algorithm used to do feature selection. ``CatBoost`` uses a Recursive
        Feature Selection approach, where each feature is removed at a time. The feature
        selected to be removed is the one with the least importance. The difference between
        each algorithm is how this importance is computed. This parameter can be one of the
        following string values: ['predict', 'loss', 'shap']. Here is a description of each of
        the algorithms allowed according to ``CatBoost``'s own documentation (text in double quotation
        marks were extracted from ``Catboost``'s official documentation):

            * **'predict':** uses the :class:`catboost.EFeaturesSelectionAlgorithm.RecursiveByPredictionValuesChange`
              algorithm. According to ``CatBoost``'s own documentation: "the fastest algorithm
              and the least accurate method (not recommended for ranking losses)" - "For each
              feature, PredictionValuesChange shows how much on average the prediction changes
              if the feature value changes. The bigger the value of the importance the bigger
              on average is the change to the prediction value, if this feature is changed.";
            * **'loss':** uses the :class:`catboost.EFeaturesSelectionAlgorithm.RecursiveByLossFunctionChange`
              algorithm. According to ``CatBoost``'s own documentation: "the optimal option
              according to accuracy/speed balance" - "For each feature the value represents the
              difference between the loss value of the model with this feature and without it.
              The model without this feature is equivalent to the one that would have been trained
              if this feature was excluded from the dataset. Since it is computationally expensive
              to retrain the model without one of the features, this model is built approximately
              using the original model with this feature removed from all the trees in the ensemble.
              The calculation of this feature importance requires a dataset and, therefore, the
              calculated value is dataset-dependent.";
            * **'shap':** uses the :class:`catboost.EFeaturesSelectionAlgorithm.RecursiveByShapValues` algorithm.
              According to ``CatBoost``'s own documentation: "the most accurate method.". For this
              algorithm, ``CatBoost`` uses Shap Values to determine the importance of each feature;

    :param steps: the number of times the model is trained. The greater the number of steps, the more
        accurate is the importance score of each feature;

    :param save_json: saves the summary generated by the ``CatBoost`` model.
        For more information on the data contained in this summary, please check
        ``CatBoost``'s official documentation:
        https://catboost.ai/en/docs/concepts/output-data_features-selection

    :param json_summary: the path and name of the json file created when saving the summary. This file
        is only saved when ``save_json`` is set to True;

    :param verbose: indicates whether internal messages should be printed or not.
    """

    PCT_FEAT_SEL = 0.5
    ALGORITHMS = ["predict", "loss", "shap"]

    # -----------------------------------

    def __init__(
        self,
        df: Union[pd.DataFrame, np.ndarray] = None,
        label_col: str = None,
        X: Union[pd.DataFrame, np.ndarray] = None,
        y: Union[pd.DataFrame, np.ndarray] = None,
        transform_pipe: list = None,
        regression: bool = None,
        estimator: Union[CatBoostClassifier, CatBoostRegressor] = None,
        in_place: bool = False,
        catboost_log: bool = True,
        catboost_plot: bool = False,
        test_size: float = 0.2,
        cat_col: list = None,
        n_feat: int = None,
        fixed_cols: list = None,
        algorithm: str = "loss",
        steps: int = 1,
        save_json: bool = False,
        json_summary: str = "cb_feat_summary.json",
        verbose: bool = True,
    ):
        super().__init__(df, label_col, X, y, transform_pipe, in_place, verbose)
        self.cat_col = cat_col
        self.n_feat = n_feat
        self.steps = steps
        self.test_size = test_size
        self.estimator = estimator
        self.catboost_log = catboost_log
        self.catboost_plot = catboost_plot
        self.fixed_cols = fixed_cols
        self.save_json = save_json
        self.json_summary = json_summary
        self.regression = regression
        self._check_test_size()
        self._set_algorithm(algorithm)
        self._set_estimator()

        if not self.verbose:
            self.catboost_log = False

    # -----------------------------------
    def _get_preprocessing_requirements(self):
        """
        Overridden method because CatBoost works with categorical variables. This means
        that it is not necessary to encode the categorical variables beforehand, which
        is a requirement in the base _get_preprocessing_requirements implemented in
        this class's parent (FeatureSelection).
        """
        requirements = {DataImputer: BasicImputer(verbose=self.verbose)}
        return requirements

    # -----------------------------------
    def _set_algorithm(self, algorithm: str):
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"ERROR: the 'algorithm' parameter must be one of the following: {self.ALGORITHMS}.")

        if algorithm == "predict":
            self.algorithm = EFeaturesSelectionAlgorithm.RecursiveByPredictionValuesChange
        elif algorithm == "loss":
            self.algorithm = EFeaturesSelectionAlgorithm.RecursiveByLossFunctionChange
        else:
            self.algorithm = EFeaturesSelectionAlgorithm.RecursiveByShapValues

    # -----------------------------------
    def _get_base_estimator(self):
        """
        Creates a default CatBoostClassifier or CatBoostRegressor. If the estimator
        has already been set, then do nothing.
        """
        if self.estimator is not None:
            return

        log_level = "Silent"
        if self.catboost_log:
            log_level = "Verbose"

        if self.regression:
            self.estimator = CatBoostRegressor(loss_function="RMSE", logging_level=log_level, cat_features=self.cat_col)
        else:
            self.estimator = CatBoostClassifier(
                loss_function="CrossEntropy", logging_level=log_level, cat_features=self.cat_col
            )

    # -----------------------------------
    def _set_estimator(self):
        """
        Sets the self.estimator attribute based on the estimator passed by
        the user through the estimator parameter. If estimator is None, then
        a default estimator is used (_get_base_estimator() method). Otherwise,
        the estimator is checked to see if it is a CatBoostClassifier or a
        CatBoostRegressor object. If not, an error is raised, since only these
        two objects are allowed.
        """
        if self.estimator is None:
            if self.regression is not None:
                self.estimator = self._get_base_estimator()
        else:
            if isinstance(self.estimator, CatBoostClassifier):
                self.regression = False
            elif isinstance(self.estimator, CatBoostRegressor):
                self.regression = True
            else:
                raise ValueError(
                    "ERROR: Expected 'estimator' to be a CatBoostClassifier or a CatBoostRegressor object."
                )

    # -----------------------------------
    def _check_regression(self):
        if self.regression is not None:
            return

        self.regression = False
        if "float" in self.y.dtype.name:
            self.regression = True

    # -----------------------------------
    def _check_test_size(self):
        """
        Checks if the value provided to the test_size parameter is valid.
        If it is not valid, an appropriate ValueError is raised informing
        the user of a possible explanation for the error.
        """
        err_float_01(self.test_size, "test_size")

    # -----------------------------------
    def _set_n_feat(self):
        """
        Sets the n_feat attribute with a default value if the value provided
        for the n_feat parameter was None. In this case, we set the internal
        variable select_best_n to True. This means that after running the
        _run_feat_selection() method, we will remove the features that resulted
        in the lowest loss value. We do this by checking the 'loss_values'
        returned by the select_features method from CatBoost.
        """
        self.select_best_n = False
        if self.n_feat is None and self.df is not None:
            self.select_best_n = True
            self.n_feat = int(len(self.df.columns) * self.PCT_FEAT_SEL)
            self.n_feat = max(1, self.n_feat)

        if self.n_feat is not None:
            error = False
            if type(self.n_feat) != int:
                error = True
            elif self.n_feat <= 0 or self.n_feat > self.df.shape[1]:
                error = True
            if error:
                raise ValueError(
                    f"ERROR: expected 'n_feat' to be an integer value between 1 and the number of columns of the targeted dataframe."
                )

    # -----------------------------------
    def _check_fixed_columns(self):
        """
        Checks for any errors or inconsistencies in the fixed_cols parameter.
        If any errors are encountered, an error is raised.
        """
        self.feat_to_select_from = self.df.columns
        if self.fixed_cols is None:
            return

        if type(self.fixed_cols) != list:
            raise ValueError(
                "ERROR: 'fixed_cols' must be a list. It should contain a list of column names or indices that "
                + "should be present in the set of selected features."
            )

        self.fixed_cols = self._check_error_col_list(self.df, self.fixed_cols, "fixed_cols")

        if len(self.fixed_cols) >= self.n_feat:
            raise ValueError(
                "ERROR: the number features to be selected (n_feat) must be greater than the number of fixed columns "
                + f"(fixed_cols). Instead, got n_feat = {self.n_feat} <= fixed_cols = {self.fixed_cols}."
            )

        self.feat_to_select_from = [col for col in self.df.columns if col not in self.fixed_cols]
        self.n_feat -= len(self.fixed_cols)

    # -----------------------------------
    def _set_cat_col(self):
        """
        Sets the list of categorical columns, which is a list with the name or
        index of all categorical columns. These columns do not need to be encoded,
        since CatBoost does this encoding internally. If None, this list will be
        automatically set as a list with all categorical features found in the
        dataset. Raises an error if the value provided to the cat_col parameter is
        not a list.
        """
        if self.cat_col is None and self.df is not None:
            self.cat_col = get_cat_cols(self.df)
            if self.cat_col == []:
                self.cat_col = None

        if self.cat_col is not None:
            if type(self.cat_col) != list:
                raise ValueError(
                    "ERROR: 'cat_col' must be a list. It should contain a list of column names from categorical features."
                )
            self.cat_col = self._check_error_col_list(self.df, self.cat_col, "cat_col")

    # -----------------------------------
    def _get_train_test_sets(self):
        if self.regression:
            train_X, test_X, train_y, test_y = train_test_split(self.df, self.y, test_size=self.test_size)
        else:
            train_X, test_X, train_y, test_y = train_test_split(
                self.df, self.y, test_size=self.test_size, stratify=self.y
            )
        feature_names = list(self.df.columns)
        train_pool = Pool(train_X, train_y, feature_names=feature_names, cat_features=self.cat_col)
        test_pool = Pool(test_X, test_y, feature_names=feature_names, cat_features=self.cat_col)
        return train_pool, test_pool

    # -----------------------------------
    def _add_fixed_cols(self):
        """
        Adds the fixed columns to the set of selected features. The result
        provided by CatBoost includes only the selected features from the
        pool of features allowed to be removed (that is, features not in
        in the fixed_cols parameter). This way, we must always add these
        columns after CatBoost selects the features.
        """
        if self.fixed_cols is None:
            return

        for col in self.fixed_cols:
            col_idx = self.df.columns.get_loc(col)
            self.summary["selected_features"].append(col_idx)
            self.summary["selected_features_names"].append(col)

    # -----------------------------------
    def _run_feat_selection(self):
        """
        Runs the feature selection method based on the importance of each feature
        computed by the CatBoost model. A summary with the results is saved in
        self.summary, which is then used by the _get_selected_features method
        to retrieve the selected features.
        """
        train_pool, test_pool = self._get_train_test_sets()
        log_level = "Silent"
        if self.catboost_log:
            log_level = "Verbose"
        self.feat_to_select_from = [self.df.columns.get_loc(col) for col in self.feat_to_select_from]
        self.summary = self.estimator.select_features(
            train_pool,
            eval_set=test_pool,
            features_for_select=self.feat_to_select_from,
            num_features_to_select=self.n_feat,
            steps=self.steps,  # more steps - more accurate selection
            algorithm=self.algorithm,
            shap_calc_type=EShapCalcType.Regular,  # can be Approximate, Regular and Exact
            train_final_model=False,  # to train model with selected features
            logging_level=log_level,
            plot=self.catboost_plot,
        )

        self._add_fixed_cols()

    # -----------------------------------
    def _get_best_feat_to_remove(self):
        """
        Selects the best set of features tested based on the results
        provided by CatBoost through the results summary returned
        by the CatBoost's internal method select_features().
        """
        if not self.select_best_n:
            return

        min_loss = -1
        best_index = -1
        for i, loss_value in enumerate(self.summary["loss_graph"]["loss_values"]):
            if min_loss < 0 or loss_value <= min_loss:
                min_loss = loss_value
                best_index = i

        removed_feat_indices = []
        removed_feat_names = []
        if best_index > 0:
            for i in range(best_index):
                removed_feat_indices.append(self.summary["eliminated_features"][i])
                removed_feat_names.append(self.summary["eliminated_features_names"][i])

        self.summary["eliminated_features"] = removed_feat_indices
        self.summary["eliminated_features_names"] = removed_feat_names
        self.summary["selected_features"] = [i for i in range(self.df.shape[1]) if i not in removed_feat_indices]
        self.summary["selected_features_names"] = [col for col in self.df.columns if col not in removed_feat_names]

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
            dict_format = json.dumps(self.summary, default=default)
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
        self._set_cat_col()
        self._get_base_estimator()
        self._set_n_feat()
        self._check_fixed_columns()
        self._run_feat_selection()
        self._get_best_feat_to_remove()
        self._save_json()

    # -----------------------------------
    def get_summary(self):
        """
        Public method that returns the summary generated by the ``CatBoost`` model.
        For more information on the data contained in this summary, please check
        ``CatBoost``'s official documentation:
        https://catboost.ai/en/docs/concepts/output-data_features-selection

        :return: a dictionary with the results obtained by the feature
            selection method.
        :rtype: dict
        """
        return self.summary.copy()

    # -----------------------------------
    def _get_selected_features(self):
        """
        Returns the features selected by the CatBoost model.
        """
        features = self.summary["selected_features_names"]
        if self.df is not None:
            if type(self.df.columns.values.tolist()[0]) == int:
                features = [int(feat) for feat in features]
        return features
