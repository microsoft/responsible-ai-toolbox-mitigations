from typing import Union
from copy import deepcopy

import pandas as pd
import numpy as np

from ..dataprocessing import DataProcessing
from .cohort_handler import CohortHandler
from .cohort_definition import CohortDefinition


class CohortManager(CohortHandler):
    """
    Concrete class that manages multiple cohort pipelines that are
    applied using the ``fit()``, ``transform()``, ``fit_resample()``,
    ``predict()``, and ``predict_proba()`` interfaces. The ``CohortManager``
    uses multiple ``CohortDefinition`` objects to control the filters
    of each cohort, while using transformation pipelines to control
    which transformations should be applied to each cohort.

    :param transform_pipe: the transformation pipeline to be used for each
        cohort. There are different ways to present this parameter:

            1. **An empty list or None:** in this case, the ``CohortManager``
               won't apply any transformations over the dataset. The ``transform()``
               method will simply return the dataset provided;
            2. **A single transformer:** in this case, this single transformer is
               placed in a list (a list with a single transformer), which is then
               replicated such that each cohort has its own list of transformations
               (pipeline);
            3. **A list of transformers:** in this case, this pipeline is replicated
               for each cohort;
            4. **A list of pipelines:** a list of pipelines is basically a list of
               lists of transformations. In this case, the list of pipelines should have
               one pipeline for each cohort created, that is, the length of the
               ``transform_pipe`` parameter should be the same as the number of cohorts
               created. The pipelines will be assigned to each cohort following the same
               order as the ``cohort_def`` parameter (depicted in the following example);

    :param cohort_def: a list of cohort definitions or a dictionary of cohort definitions.
        A cohort condition is the same variable received by the ``cohort_definition`` parameter
        of the ``CohortDefinition`` class. When using a list of cohort definitions, the cohorts
        will be named automatically. For the dictionary of cohort definitions, the key used represents
        the cohort's name, and the value assigned to each key is given by that cohort's conditions.
        This parameter can't be used together with the ``cohort_col`` parameter. Only one these two
        parameters must be used at a time. This parameter is ignored if ``cohort_json_files`` is
        provided;

    :param cohort_col: a list of column names or indices, from which one cohort is created for each
        unique combination of values for these columns. This parameter can't be used together with
        the ``cohort_def`` parameter. Only one these two parameters must be used at a time. This
        parameter is ignored if ``cohort_json_files`` is provided;

    :param cohort_json_files: a list with the name of the JSON files that contains the definition
        of each cohort. Each cohort is saved in a single JSON file, so the length of the
        ``cohort_json_files`` should be equal to the number of cohorts to be used.

    :param df: the data frame to be used during the fit method.
        This data frame must contain all the features, including the label
        column (specified in the  ``label_col`` parameter). This parameter is
        mandatory if  ``label_col`` is also provided. The user can also provide
        this dataset (along with the  ``label_col``) when calling the :meth:`fit`
        method. If df is provided during the class instantiation, it is not
        necessary to provide it again when calling :meth:`fit`. It is also possible
        to use the  ``X`` and  ``y`` instead of ``df`` and ``label_col``, although it is
        mandatory to pass the pair of parameters (X,y) or (df, label_col) either
        during the class instantiation or during the :meth:`fit` method;

    :param label_col: the name or index of the label column. This parameter is
        mandatory if ``df`` is provided;

    :param X: contains only the features of the original dataset, that is, does not
        contain the label column. This is useful if the user has already separated
        the features from the label column prior to calling this class. This parameter
        is mandatory if ``y`` is provided;

    :param y: contains only the label column of the original dataset.
        This parameter is mandatory if ``X`` is provided;

    :param verbose: indicates whether internal messages should be printed or not.
    """

    # -----------------------------------
    def __init__(
        self,
        transform_pipe: list = None,
        cohort_def: Union[dict, list, str] = None,
        cohort_col: list = None,
        cohort_json_files: list = None,
        df: pd.DataFrame = None,
        label_col: str = None,
        X: pd.DataFrame = None,
        y: pd.DataFrame = None,
        verbose: bool = True,
    ):
        self._pipe_has_transform = False
        self._pipe_has_fit_resample = False
        self._pipe_has_predict = False
        self._pipe_has_predict_proba = False
        self._transf_pipe = transform_pipe
        super().__init__(cohort_def, cohort_col, cohort_json_files, df, label_col, X, y, verbose)
        self._build_cohorts()
        self._set_transforms()

    # -----------------------------------
    def _validate_transforms(self, transform_pipe: list):
        """
        Validates the transform pipeline provided in the constructor method.
        This method checks which methods should be unlocked for the
        ``CohortManager`` based on the pipelines: ``fit``, ``transform``,
        ``predict``, ``predict_proba``, ``fit_resample``. Also, checks for errors
        and inconsistencies in the pipeline.

        :param transform_pipe: the transformation pipeline to be used for each
            cohort. For more details, check the documentation of this parameter in
            the constructor method of this class.
        :return: a tuple containing (i) the updated transform pipeline, (ii) a flag
            indicating if the pipeline has a ``transform`` operation, (iii) a flag
            indicating if the pipeline has a ``predict`` operation, (iv) a flag
            indicating if the pipeline has a ``predict_proba`` operation, (v) a flag
            indicating if the pipeline has a ``fit_resample`` operation, in that order.
        :rtype: tuple
        """
        has_fit_resample = False
        has_transform = False
        has_predict = False
        has_predict_proba = False

        if transform_pipe is None:
            transform_pipe = []

        if type(transform_pipe) != list:
            transform_pipe = [transform_pipe]

        for i, transform in enumerate(transform_pipe):
            # all objects in the pipeline must have a fit() method
            class_name = transform.__class__.__name__
            has_fit = self.obj_has_method(transform, "fit")
            _has_fit_resample = self.obj_has_method(transform, "fit_resample")
            if _has_fit_resample:
                has_fit_resample = True
            if not has_fit and not _has_fit_resample:
                raise ValueError(
                    f"ERROR: the transform from class {class_name} passed to the "
                    + f"transform_pipe parameter does not have a fit() or a fit_resample() method."
                )

            _has_transf = self.obj_has_method(transform, "transform")
            if _has_transf:
                has_transform = True
            # only the last object is allowed to not have a transform() method
            if not _has_transf and i < len(transform_pipe) - 1:
                raise ValueError(
                    "ERROR: only the last object in the transform_pipe parameter is allowed to not have "
                    + f"a transform() method, but the object in position {i}, from class {class_name}, doesn't "
                    + "have a transform() method."
                )

            # check if the last object has a predict() or predict_proba() method.
            # only the last object is allowed to have these methods.
            _has_pred = self.obj_has_method(transform, "predict")
            _has_pred_proba = self.obj_has_method(transform, "predict_proba")
            if _has_pred:
                has_predict = True
            if _has_pred_proba:
                has_predict_proba = True
            if (_has_pred or _has_pred_proba) and i < len(transform_pipe) - 1:
                raise ValueError(
                    "ERROR: only the last object in the transform_pipe parameter is allowed to have "
                    + f"a predict() or a predict_proba() method, but the object in position {i}, from "
                    + f"class {class_name}, has one of these methods."
                )

            # special checks for classes that inherits from the DataProcessing class
            if isinstance(transform, DataProcessing):
                # check if the transformer works for the CohortManager class
                if not transform._works_with_cohort_manager():
                    raise ValueError(
                        "ERROR: one of the transformers in the transform_pipe parameter, from class "
                        + f"{class_name}, is not allowed. "
                    )

                # check if the transformer creates compatible cohorts or not
                if not transform._is_cohort_merging_compatible():
                    self._cohorts_compatible = False

        if has_fit_resample and (has_transform or has_predict or has_predict_proba):
            raise ValueError(
                "ERROR: the transform pipeline cannot mix transformers with a fit_resample() with others "
                + "with a transform(), predict(), or predict_proba(). If one of the transformers have "
                + "a fit_resample() method, then all transformers must implement only this same method."
            )

        return transform_pipe, has_transform, has_predict, has_predict_proba, has_fit_resample

    # -----------------------------------
    def _set_transforms(self):
        """
        Validates the transform pipeline provided in the constructor method.
        This method checks if the transform pipeline represents a single pipeline
        (in which case, the pipeline should be replicated to each cohort), a list
        of pipelines (one for each cohort), or an empty pipeline.
        """
        if self._cohort_pipe is not None or self.cohorts is None:
            return

        transform_pipe = self._transf_pipe
        if transform_pipe is None:
            transform_pipe = []

        if type(transform_pipe) != list:
            transform_pipe = [transform_pipe]

        self._cohort_pipe = []
        if len(transform_pipe) == 0:
            self._cohort_pipe = [[] for _ in range(len(self.cohorts))]
        elif type(transform_pipe[0]) != list:
            _pipe, _has_transf, _has_pred, _has_pred_proba, has_resample = self._validate_transforms(transform_pipe)
            self._pipe_has_transform = _has_transf
            self._pipe_has_predict = _has_pred
            self._pipe_has_predict_proba = _has_pred_proba
            self._pipe_has_fit_resample = has_resample
            for _ in range(len(self.cohorts)):
                cht_pipe = [deepcopy(tf) for tf in _pipe]
                self._cohort_pipe.append(cht_pipe)
        else:
            if len(transform_pipe) != len(self.cohorts):
                raise ValueError(
                    "ERROR: the list of transform pipelines should have the same length as the list of cohorts. "
                    + f"However, there are {len(transform_pipe)} different pipelines, while there are "
                    + f"{len(self.cohorts)} cohorts."
                )
            for transf_pipe in transform_pipe:
                _pipe, _has_transf, _has_pred, _has_pred_proba, has_resample = self._validate_transforms(transf_pipe)
                self._pipe_has_transform = self._pipe_has_transform or _has_transf
                self._pipe_has_predict = self._pipe_has_predict or _has_pred
                self._pipe_has_predict_proba = self._pipe_has_predict_proba or _has_pred_proba
                self._pipe_has_fit_resample = self._pipe_has_fit_resample or has_resample
                self._cohort_pipe.append(_pipe)

    # -----------------------------------
    def _instantiate_cohort(self, cohort_definition: Union[list, str], name: str = "cohort"):
        """
        Create a ``CohortDefinition`` based on the specifications provided in the parameters.

        :param cohort_definition: a list of conditions or a string containing the path
            to a JSON file that has the list condition. Check the description of this parameter
            in the constructor method of the ``CohortDefinition`` class for more info.
        :param name: a string indicating the name of the cohort. This parameter may be accessed
            later using the ``name`` attribute.
        :return: a ``CohortDefinition`` object.
        :rtype: CohortDefinition
        """
        return CohortDefinition(cohort_definition, name)

    # -----------------------------------
    def _check_compatibility_between_cohorts(self, cht_df_dict: dict):
        """
        Check if all cohorts are compatible between each other after
        applying their transformations. Two cohorts are considered
        compatible when they have the same number of columns and the
        exact same column names.

        :param cht_df_dict: a dictionary where each key is a cohort
            name, and the value associated to each key is a dataframe
            representing the subset that belongs to that cohort.
        """
        if self._cohorts_compatible is None:
            self._cohorts_compatible = True
        cht_names = list(cht_df_dict.keys())
        columns = cht_df_dict[cht_names[0]].columns
        for i in range(1, len(cht_names)):
            cht = cht_df_dict[cht_names[i]]
            if len(cht.columns) != len(columns):
                self._cohorts_compatible = False
                break

            diff_col = [col for col in columns if col not in cht.columns]
            if len(diff_col) > 0:
                self._cohorts_compatible = False
                break

    # -----------------------------------
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray] = None,
        y: Union[pd.Series, np.ndarray] = None,
        df: pd.DataFrame = None,
        label_col: str = None,
    ):
        """
        Calls the ``fit()`` method of all transformers in all pipelines. Each cohort
        has its own pipeline. This way, the following steps are executed: (i) iterate
        over each cohort, (ii) filter the dataset (``X`` or ``df``) using each cohort's
        filter, (iii) cycle through each of the transformers in the cohort's pipeline
        and call the transformer's ``fit()`` method, (iv) after fitting the transformer,
        call its ``transform()`` method to get the updated subset, which is then used in
        the ``fit()`` call of the following transformer. Finally, check if all instances
        belong to only a single cohort.

        :param X: contains only the features of the original dataset, that is, does not
            contain the label column;
        :param y: contains only the label column of the original dataset;
        :param df: the full dataset;
        :param label_col: the name or index of the label column;

        Check the documentation of the _set_df_mult method (DataProcessing class)
        for more information on how these parameters work.
        """
        self._set_df_mult(df, label_col, X, y, require_set=True)
        self._build_cohorts()
        self._set_transforms()
        index_used = []
        cht_df_dict = {}
        for i, cohort in enumerate(self.cohorts):
            cht_x, cht_y, index_list = cohort.get_cohort_subset(
                self.df_info.df, self.y_info.df, index_used, return_index_list=True
            )
            index_used += index_list
            if cht_x.empty:
                raise ValueError(
                    f"ERROR: no valid instances found for cohort {cohort.name} in the fit() method. The query used "
                    + f"by this cohort is the following: {cohort.query}."
                )
            for tf in self._cohort_pipe[i]:
                tf.fit(cht_x, cht_y)
                has_transf = self.obj_has_method(tf, "transform")
                if not has_transf:
                    break
                cht_x = tf.transform(cht_x)
            cht_df_dict[cohort.name] = cht_x

        self._check_compatibility_between_cohorts(cht_df_dict)
        self._check_intersection_cohorts(index_used)
        self._raise_missing_instances_error(self.df_info.df, index_used)

        self.fitted = True
        self.df_info.clear_df_mem()
        self.y_info.clear_df_mem()
        return self

    # -----------------------------------
    def transform(self, X: Union[pd.DataFrame, np.ndarray]):
        """
        Calls the ``transform()`` method of all transformers in all pipelines. Each cohort
        has its own pipeline. This way, the following steps are executed: (i) iterate
        over each cohort, (ii) filter the dataset ``X`` using eachcohort's filter, (iii)
        cycle through each of the transformers in the cohort's pipeline and call the
        transformer's ``transform()`` method, which returns a new transformed subset,
        that is then used in the ``transform()`` call of the following transformer.
        Finally, check if all instances belong to only a single cohort, and merge all cohort
        subsets into a single dataset.

        :param X: contains only the features of the dataset to be transformed;
        :return: a dataset containing the transformed instances of all cohorts.
        :rtype: pd.DataFrame
        """
        if not self._pipe_has_transform:
            self.print_message("WARNING: none of the objects in the transform_pipe parameter have a transform() method")
            return X

        self._check_if_fitted()
        df = self._fix_col_transform(X)

        index_used = []
        cht_df_dict = {}
        org_index = df.index.copy()
        for i, cohort in enumerate(self.cohorts):
            cht_x, index_list = cohort.get_cohort_subset(df, y=None, index_used=index_used, return_index_list=True)
            index_used += index_list
            if not cht_x.empty:
                for tf in self._cohort_pipe[i]:
                    has_transf = self.obj_has_method(tf, "transform")
                    if not has_transf:
                        break
                    cht_x = tf.transform(cht_x)
                cht_df_dict[cohort.name] = cht_x

        self._check_intersection_cohorts(index_used)
        self._raise_missing_instances_error(df, index_used)
        final_df = self._merge_cohort_datasets(cht_df_dict, org_index)

        return final_df

    # -----------------------------------
    def _predict(self, X: Union[pd.DataFrame, np.ndarray], probability: bool = False, split_pred: bool = False):
        """
        Calls the ``transform()`` method of all transformers in all pipelines, followed
        by the ``predict()`` or ``predict_proba()`` method for the estimator (which is
        always the last object in the pipeline). Each cohort has its own pipeline. This way,
        the following steps are executed: (i) iterate over each cohort, (ii) filter the
        dataset ``X`` using each cohort's filter, (iii) cycle through each of the transformers
        in the cohort's pipeline and call the transformer's ``transform()`` method, which
        returns a new transformed subset, that is then used in the ``transform()`` call of the
        following transformer, (iv) finally, after cycling through all transformers that are
        not estimators, call the ``predict()`` or ``predict_proba()`` method of the estimator
        using the transformed subset as input. Finally, check if all instances belong to only
        a single cohort, and merge all cohort predictions into a single array.

        :param X: contains only the features of the dataset to be transformed;
        :param probability: if True, then ``predict_proba()`` is called. Otherwise, ``predict()``
            is called;
        :param split_pred: if True, return a dictionary with the predictions
            for each cohort. If False, return a single predictions array;
        :return: an array with the predictions of all instances of the dataset, built from the
            predictions of each cohort, or a dictionary with the predictions for each cohort;
        :rtype: np.ndarray or dict
        """
        if not probability and not self._pipe_has_predict:
            raise ValueError("ERROR: none of the objects in the transform_pipe parameter have a predict() method.")
        if probability and not self._pipe_has_predict_proba:
            raise ValueError(
                "ERROR: none of the objects in the transform_pipe parameter have a predict_proba() method."
            )

        self._check_if_fitted()
        df = self._fix_col_transform(X)
        df = df.reset_index(drop=True)

        index_used = []
        pred_dict = {}
        for i, cohort in enumerate(self.cohorts):
            cht_x, index_list = cohort.get_cohort_subset(df, y=None, index_used=index_used, return_index_list=True)
            index_used += index_list
            if not cht_x.empty:
                for tf in self._cohort_pipe[i]:
                    has_transf = self.obj_has_method(tf, "transform")
                    if has_transf:
                        cht_x = tf.transform(cht_x)
                    else:
                        if not probability:
                            pred = tf.predict(cht_x)
                        else:
                            pred = tf.predict_proba(cht_x)
                pred_dict[cohort.name] = pred

        self._check_intersection_cohorts(index_used)
        self._raise_missing_instances_error(df, index_used)
        final_pred = self._merge_cohort_predictions(pred_dict, index_used, split_pred)

        return final_pred

    # -----------------------------------
    def predict(self, X: Union[pd.DataFrame, np.ndarray], split_pred: bool = False):
        """
        Calls the ``transform()`` method of all transformers in all pipelines, followed
        by the ``predict()`` method for the estimator (which is always the last object
        in the pipeline).

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
        by the ``predict_proba()`` method for the estimator (which is always the last object
        in the pipeline).

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
    def fit_resample(
        self,
        X: Union[pd.DataFrame, np.ndarray] = None,
        y: Union[pd.DataFrame, np.ndarray] = None,
        df: Union[pd.DataFrame, np.ndarray] = None,
        rebalance_col: str = None,
    ):
        """
        Calls the ``fit_resample()`` method of all transformers in all pipelines. Each
        cohort has its own pipeline. This way, the following steps are executed:
        (i) iterate over each cohort, (ii) filter the dataset (``X`` or ``df``) using
        each cohort's filter, (iii) cycle through each of the transformers in the cohort's
        pipeline and call the transformer's ``fit_resample()`` method, (iv) after resampling
        using the current transformer, save the new subset and use it when calling the
        ``fit_resample()`` of the following transformer. Finally, check if all instances
        belong to only a single cohort.

        :param X: contains only the features of the original dataset, that is, does
            not contain the column used for rebalancing. This is useful if the user has
            already separated the features from the label column prior to calling this
            class. This parameter is mandatory if  ``y`` is provided;
        :param y: contains only the rebalance column of the original dataset. The rebalance
            operation is executed based on the data distribution of this column. This parameter
            is mandatory if  ``X`` is provided;
        :param df: the dataset to be rebalanced, which is used during the :meth:`fit` method.
            This data frame must contain all the features, including the rebalance
            column (specified in the  ``rebalance_col`` parameter). This parameter is
            mandatory if  ``rebalance_col`` is also provided. The user can also provide
            this dataset (along with the  ``rebalance_col``) when calling the :meth:`fit`
            method. If ``df`` is provided during the class instantiation, it is not
            necessary to provide it again when calling :meth:`fit`. It is also possible
            to use the  ``X`` and  ``y`` instead of  ``df`` and  ``rebalance_col``, although it is
            mandatory to pass the pair of parameters (X,y) or (df, rebalance_col) either
            during the class instantiation or during the :meth:`fit` method;
        :param rebalance_col: the name or index of the column used to do the rebalance
            operation. This parameter is mandatory if  ``df`` is provided.
        :return: the resampled dataset.
        :rtype: pd.DataFrame
        """
        self._set_df_mult(df, rebalance_col, X, y, require_set=True)
        self._build_cohorts()
        self._set_transforms()

        if not self._pipe_has_fit_resample:
            raise ValueError("ERROR: none of the objects in the transform_pipe parameter have a fit_resample() method.")

        index_used = []
        cht_df_dict = {}
        for i, cohort in enumerate(self.cohorts):
            cht_x, cht_y, index_list = cohort.get_cohort_subset(
                self.df_info.df, self.y_info.df, index_used, return_index_list=True
            )
            index_used += index_list
            if not cht_x.empty:
                for tf in self._cohort_pipe[i]:
                    has_fit_resample = self.obj_has_method(tf, "fit_resample")
                    if has_fit_resample:
                        cht_x, cht_y = tf.fit_resample(cht_x, cht_y)
                cht_df = pd.concat([cht_x, cht_y], axis=1)
                cht_df_dict[cohort.name] = cht_df

        self._check_compatibility_between_cohorts(cht_df_dict)
        self._check_intersection_cohorts(index_used)
        self._raise_missing_instances_error(self.df_info.df, index_used)
        final_df = self._merge_cohort_datasets(cht_df_dict)
        self.fitted = True

        if self.input_scheme == self.INPUT_XY:
            if type(final_df) == dict:
                final_df_y = {}
                for cht_name in final_df.keys():
                    final_df_y[cht_name] = final_df[cht_name][self.label_col_name]
                    final_df[cht_name] = final_df[cht_name].drop(columns=[self.label_col_name])
                return final_df, final_df_y
            X_resample = final_df.drop(columns=[self.label_col_name])
            y_resample = final_df[self.label_col_name]
            return X_resample, y_resample

        return final_df
