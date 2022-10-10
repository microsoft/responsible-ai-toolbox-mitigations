from typing import Union
from copy import deepcopy

import pandas as pd
import numpy as np
import json

from ..dataprocessing import DataProcessing
from .cohort_definition import CohortDefinition, CohortFilters


class CohortManager(DataProcessing):
    """
    Concrete class that manages multiple cohort pipelines that are
    applied using the ``fit()``, ``transform()``, fit_resample()``,
    ``predict()``, and ``predict_proba()`` interfaces. The ``CohortManager``
    uses multiple ``CohortDefinition`` objects to control the filters
    of each cohort, while using transformation pipelines to control
    which transformations should be applied to each cohort.

    :param transform_pipe: the transformation pipeline to be used for each
        cohort. There are different ways to present this parameter:

            1. **An empty list or ``None``:** in this case, the ``CohortManager``
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

    :param cohort_def: a list of cohort definitions, a dictionary of cohort definitions, or
        the path to a JSON file containing the definition of all cohorts. A cohort condition
        is the same variable received by the ``cohort_definition`` parameter of the
        ``CohortDefinition`` class. When using a list of cohort definitions, the cohorts will
        be named automatically. For the dictionary of cohort definitions, the key used represents
        the cohort's name, and the value assigned to each key is given by that cohort's conditions.
        This parameter can't be used together with the ``cohort_col`` parameter. Only one these two
        parameters must be used at a time;

    :param cohort_col: a list of column names or indices, from which one cohort is created for each
        unique combination of values for these columns. This parameter can't be used together with
        the ``cohort_def`` parameter. Only one these two parameters must be used at a time;

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
        df: pd.DataFrame = None,
        label_col: str = None,
        X: pd.DataFrame = None,
        y: pd.DataFrame = None,
        verbose: bool = True,
    ):
        super().__init__(verbose)
        self.df = None
        self.df_org = None
        self.y = None
        self.fitted = False
        self.cohorts = None
        self._cohort_names = []
        self._cohort_index_list = []
        self._cohort_pipe = None
        self._pipe_has_transform = False
        self._pipe_has_fit_resample = False
        self._pipe_has_predict = False
        self._pipe_has_predict_proba = False
        self._cohorts_compatible = None
        self._transf_pipe = transform_pipe
        self._set_df_mult(df, label_col, X, y)
        self._set_cohort_def(cohort_def, cohort_col)
        self._build_cohorts()
        self._set_transforms()

    # -----------------------------------
    def _get_fit_input_type(self):
        return self.FIT_INPUT_XY

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
                self._pipe_has_transform = self._pipe_has_transform and _has_transf
                self._pipe_has_predict = self._pipe_has_predict and _has_pred
                self._pipe_has_predict_proba = self._pipe_has_predict_proba and _has_pred_proba
                self._pipe_has_fit_resample = self._pipe_has_fit_resample and has_resample
                self._cohort_pipe.append(_pipe)

    # -----------------------------------
    def _set_cohort_def(self, cohort_def: Union[dict, list], cohort_col: list):
        """
        Validate and set the cohort definitions based on the ``cohort_def`` and ``cohort_col``
        parameters. Checks if only one of these two parameters must be provided (the other one
        should be set to ``None``), and also if there are any errors or inconsistencies in these
        parameters. If not cohort names are provided, this method will also create a default
        name for each cohort.

        :param cohort_def: a list of cohort definitions, a dictionary of cohort definitions, or
            the path to a JSON file containing the definition of all cohorts. A cohort condition
            is the same variable received by the ``cohort_definition`` parameter of the
            ``CohortDefinition`` class. When using a list of cohort definitions, the cohorts will
            be named automatically. For the dictionary of cohort definitions, the key used represents
            the cohort's name, and the value assigned to each key is given by that cohort's conditions.
            This parameter can't be used together with the ``cohort_col`` parameter. Only one these two
            parameters must be used at a time;
        :param cohort_col: a list of column names or indices, from which one cohort is created for each
            unique combination of values for these columns. This parameter can't be used together with
            the ``cohort_def`` parameter. Only one these two parameters must be used at a time;
        """
        if cohort_def is None and cohort_col is None:
            raise ValueError(
                "ERROR: at least one of the following parameters must be provided: [cohort_def, cohort_col]."
            )
        if cohort_col is not None and cohort_def is not None:
            raise ValueError(
                "ERROR: only one of the following parameters must be provided (not both): [cohort_def, cohort_col]."
            )

        if cohort_def is not None:
            if type(cohort_def) == str:
                cohort_def = self._load(cohort_def)

            if type(cohort_def) != dict and type(cohort_def) != list:
                raise ValueError(
                    "ERROR: 'cohort_def' must be a dict or a list with the definition of all cohorts, or a string "
                    + "with the path of a valid json containing the definition of all cohorts."
                )

            if type(cohort_def) == dict:
                new_cohort_def = []
                for key in cohort_def:
                    new_cohort_def.append(cohort_def[key])
                    self._cohort_names.append(key)
                cohort_def = new_cohort_def
            elif type(cohort_def) == list:
                self._cohort_names = [f"cohort_{i}" for i in range(len(cohort_def))]
        else:
            if type(cohort_col) != list or cohort_col == []:
                raise ValueError(
                    "ERROR: cohort_col must be a list of column names that indicates "
                    + "the columns used to build the cohorts."
                )

        self.cohort_col = cohort_col
        self.cohort_def = cohort_def

    # -----------------------------------
    def _get_cohort_def_from_condition(self, cht_values: list):
        """
        Builds a cohort definition list, similar to the one used in the
        ``cohort_def`` parameter, based on a list of value tuples defining
        multiple cohorts. This list is created based on the ``cohort_col``
        parameter. These tuples are associated to the columns in
        ``cohort_col`` (also in the same order), and each value
        represents the condition for an instance to be considered from
        the cohort or not.

        :param cht_values: a list of values that defines a new cohort.
            The values in this list are associated with the columns in
            ``cohort_col`` (also in the same order), and each value
            represents the condition for an instance to be considered
            from the cohort or not.
        """
        cht_def = []
        for j, col_name in enumerate(self.cohort_col):
            cond_term = [col_name, CohortFilters.EQUAL, cht_values[j]]
            cht_def.append(cond_term)
            # if not the last cohort column name, add a 'and' term
            # between two consecutive conditions
            if j < len(self.cohort_col) - 1:
                cht_def.append(CohortFilters.AND)

        return cht_def

    # -----------------------------------
    def _cohort_col_to_def(self):
        """
        Converts the ``cohort_col`` to the ``cohort_def`` parameter.
        The former is a list of columns that should be used to create the
        cohorts. This way, all unique combination of values present are
        used to create a new cohort. These value combinations are inserted
        into a list of conditions, just like the one used by the ``cohort_def``
        parameter.
        """
        # can only convert 'cohort_col to the 'cohort_def'
        # if the dataset is already set
        if self.df is None:
            return

        # get the list of unique combination of values across all columns in 'cohort_col'
        subset = self._get_df_subset(self.df, self.cohort_col)
        unique_df = subset.groupby(self.cohort_col).size().reset_index().rename(columns={0: "count"})
        unique_df.drop(columns=["count"], inplace=True)
        unique_arr = unique_df.to_numpy()
        cohort_list = [list(unique_arr[i]) for i in range(unique_arr.shape[0])]

        # build the list of conditions based on the list of unique combinations of values
        self.cohort_def = []
        for i, cht in enumerate(cohort_list):
            name = f"cohort_{i}"
            cht_def = self._get_cohort_def_from_condition(cht)
            self.cohort_def.append(cht_def)
            self._cohort_names.append(name)

    # -----------------------------------
    def _build_cohorts(self):
        """
        Creates the ``CohortDefinition`` objects associated to each of the
        conditions in ``cohort_def``. If the cohorts have already been created,
        then return without making any changes. If ``cohort_def`` was not provided,
        then convert ``cohort_col`` to ``cohort_def``.
        """
        if self.cohorts is not None:
            return

        if self.cohort_def is None:
            self.cohort_col = self._check_error_col_list(self.df, self.cohort_col, "cohort_col")
            self._cohort_col_to_def()
            if self.cohort_def is None:
                return

        self.cohorts = []
        for i, cohort_def in enumerate(self.cohort_def):
            if cohort_def is None and i < len(self.cohort_def) - 1:
                raise ValueError("ERROR: only the last cohort is allowed to have a condition list assigned to 'None'.")
            cohort = CohortDefinition(cohort_def, self._cohort_names[i])
            self.cohorts.append(cohort)

    # -----------------------------------
    def save_conditions(self, json_file: str):
        """
        Save the definition of all cohorts into a JSON file.

        :param json_file: the path to the JSON file.
        """
        if self.cohorts is None:
            raise ValueError(
                "ERROR: calling the save_conditions() method before building the cohorts. To build the cohorts, "
                + "either pass a valid dataframe to the constructor of the CohortManager class, "
                + " or call the fit() method."
            )
        cht_dict = {}
        for cht in self.cohorts:
            cht_dict[cht.name] = cht.conditions

        with open(json_file, "w") as file:
            json.dump(cht_dict, file, indent=4)

    # -----------------------------------
    def _load(self, json_file: str):
        """
        Load the condition list of all cohorts from a JSON file.

        :param json_file: the path to the JSON file;
        :return: a dictionary with the list of conditions for each of
            the existing cohorts;
        :rtype: dict
        """
        with open(json_file, "r") as file:
            conditions = json.load(file)
        return conditions

    # -----------------------------------
    def _check_intersection_cohorts(self, index_used: list):
        """
        Checks if there are any index that repeats in the ``index_used`` list.
        If that happens, this means that at least one instance of the dataset
        belongs to more than one cohort, which is not allowed.

        :param index_used: a list with the index of all instances that have a
            mathcing cohort.
        """
        set_index = list(set(index_used))
        if len(set_index) != len(index_used):
            raise ValueError("ERROR: some rows of the dataset belong to more than one cohort.")

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
    def _merge_cohort_datasets(self, cht_df: dict, org_index: list = None):
        """
        Merges all cohort subsets (after going through their own
        transformations) into a single dataset. After concatanating all
        predictions into a single array, it is reordered so that it keeps
        the same order as the original dataset. The ``org_index`` is
        is used to determine this order.

        :param cht_df: a dictionary where each key is a cohort
            name, and the value associated to each key is a dataframe
            representing the subset that belongs to that cohort;
        :param org_index: the index list of the original dataset. This
            is used to define the order that the final predictions should
            be sorted so that it uses the same instance order used in the
            original dataset;
        :return: a dataset containing the subset of cohorts after their
            transformations. If ``org_index`` is provided, the resulting
            dataset will follow the same order as the original dataset;
        :rtype: pd.DataFrame
        """
        if not self._cohorts_compatible:
            self.print_message(
                "WARNING: the transformations used over the cohorts resulted in each cohort having different "
                + "columns. The transform() method will return a list of transformed subsets (one for each cohort)."
            )
            return cht_df

        final_df = None
        for df in cht_df.values():
            if final_df is None:
                final_df = df
            else:
                final_df = pd.concat([final_df, df], axis=0)

        if org_index is not None:
            new_index = [index for index in org_index if index in final_df.index]
            final_df = final_df.reindex(new_index)
        else:
            final_df.index = [i for i in range(final_df.shape[0])]

        return final_df

    # -----------------------------------
    def _merge_cohort_predictions(self, cht_pred: dict, index_list: list):
        """
        Merges all cohort predictions (after going through their own
        transformations) into a single prediction array. If ``org_index`` is
        provided, the resulting dataset will follow the same order as
        the original dataset.

        :param cht_df: a dictionary where each key is a cohort
            name, and the value associated to each key is a dataframe
            representing the subset that belongs to that cohort;
        :param org_index: the index list of the original dataset. When
            provided, the final dataset returned (with the subset of
            cohorts transformed) will be sorted so that it uses the same
            index list, that is, the returned array maintains the same
            instance order used in the original dataset;
        :return: an array with the predictions of all instances of the
            dataset, built from the predictions of each cohort.
        :rtype: np.ndarray
        """
        if not self._cohorts_compatible:
            return cht_pred

        final_pred = None
        for pred in cht_pred.values():
            if final_pred is None:
                final_pred = pred
            else:
                final_pred = np.concatenate((final_pred, pred), axis=0)

        new_index = np.argsort(index_list)
        final_pred = final_pred[new_index]

        return final_pred

    # -----------------------------------
    def _raise_missing_instances_error(self, df: pd.DataFrame, index_list: list):
        """
        Raises an error when there are instances that don't belong to any cohort.

        :param df: the dataset used for inference;
        :param index_list: the indices from df that are associated to an instance
            belonging to a cohort. From this, we can obtain the indices that don't
            belong to any cohort.
        """
        if len(list(df.index)) == len(index_list):
            return

        set_difference = set(df.index) - set(index_list)
        missing = list(set_difference)
        missing_subset = df.filter(items=missing, axis=0)
        raise ValueError(
            "ERROR: a subset of the instances passed to the transform(), predict(), or predict_proba() "
            + "doesn't fit into any of the existing cohorts.\n"
            + f"The subset is given as follows:\n{missing_subset}"
        )

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
            cht_x, cht_y, index_list = cohort.get_cohort_subset(self.df, self.y, index_used, return_index_list=True)
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
        self._raise_missing_instances_error(self.df, index_used)

        self.fitted = True
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
            self.print_message(
                "WARNING: a least one of the cohort pipelines doesn't have any transformations that "
                + "have a transform() method"
            )

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
    def _predict(self, X: Union[pd.DataFrame, np.ndarray], prob: bool = False):
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
        :param prob: if True, then ``predict_proba()`` is called. Otherwise, ``predict()``
            is called;
        :return: an array with the predictions of all instances of the
            dataset, built from the predictions of each cohort.
        :rtype: np.ndarray
        """
        if not prob and not self._pipe_has_predict:
            raise ValueError("ERROR: none of the objects in the transform_pipe parameter have a predict() method.")
        if prob and not self._pipe_has_predict_proba:
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
                        if not prob:
                            pred = tf.predict(cht_x)
                        else:
                            pred = tf.predict_proba(cht_x)
                pred_dict[cohort.name] = pred

        self._check_intersection_cohorts(index_used)
        self._raise_missing_instances_error(df, index_used)
        final_pred = self._merge_cohort_predictions(pred_dict, index_used)

        return final_pred

    # -----------------------------------
    def predict(self, X: Union[pd.DataFrame, np.ndarray]):
        """
        Calls the ``transform()`` method of all transformers in all pipelines, followed
        by the ``predict()`` method for the estimator (which is always the last object
        in the pipeline).

        :param X: contains only the features of the dataset to be transformed;
        :return: an array with the predictions of all instances of the
            dataset, built from the predictions of each cohort.
        :rtype: np.ndarray
        """
        final_pred = self._predict(X, prob=False)
        return final_pred

    # -----------------------------------
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]):
        """
        Calls the ``transform()`` method of all transformers in all pipelines, followed
        by the ``predict_proba()`` method for the estimator (which is always the last object
        in the pipeline).

        :param X: contains only the features of the dataset to be transformed;
        :return: an array with the predictions of all instances of the
            dataset, built from the predictions of each cohort.
        :rtype: np.ndarray
        """
        final_pred = self._predict(X, prob=True)
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
            cht_x, cht_y, index_list = cohort.get_cohort_subset(self.df, self.y, index_used, return_index_list=True)
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
        self._raise_missing_instances_error(self.df, index_used)
        final_df = self._merge_cohort_datasets(cht_df_dict)
        self.fitted = True

        if self.input_scheme == self.INPUT_XY:
            X_resample = final_df.drop(columns=[self.label_col_name])
            y_resample = final_df[self.label_col_name]
            return X_resample, y_resample

        return final_df

    # -----------------------------------
    def get_subsets(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray] = None,
        apply_transform: bool = False,
    ):
        """
        Fetches a dictionary with the subset associated to all of
        the existing cohorts and their label column (only if ``y``
        is provided). If ``apply_transform`` is set to True, then
        the returned subsets are transformed using the cohort's
        pipeline before being returned (similar to calling the
        ``transform()`` method).

        :param X: a dataset that has at least the columns used by
            the cohorts' filters (this means that the dataset may
            also have other columns not used by the filters);
        :param y: a dataset containing only the label column (`y`
            dataset). This parameter is optional, and it is useful
            when it is necessary to filter a feature dataset (`X`)
            and a label dataset (`y`), and get a list of subsets
            from ``X`` and ``y``;
        :param apply_transform: boolean value indicating if we want
            to apply the transformations pipeline used for each cohort
            or not. If True, this method will behave similarly to the
            `transform()` method, with the main difference being that
            this method will always return a list of subsets, even if
            the cohorts are compatible with each other.
        :return: a dictionary where the primary keys are the name of the
            cohorts, and the secondary keys are:

                * `X`: the subset of the features dataset;
                * `y`: the subset of the label dataset. This key will only
                  be returned if the `y` dataset is passed in the method's
                  call.
        :rtype: dict
        """
        self._check_if_fitted()
        X = self._fix_col_transform(X)

        index_used = []
        out_dict = {}
        for i, cohort in enumerate(self.cohorts):
            if y is None:
                cht_X, index_list = cohort.get_cohort_subset(X, y, index_used, return_index_list=True)
            else:
                cht_X, cht_y, index_list = cohort.get_cohort_subset(X, y, index_used, return_index_list=True)
            index_used += index_list

            if apply_transform and not cht_X.empty:
                for tf in self._cohort_pipe[i]:
                    has_transf = self.obj_has_method(tf, "transform")
                    if not has_transf:
                        break
                    cht_X = tf.transform(cht_X)

            if y is None:
                out_dict[cohort.name] = {"X": cht_X}
            else:
                out_dict[cohort.name] = {"X": cht_X, "y": cht_y}

        self._check_intersection_cohorts(index_used)
        self._raise_missing_instances_error(X, index_used)

        return out_dict

    # -----------------------------------
    def get_queries(self):
        """
        Returns a dictionary with one key for each cohort's name, where
        each key is assigned to the pandas query used for filtering the
        instances that belongs to the cohort.

        :return: a dictionary containing the pandas queries used by each
            of the existing cohorts.
        :rtype: dict
        """
        if self.cohorts is None:
            raise ValueError("ERROR: call the fit() method before fetching the queries.")
        queries = {}
        for cht in self.cohorts:
            queries[cht.name] = cht.query
        return queries
