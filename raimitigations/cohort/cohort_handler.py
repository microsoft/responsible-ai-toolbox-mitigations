from abc import abstractmethod
from typing import Union
from copy import deepcopy

import pandas as pd
import numpy as np

from ..dataprocessing.data_processing import DataFrameInfo
from ..dataprocessing import DataProcessing
from .cohort_definition import CohortFilters


class CohortHandler(DataProcessing):
    """
    Abstract class that manages multiple cohorts.

    :param cohort_def: a list of cohort definitions or a dictionary of cohort definitions. A
        cohort condition is the same variable received by the ``cohort_definition`` parameter
        of the ``CohortDefinition`` class. When using a list of cohort definitions, the cohorts
        will be named automatically. For the dictionary of cohort definitions, the key used represents
        the cohort's name, and the value assigned to each key is given by that cohort's conditions.
        This parameter can't be used together with the ``cohort_col`` parameter. Only one these two
        parameters must be used at a time;

    :param cohort_col: a list of column names or indices, from which one cohort is created for each
        unique combination of values for these columns. This parameter can't be used together with
        the ``cohort_def`` parameter. Only one of these two parameters must be used at a time;

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
        cohort_def: Union[dict, list, str] = None,
        cohort_col: list = None,
        cohort_json_files: list = None,
        df: pd.DataFrame = None,
        label_col: str = None,
        X: pd.DataFrame = None,
        y: pd.DataFrame = None,
        verbose: bool = True,
    ):
        super().__init__(verbose)
        self.df_info = DataFrameInfo()
        self.y_info = DataFrameInfo()
        self.fitted = False
        self.cohorts = None
        self._cohort_names = []
        self._cohorts_compatible = None
        self._cohort_pipe = None
        self._use_baseline_cohorts = False
        self._set_df_mult(df, label_col, X, y)
        self._set_cohort_def(cohort_def, cohort_col, cohort_json_files)

    # -----------------------------------
    def _get_fit_input_type(self):
        return self.FIT_INPUT_XY

    # -----------------------------------
    @abstractmethod
    def _instantiate_cohort(self, cohort_definition: Union[list, str], name: str):
        """
        Create a cohort object (from the ``CohortDefinition`` class or any of
        its children classes) based on the specifications provided in the parameters.

        :param cohort_definition: a list of conditions or a string containing the path
            to a JSON file that has the list condition. Check the description of this parameter
            in the constructor method of the ``CohortDefinition`` class for more info.
        :param name: a string indicating the name of the cohort. This parameter may be accessed
            later using the ``name`` attribute.
        :return: an object from the ``CohortDefinition`` class or any of its children classes.
        :rtype: CohortDefinition
        """
        pass

    # -----------------------------------
    def _load_json_files(self, cohort_json_files: list):
        """
        Load a list of json files, convert the json structure to the list of
        conditions used by the CohortDefinition class, and then save these
        conditions in a list of cohort definitions. Finally, return the list
        of cohort definitions.

        :param cohort_json_files: a list with the name of the JSON files that
            contains the definition of each cohort. Each cohort is saved in a
            single JSON file, so the length of the ``cohort_json_files`` should
            be equal to the number of cohorts to be used.
        :return: a list of cohort definitions;
        :rtype: list
        """
        if type(cohort_json_files) != list:
            raise ValueError(
                "ERROR: the 'cohort_json_files' should be a list with the name of the JSON files that "
                + "contains the definition of each cohort. Expected a list, but instead got a "
                + f"{type(cohort_json_files)}."
            )
        cohort_def = []
        none_cohort = False
        for json_file in cohort_json_files:

            if json_file is None:
                if none_cohort:
                    raise ValueError("ERROR: only one 'None' value is allowed in the 'cohort_json_files' parameter.")
                name = "Remaining Instances"
                if name in self._cohort_names:
                    temp_name = deepcopy(name)
                    count = 0
                    while temp_name in self._cohort_names:
                        temp_name = f"{name} {count}"
                        count += 1
                    name = temp_name
                cohort = self._instantiate_cohort(None, name)
            else:
                cohort = self._instantiate_cohort(json_file)

            if cohort.name in self._cohort_names:
                raise ValueError(
                    f"ERROR: multiple cohorts are named '{cohort.name}'. Make sure that the each cohort has "
                    + "a unique name. The following name."
                )
            self._cohort_names.append(deepcopy(cohort.name))
            cohort_def.append(deepcopy(cohort.conditions))

        return cohort_def

    # -----------------------------------
    def _set_cohort_def(self, cohort_def: Union[dict, list], cohort_col: list, cohort_json_files: list):
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
            parameters must be used at a time. This parameter is ignored if ``cohort_json_files`` is provided;
        :param cohort_col: a list of column names or indices, from which one cohort is created for each
            unique combination of values for these columns. This parameter can't be used together with
            the ``cohort_def`` parameter. Only one these two parameters must be used at a time. This
            parameter is ignored if ``cohort_json_files`` is provided;
        :param cohort_json_files: a list with the name of the JSON files that contains the definition
            of each cohort. Each cohort is saved in a single JSON file, so the length of the
            ``cohort_json_files`` should be equal to the number of cohorts to be used.
        """
        if cohort_json_files is not None:
            cohort_def = self._load_json_files(cohort_json_files)
            self.cohort_col = cohort_col
            self.cohort_def = cohort_def
            return

        if cohort_def is None and cohort_col is None:
            raise ValueError(
                "ERROR: at least one of the following parameters must be provided: [cohort_def, cohort_col]."
            )
        if cohort_col is not None and cohort_def is not None:
            raise ValueError(
                "ERROR: only one of the following parameters must be provided (not both): [cohort_def, cohort_col]."
            )

        if cohort_def is not None:
            if not isinstance(cohort_def, (dict, list)):
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
            if type(cohort_col) != list or not cohort_col:
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
        if self.df_info.df is None:
            return

        # get the list of unique combination of values across all columns in 'cohort_col'
        subset = self._get_df_subset(self.df_info.df, self.cohort_col)
        unique_df = subset.groupby(self.cohort_col, dropna=False).size().reset_index().rename(columns={0: "count"})
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
            self._use_baseline_cohorts = True
            self.cohort_col = self._check_error_col_list(self.df_info.columns, self.cohort_col, "cohort_col")
            self._cohort_col_to_def()
            if self.cohort_def is None:
                return

        self.cohorts = []
        prev_cohort_def = []
        for i, cohort_def in enumerate(self.cohort_def):
            if cohort_def is None and i < len(self.cohort_def) - 1:
                raise ValueError("ERROR: only the last cohort is allowed to have a condition list assigned to 'None'.")
            cohort = self._instantiate_cohort(cohort_def, self._cohort_names[i])
            if cohort_def is None:
                cohort.create_query_remaining_instances_cohort(prev_cohort_def)
            else:
                prev_cohort_def.append(cohort_def)
            self.cohorts.append(cohort)

    # -----------------------------------
    def _check_intersection_cohorts(self, index_used: list):
        """
        Checks if there are any index that repeats in the ``index_used`` list.
        If that happens, this means that at least one instance of the dataset
        belongs to more than one cohort, which is not allowed.

        :param index_used: a list with the index of all instances that have a
            matching cohort.
        """
        set_index = list(set(index_used))
        if len(set_index) != len(index_used):
            raise ValueError("ERROR: some rows of the dataset belong to more than one cohort.")

    # -----------------------------------
    def _merge_cohort_datasets(self, cht_df: dict, org_index: list = None):
        """
        Merges all cohort subsets (after going through their own
        transformations) into a single dataset. After concatenating all
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
            final_df = pd.concat([final_df, df], axis=0)

        if org_index is not None:
            new_index = [index for index in org_index if index in final_df.index]
            final_df = final_df.reindex(new_index)
        else:
            final_df.index = [i for i in range(final_df.shape[0])]

        return final_df

    # -----------------------------------
    def _merge_cohort_predictions(self, cht_pred: dict, index_list: list, split_pred: bool):
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
        :param split_pred: if True, return a dictionary with the predictions
            for each cohort. If False, return a single predictions array;
        :return: an array with the predictions of all instances of the
            dataset, built from the predictions of each cohort.
        :rtype: np.ndarray
        """
        if split_pred:
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
    def save_cohorts(self, json_file_names: list = None):
        """
        Save the definition of each cohort in their respective JSON
        file, which means that one JSON file will be created for each
        cohort. The name of the JSON files created is provided through
        the 'json_file_names'. If no list of JSON file names is used
        (json_file_names = None), then a default list of JSON file
        names is created.

        :param json_file: a list of JSON file names. The first file name
            is used to save the first cohort, and so on. If not provided,
            a default list of file names is created.
        """
        if self.cohorts is None:
            raise ValueError(
                "ERROR: calling the save_cohorts() method before building the cohorts. To build the cohorts, "
                + "either pass a valid dataframe to the constructor of the CohortManager class, "
                + " or call the fit() method."
            )
        if json_file_names is None:
            json_file_names = [f"cohort_{i}.json" for i in range(len(self.cohorts))]
        elif type(json_file_names) != list:
            raise ValueError(
                "ERROR: the 'json_file_names' parameter must be a list of JSON file names (list of strings)."
            )

        for i, cohort in enumerate(self.cohorts):
            cohort.save(json_file_names[i])

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

                * ``X``: the subset of the features dataset;
                * ``y``: the subset of the label dataset. This key will only
                  be returned if the `y` dataset is passed in the method's
                  call.
        :rtype: dict
        """
        self._check_if_fitted()
        X = self._fix_col_transform(X)
        if y is not None:
            y = self._numpy_array_to_df(y)
            y.index = X.index

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
