from typing import Union
import itertools
from copy import deepcopy

import pandas as pd
import numpy as np
import json

from ..dataprocessing import DataProcessing
from .cohort_definition import CohortDefinition, CohortFilters


class CohortManager(DataProcessing):

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
        self._cohort_pipe = []
        self._pipe_has_transform = False
        self._pipe_has_predict = False
        self._pipe_has_predict_proba = False
        self._cohorts_compatible = None
        self._set_df_mult(df, label_col, X, y)
        self._set_transforms(transform_pipe)
        self._set_cohort_def(cohort_def, cohort_col)
        self._build_cohorts()

    # -----------------------------------

    def _get_fit_input_type(self):
        return self.FIT_INPUT_XY

    # -----------------------------------

    def _set_transforms(self, transform_pipe: list):
        if transform_pipe is None:
            transform_pipe = []

        if type(transform_pipe) != list:
            transform_pipe = [transform_pipe]

        for i, transform in enumerate(transform_pipe):
            # all objects in the pipeline must have a fit() method
            class_name = transform.__class__.__name__
            has_fit = self.obj_has_method(transform, "fit")
            if not has_fit:
                raise ValueError(
                    (
                        f"ERROR: the transform from class {class_name} passed to the "
                        f"transform_pipe parameter does not have a fit() method."
                    )
                )

            has_transf = self.obj_has_method(transform, "transform")
            if has_transf:
                self._pipe_has_transform = True
            # only the last object is allowed to not have a transform() method
            if not has_transf and i < len(transform_pipe) - 1:
                raise ValueError(
                    (
                        "ERROR: only the last object in the transform_pipe parameter is allowed to not have "
                        f"a transform() method, but the object in position {i}, from class {class_name}, doesn't "
                        "have a transform() method."
                    )
                )

            # check if the last object has a predict() or predict_proba() method.
            # only the last object is allowed to have these methods.
            has_predict = self.obj_has_method(transform, "predict")
            has_predict_proba = self.obj_has_method(transform, "predict_proba")
            if has_predict:
                self._pipe_has_predict = True
            if has_predict_proba:
                self._pipe_has_predict_proba = True
            if (has_predict or has_predict_proba) and i < len(transform_pipe) - 1:
                raise ValueError(
                    (
                        "ERROR: only the last object in the transform_pipe parameter is allowed to have "
                        f"a predict() or a predict_proba() method, but the object in position {i}, from "
                        f"class {class_name}, has one of these methods."
                    )
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

        self.transform_pipe = transform_pipe

    # -----------------------------------

    def _set_cohort_def(self, cohort_def: Union[dict, list], cohort_col: list):
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

    def _get_cohort_def_from_condition(self, cht_values: tuple):
        """
        Builds a cohort definition list, similar to the one used in the
        ``cohort_def`` parameter, based on a list of value tuples defining
        multiple cohorts. This list is created based on the ``cohort_col``
        parameter. These tuples are associated to the columns in
        ``cohort_col`` (also in the same order), and each value
        represents the condition for an instance to be considered from
        the cohort or not.

        :param cht_values: a tuple of values that defines a new cohort.
            The values in this tuple are associated with the columns in
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
        if self.df is None:
            return

        sets = []
        for col in self.cohort_col:
            subset = self._get_df_subset(self.df, [col])
            values = subset.iloc[:, 0].unique()
            sets.append(values)

        cohort_list = list(itertools.product(*sets))

        self.cohort_def = []
        for i, cht in enumerate(cohort_list):
            name = f"cohort_{i}"
            cht_def = self._get_cohort_def_from_condition(cht)
            self.cohort_def.append(cht_def)
            self._cohort_names.append(name)

    # -----------------------------------

    def _build_cohorts(self):
        if self.cohorts is not None:
            return

        if self.cohort_def is None:
            self._cohort_col_to_def()
            if self.cohort_def is None:
                return

        self.cohorts = []
        for i, cohort_def in enumerate(self.cohort_def):
            cohort = CohortDefinition(cohort_def, self._cohort_names[i])
            self.cohorts.append(cohort)
            cohort_pipe = [deepcopy(tf) for tf in self.transform_pipe]
            self._cohort_pipe.append(cohort_pipe)

    # -----------------------------------

    def save(self, json_file: str):
        if self.cohorts is None:
            raise ValueError(
                (
                    "ERROR: calling the save() method before building the cohorts. To build the cohorts, either pass "
                    "a valid dataframe to the constructor of the CohortManager class, or call the fit() method."
                )
            )
        cht_dict = {}
        for cht in self.cohorts:
            cht_dict[cht.name] = cht.conditions

        with open(json_file, "w") as file:
            json.dump(cht_dict, file, indent=4)

    # -----------------------------------

    def _load(self, json_file: str):
        with open(json_file, "r") as file:
            conditions = json.load(file)
        return conditions

    # -----------------------------------

    def _check_intersection_cohorts(self, index_used: list):
        set_index = list(set(index_used))
        if len(set_index) != len(index_used):
            raise ValueError("ERROR: some rows of the dataset belong to more than one cohort.")

    # -----------------------------------

    def _check_compatibility_between_cohorts(self, cht_df: list):
        if self._cohorts_compatible is None:
            self._cohorts_compatible = True
        columns = cht_df[0].columns
        for i in range(1, len(cht_df)):
            if len(cht_df[i].columns) != len(columns):
                self._cohorts_compatible = False
                break

            diff_col = [col for col in columns if col not in cht_df[i].columns]
            if len(diff_col) > 0:
                self._cohorts_compatible = False
                break

    # -----------------------------------

    def _merge_cohort_datasets(self, cht_df: list, org_index: list):
        if not self._cohorts_compatible:
            self.print_message(
                (
                    "WARNING: the transformations used over the cohorts resulted in each cohort having different "
                    "columns. The transform() method will return a list of transformed subsets (one for each cohort)."
                )
            )
            return cht_df

        final_df = None
        for df in cht_df:
            if final_df is None:
                final_df = df
            else:
                final_df = pd.concat([final_df, df], axis=0)

        new_index = [index for index in org_index if index in final_df.index]
        final_df = final_df.reindex(new_index)

        return final_df

    # -----------------------------------

    def _merge_cohort_predictions(self, cht_pred: list, index_list: list):
        if not self._cohorts_compatible:
            return cht_pred

        final_pred = None
        for pred in cht_pred:
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
            (
                "ERROR: a subset of the instances passed to the transform(), predict(), or predict_proba() "
                "doesn't fit into any of the existing cohorts.\n" + f"The subset is given as follows:\n{missing_subset}"
            )
        )

    # -----------------------------------

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray] = None,
        y: Union[pd.Series, np.ndarray] = None,
        df: pd.DataFrame = None,
        label_col: str = None,
    ):
        self._set_df_mult(df, label_col, X, y, require_set=True)
        self._build_cohorts()
        index_used = []
        cht_df_list = []
        for i, cohort in enumerate(self.cohorts):
            cht_x, cht_y, index_list = cohort.get_cohort_subset(self.df, self.y, index_used, return_index_list=True)
            index_used += index_list
            if cht_x.empty:
                raise ValueError(f"ERROR: no valid instances found for cohort {cohort.name} in the fit() method.")
            for tf in self._cohort_pipe[i]:
                tf.fit(cht_x, cht_y)
                has_transf = self.obj_has_method(tf, "transform")
                if not has_transf:
                    break
                cht_x = tf.transform(cht_x)
            cht_df_list.append(cht_x)

        self._check_compatibility_between_cohorts(cht_df_list)
        self._check_intersection_cohorts(index_used)

        self.fitted = True
        return self

    # -----------------------------------

    def transform(self, df: Union[pd.DataFrame, np.ndarray]):

        if not self._pipe_has_transform:
            self.print_message("WARNING: none of the objects in the transform_pipe parameter have a transform() method")
            return df

        self._check_if_fitted()
        df = self._fix_col_transform(df)

        index_used = []
        cht_df_list = []
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
                cht_df_list.append(cht_x)

        self._check_intersection_cohorts(index_used)
        self._raise_missing_instances_error(df, index_used)
        final_df = self._merge_cohort_datasets(cht_df_list, org_index)

        return final_df

    # -----------------------------------

    def _predict(self, X: Union[pd.DataFrame, np.ndarray], prob: bool = False):
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
        pred_list = []
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
                pred_list.append(pred)

        self._check_intersection_cohorts(index_used)
        self._raise_missing_instances_error(df, index_used)
        final_pred = self._merge_cohort_predictions(pred_list, index_used)

        return final_pred

    # -----------------------------------

    def predict(self, X: Union[pd.DataFrame, np.ndarray]):
        final_pred = self._predict(X, prob=False)
        return final_pred

    # -----------------------------------

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]):
        final_pred = self._predict(X, prob=True)
        return final_pred

    # -----------------------------------

    def get_subsets(
        self,
        X: Union[pd.DataFrame, np.ndarray] = None,
        y: Union[pd.Series, np.ndarray] = None,
        apply_transform: bool = False,
    ):
        self._check_if_fitted()
        X = self._fix_col_transform(X)

        index_used = []
        out_dict = {}
        for i, cohort in enumerate(self.cohorts):
            cht_X, cht_y, index_list = cohort.get_cohort_subset(X, y=y, index_used=index_used, return_index_list=True)
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
