from typing import Union
import pandas as pd
import numpy as np
import json


class CohortFilters:

    GREATER = ">"
    GREATER_EQUAL = ">="
    LESS = "<"
    LESS_EQUAL = "<="
    EQUAL = "=="
    RANGE = "range"
    DIFFERENT = "!="
    ALL = [GREATER, GREATER_EQUAL, LESS, LESS_EQUAL, EQUAL, RANGE, DIFFERENT]

    OP_ALLOW_LIST = [EQUAL, RANGE, DIFFERENT]
    OP_REQ_NUMER = [GREATER, GREATER_EQUAL, LESS, LESS_EQUAL]

    SIMPLE_OP = [GREATER, GREATER_EQUAL, LESS, LESS_EQUAL, EQUAL, DIFFERENT]

    AND = "and"
    OR = "or"
    CONDITION_CONNECTORS = [AND, OR]


class CohortDefinition:
    """
    Implements an interface for building and filtering cohorts from any dataset.
    This class is not associated to any specific dataset. It simply converts a
    set of conditions into a query. Based on this query, it is able to fetch
    a cohort that satisfies these conditions.

    :param cohort_definition: a list of conditions or a string containing the path
        to a JSON file that has the list condition. A basic condition is a list
        with three values:

            1. **Column:** name or index of the column being analyzed
            2. **Inner Operator:** one of the following operators: ``'=='``, ``'!='``,
               ``'>'``, ``'>='``, ``'<'``, ``'<='``, or ``'range'``)
            3. **Value:** the value used in the condition. It can be a numerical or
               categorical value.

        An ``and`` or ``or`` operator may be placed between two basic conditions. Complex
        conditions may be created by concatenating multiple conditions;

    :param name: a string indicating the name of the cohort. This parameter may be accessed
        later using the ``name`` attribute.
    """

    STATE_COND1 = 0
    STATE_AND_OR = 1
    STATE_COND2 = 2

    # -----------------------------------
    def __init__(self, cohort_definition: Union[list, str] = None, name: str = "cohort"):
        self._set_conditions(cohort_definition)
        self.query = None
        self.columns_used = []
        self.name = name
        self._build_query()

    # -----------------------------------
    def _set_conditions(self, conditions: Union[list, str] = None):
        """
        Sets the ``conditions`` attribute based on the parameter passed through
        the constructor method. Before setting this attribute, perform some error
        checks, and check if the parameter provided is indeed a list of conditions
        or if it is the path of a JSON file containing the conditions.

        :param conditions: a list of conditions or a string containing the path
            to a JSON file that has the list condition.
        """
        self.conditions = None
        if conditions is None:
            return
        if type(conditions) == str:
            conditions = self._load(conditions)
        elif type(conditions) != list:
            raise ValueError(
                "ERROR: the cohort_definition parameter must be a string or a list. If cohort_definition is a "
                + "string, it must be a valid json file name that contains the conditions for a cohort."
            )
        self.conditions = conditions

    # -----------------------------------
    def _get_condition_parts(self, condition: list):
        """
        Extracts the column, operator, and value used for a basic condition. Afterward,
        perform several error checks in order to validate that the operator used is
        compatible with the value provided.

        :param condition: a list with three variables: the column name or index, the
            operator, and the value.
        :return: a tuple with the following values: (column, operator, value)
        :rtype: tuple
        """
        instruction_msg = (
            "each condition must be comprised of exactly three values: "
            "[COLUMN, OPERATOR, VALUE], where COLUMN is either string representing the column name or "
            "an integer representing the column index, OPERATOR is one of the following operators: "
            f"{CohortFilters.ALL}, and VALUE representing the value associated to the operator and "
            "the specified column."
        )
        range_msg = (
            f"ERROR: the value associated to the RANGE operator '{CohortFilters.RANGE}' must be a list "
            f"with only two numbers: the minimum and maximum values of the range, respectively. "
        )

        if len(condition) != 3:
            raise ValueError(f"ERROR: {instruction_msg}")

        column = condition[0]
        operator = condition[1]
        value = condition[2]

        if operator not in CohortFilters.ALL:
            raise ValueError(
                f"ERROR: invalid operator found. Expected one of the following operators: {CohortFilters.ALL}. "
                + f"Instead, found {operator}."
            )

        if type(value) == list:
            if len(value) == 0:
                raise ValueError(
                    "ERROR: invalid list passed as the value for a condition. Expected a list with at least "
                    + f"one value, but got an empty list. Condition: {column} {operator} {value}."
                )
            if operator not in CohortFilters.OP_ALLOW_LIST:
                raise ValueError(f"ERROR: a list value is not allowed for the operator {operator}.")
            if operator == CohortFilters.RANGE:
                if len(value) != 2:
                    raise ValueError(f"{range_msg}Instead, got the following list with {len(value)} elements: {value}.")
        elif type(value) == str or isinstance(value, (int, float, np.integer, np.float64)):
            if operator not in CohortFilters.SIMPLE_OP:
                raise ValueError(
                    f"ERROR: invalid operator '{operator}' associated to value '{value}'. The allowed "
                    + f"operators for this value are: {CohortFilters.SIMPLE_OP}"
                )
            if type(value) == str:
                if operator in CohortFilters.OP_REQ_NUMER:
                    value = f"`{value}`"
                else:
                    value = f"'{value}'"
        else:
            raise ValueError(
                f"ERROR: invalid value provided: {value} {type(value)}. Condition: {column} {operator} {value}."
            )

        return column, operator, value

    # -----------------------------------
    def _get_single_condition_query(self, condition: list):
        """
        Builds the query for a basic condition, that is, a list with a
        column name or index, an operator, and a value.

        :param condition: a list with three variables: the column name or index, the
            operator, and the value.
        :return: a pandas query that applies the same filter as the one specified by
            the ``condition`` parameter
        :rtype: str
        """
        column, operator, value = self._get_condition_parts(condition)
        self.columns_used.append(column)
        if type(value) == list:
            if operator == CohortFilters.EQUAL:
                query = f"`{column}` in {value}"
            elif operator == CohortFilters.DIFFERENT:
                query = f"`{column}` not in {value}"
            else:
                query = f"{value[0]} <= `{column}` <= {value[1]}"
        else:
            if type(value) == float and np.isnan(value):
                query = f"`{column}`.isnull()"
            else:
                query = f"`{column}` {operator} {value}"
        return query

    # -----------------------------------
    def _create_cohort_query(self, conditions: list):
        """
        Builds a pandas query based on a list of conditions. This method is
        called recursively until it reaches a basic condition. While building
        the query, several error checks are performed to validate the structure
        of the list of conditions.

        :param condition: a list of conditions
        :return: a pandas query that applies the same filter as the one specified by
            the ``conditions`` parameter
        :rtype: str
        """
        state = self.STATE_COND1
        query = ""
        for condition in conditions:
            if type(condition) == list:
                if type(condition[0]) == list:
                    part_query = self._create_cohort_query(condition)
                else:
                    part_query = self._get_single_condition_query(condition)
                if state == self.STATE_COND1:
                    query = f"({part_query})"
                    state = self.STATE_AND_OR
                elif state == self.STATE_COND2:
                    query = f"{query}({part_query})"
                    state = self.STATE_COND1
                else:
                    raise ValueError(
                        f"ERROR: expected one of the following connectors: {CohortFilters.CONDITION_CONNECTORS}. "
                        + f"Instead, found another condition: '{condition}'"
                    )
            elif type(condition) == str:
                if condition in CohortFilters.CONDITION_CONNECTORS:
                    connector = "or"
                    if condition == CohortFilters.AND:
                        connector = "and"
                    query += f" {connector} "
                    state = self.STATE_COND2
                else:
                    raise ValueError(
                        f"ERROR: expected one of the following connectors: {CohortFilters.CONDITION_CONNECTORS}. "
                        + f"Instead, found the connector: '{condition}'"
                    )
            else:
                raise ValueError(f"ERROR: invalid value found in cohort condition: {condition}")

        if state == self.STATE_COND2:
            raise ValueError(
                "ERROR: expected a complementary condition associated to the "
                + f"connector '{connector}'. Partial query found: {query}."
            )

        self.columns_used = list(set(self.columns_used))

        return query

    # -----------------------------------
    def _build_query(self):
        if self.conditions is None:
            return

        self.query = self._create_cohort_query(self.conditions)

    # -----------------------------------
    def check_valid_df(self, df: pd.DataFrame):
        """
        Checks if a dataset contains all columns used by the cohort's query. If at least one
        of the columns used in the query is not present, an error is raised.

        :param df: a pandas dataset.
        """
        for col in self.columns_used:
            if col not in df.columns:
                raise ValueError(
                    "ERROR: one of the columns used by the cohort filter are missing in the dataset provided. "
                    + f"The current cohort uses the following columns: {self.columns_used}. The dataset provided "
                    + f"has the following columns: {df.columns}"
                )

    # -----------------------------------
    def require_remaining_index(self):
        """
        Returns True if the cohort requires the ``index_used`` parameter
        for the ``get_cohort_subset()`` method. When this happens, this
        means that this cohort was built with a ``cohort_definition``
        parameter set to ``None``.

        :return: True if the cohort requires the ``index_used`` parameter
            for the ``get_cohort_subset()`` method. False otherwise.
        :rtype: bool
        """
        return self.conditions is None or self.conditions == []

    # -----------------------------------
    def save(self, json_file: str):
        """
        Saves the conditions used by the cohort into a JSON file.

        :param json_file: the path of the JSON file to be saved.
        """
        with open(json_file, "w") as file:
            json.dump(self.conditions, file, indent=4)

    # -----------------------------------
    def _load(self, json_file: str):
        """
        Loads the conditions contained in a JSON file.

        :param json_file: the path to the JSON file that should be loaded.
        """
        with open(json_file, "r") as file:
            conditions = json.load(file)
        return conditions

    # -----------------------------------
    def get_cohort_subset(
        self, df: pd.DataFrame, y: pd.DataFrame = None, index_used: list = None, return_index_list: bool = False
    ):
        """
        Filters a dataset to fetch only the instances that follow the
        conditions of the current cohort. If the current cohort doesn't
        have any conditions, this means that this cohort is comprised of
        all instances that doesn't belong to any other cohort
        (cohort_definition = None). In this case, the cohort subset will
        be all instances whose index does not belong to any other cohort.
        The list of indices used in other cohorts is given by the
        index_used parameter. Finally, return the filtered dataset.

        :param df: a data frame containing the features of a dataset
            that should be filtered;
        :param y: the label dataset (``y``) that should also be filtered.
            This ``y`` dataset should have the same number of rows as the
            ``df`` parameter. The filtering is performed over the ``df``
            dataset, and the same indices selected in ``df`` are also
            selected in the ``y`` dataset;
        :param index_used: a list of all indices of the dataset df that
            already belongs to some other cohort. This is used when the
            cohort doesn't have a valid list of conditions. In that case,
            this cohort is the group of all instances that doesn't belong
            to any other cohort;
        :param return_index_list: if True, return the list of indices
            that belongs to the cohort. If False, this list isn't returned;
        :return: this method can return 4 different values based on its
            parameters:

                * when ``y`` is provided and ``return_index_list == True``,
                  the following tuple is returned: (subset, subset_y, index_list)
                * when ``y`` is provided and ``return_index_list == False``,
                  the following tuple is returned: (subset, subset_y)
                * when ``y`` is not provided and ``return_index_list == True``,
                  the following tuple is returned: (subset, index_list)
                * when ``y`` is not provided and ``return_index_list == False``,
                  the following dataframe is returned: subset

            Here, ``subset`` is the subset of ``df`` that belongs to the cohort,
            ``subset_y`` is the label dataset associeted to ``subset``, and
            ``index_list`` is the list of indices of instances that belongs to the
            cohort;
        :rtype: tuple or pd.DataFrame
        """
        self.check_valid_df(df)
        if self.require_remaining_index():
            if index_used is None:
                raise ValueError(
                    "ERROR: when calling the get_cohort_subset() method for a cohort with no conditions, "
                    "it is necessary to provide the 'index_used' parameter so that the remaining indexes "
                    "are used."
                )
            missing_index = [index for index in df.index if index not in index_used]
            subset = df.filter(items=missing_index, axis=0)
        else:
            subset = df.query(self.query)

        index_list = list(subset.index)
        if y is not None:
            subset_y = y.filter(items=index_list, axis=0)
            if return_index_list:
                return subset, subset_y, index_list
            else:
                return subset, subset_y

        if return_index_list:
            return subset, index_list
        else:
            return subset
