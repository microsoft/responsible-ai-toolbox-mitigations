from typing import Union
from copy import deepcopy
import pandas as pd
import numpy as np
import json

from raiutils.cohort import CohortFilterMethods, CohortFilterOps, CohortJsonConst


# -----------------------------------
def _remove_nan_from_list(value_list: list):
    new_value = []
    has_nan = False
    for v in value_list:
        if type(v) == float and np.isnan(v):
            has_nan = True
        else:
            new_value.append(deepcopy(v))
    return new_value, has_nan


# -----------------------------------
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
    OP_REQ_NUMBER = [GREATER, GREATER_EQUAL, LESS, LESS_EQUAL]

    SIMPLE_OP = [GREATER, GREATER_EQUAL, LESS, LESS_EQUAL, EQUAL, DIFFERENT]

    AND = "and"
    OR = "or"
    CONDITION_CONNECTORS = [AND, OR]

    OP_NEGATION_DICT = {
        GREATER: LESS_EQUAL,
        GREATER_EQUAL: LESS,
        LESS: GREATER_EQUAL,
        LESS_EQUAL: GREATER,
        EQUAL: DIFFERENT,
        DIFFERENT: EQUAL,
    }

    CONNECTOR_NEGATION_DICT = {AND: OR, OR: AND}


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

    RAIUTILS_STATE_COND1 = 0
    RAIUTILS_STATE_COND2 = 1
    RAIUTILS_STATE_COMPOSITE = 2

    JSON_NAME_FIELD = "name"
    JSON_FILTER_FIELD = "cohort_filter_list"

    # -----------------------------------
    def __init__(self, cohort_definition: Union[list, str] = None, name: str = "cohort"):
        self.name = name
        self._set_conditions(cohort_definition)
        self.query = None
        self.columns_used = []
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
        self._require_index = False
        if conditions is None:
            self._require_index = True
            return
        if type(conditions) == str:
            self.name, conditions = self._load(conditions)
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
                if operator in CohortFilters.OP_REQ_NUMBER:
                    value = f"`{value}`"
                else:
                    value = f'"{value}"'
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
            value, has_nan = _remove_nan_from_list(value)

            if operator == CohortFilters.EQUAL:
                query = f"`{column}` in {value}"
                if has_nan:
                    nan_query = f"`{column}`.isnull()"
                    query = f"({query}) or ({nan_query})"
            elif operator == CohortFilters.DIFFERENT:
                query = f"`{column}` not in {value}"
                if has_nan:
                    nan_query = f"not `{column}`.isnull()"
                    query = f"({query}) and ({nan_query})"
            else:
                if has_nan:
                    raise ValueError("ERROR: it is not allowed to use NaN in a range operation.")
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
                    state = self.STATE_AND_OR
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
        return self._require_index

    # -----------------------------------
    def _negate_simple_condition(self, condition: list):
        """
        Creates the negation of a simple boolean expression.
        The boolean expression must be a list with three values:
        the column being used, the operator, and the value being
        compared to.

        :param condition: a simple condition, containing three values:
            the column being used, the operator, and the value being
            compared to.
        :return: the negation of the boolean expression using the same
            representation used for the input parameter 'condition', that
            is, a list with three values: the column being used, the operator,
            and the value being compared to.
        :rtype: list
        """
        column = deepcopy(condition[0])
        operator = deepcopy(condition[1])
        value = deepcopy(condition[2])

        if operator == CohortFilters.RANGE:
            less_than = [column, CohortFilters.LESS, value[0]]
            greater_than = [column, CohortFilters.GREATER, value[1]]
            not_condition = [less_than, CohortFilters.OR, greater_than]
        else:
            operator = CohortFilters.OP_NEGATION_DICT[operator]
            not_condition = [column, operator, value]

        # check if we should include a nan check as well
        if operator in CohortFilters.OP_REQ_NUMBER or operator == CohortFilters.RANGE:
            nan_condition = [column, CohortFilters.EQUAL, np.nan]
            not_condition = [not_condition, CohortFilters.OR, nan_condition]

        return not_condition

    # -----------------------------------
    def _negate_condition_list(self, condition_list: list):
        """
        Creates the negation of a complex boolean expression. This method
        is called recursively for each sub-expression.

        :param condition_list: a list with a set of filters. Check the parameter
            'cohort_definition' from the ``CohortDefinition`` class for more info.
        :return: the negation of the expression provided as input.
        :rtype: list
        """
        not_condition_list = []
        for condition in condition_list:
            if type(condition) == list:
                if type(condition[0]) == list:
                    not_condition = self._negate_condition_list(condition)
                else:
                    not_condition = self._negate_simple_condition(condition)
                not_condition_list.append(not_condition)
            elif type(condition) == str:
                not_condition_list.append(CohortFilters.CONNECTOR_NEGATION_DICT[condition])

        return not_condition_list

    # -----------------------------------
    def create_query_remaining_instances_cohort(self, prev_conditions: list):
        """
        Creates the query for the cohort that handles all instances that doesn't
        belong to any other cohort. This query is built by doing the negation of
        the condition list of all other cohorts, and concatenate them using the
        "and" operator.

        :param prev_conditions: a list of the list conditions used by other cohorts.
            Each sub-list here follows the same pattern as the parameter
            'cohort_definition' from the ``CohortDefinition`` class.
        """
        final_condition = []
        for i, condition in enumerate(prev_conditions):
            not_condition = self._negate_condition_list(condition)
            final_condition.append(not_condition)
            if i < len(prev_conditions) - 1:
                final_condition.append(CohortFilters.AND)
        self.conditions = final_condition
        self._build_query()

    # -----------------------------------
    @staticmethod
    def _filter_op_to_raiutils_filter_op(op: str, value: Union[list, int, str, float]):
        """
        Fetches the corresponding operator between the raimitigations library
        and the raiutils library.

        :param op: a string representing one of the operators from the raimitigations
            library, defined in the ``CohortFilters`` class;
        :param value: the value assinged to the operator ``op``;
        :return: the equivalent operator from the raiutils library.
        :rtype: str
        """
        if op == CohortFilters.GREATER:
            return CohortFilterMethods.METHOD_GREATER
        elif op == CohortFilters.GREATER_EQUAL:
            return CohortFilterMethods.METHOD_GREATER_AND_EQUAL
        elif op == CohortFilters.LESS:
            return CohortFilterMethods.METHOD_LESS
        elif op == CohortFilters.LESS_EQUAL:
            return CohortFilterMethods.METHOD_LESS_AND_EQUAL
        elif op == CohortFilters.EQUAL:
            if type(value) in [list, str]:
                return CohortFilterMethods.METHOD_INCLUDES
            return CohortFilterMethods.METHOD_EQUAL
        elif op == CohortFilters.RANGE:
            return CohortFilterMethods.METHOD_RANGE
        elif op == CohortFilters.DIFFERENT:
            return CohortFilterMethods.METHOD_EXCLUDES

    # -----------------------------------
    def _build_raiutils_simple_block(self, column: str, op: str, value: Union[list, int, str, float]):
        """
        Convert a simple condition list (comprised of column, operator, and value) to
        a simple filter dictionary used in the JSON format of raiutils when saving
        a cohort.

        :param column: the name or index of the column being used in the filter;
        :param op: a string that identifies the operator being used in the filter;
        :param value: a list of values or a single value used in the filter;
        :return: a dictionary representing a simple filter following the format used
            in raiutils when saving a cohort to a JSON.
        :rtype: dict
        """
        method = self._filter_op_to_raiutils_filter_op(op, value)
        arg = value
        if type(value) != list:
            arg = [value]
        simple_block = {CohortJsonConst.COLUMN: column, CohortJsonConst.METHOD: method, CohortJsonConst.ARG: arg}
        return simple_block

    # -----------------------------------
    def _conditions_to_raiutils_filters(self, conditions: list):
        """
        Convert a condition list (used by the raimitigations) to the JSON format used
        to represent the filters of a cohort in the raiutils library. This mehtod is
        called recursively for each condition list inside the original condition list
        that is not a simple condition list (comprised of column, operator, and value).

        :param conditions: the condition list to be converted;
        :return: a dictionary representing the composite filter passed through the
            ``conditions`` parameter. The dictionary thfollows the format used in raiutils
            when saving a cohort to a JSON.
        :rtype: dict
        """
        state = self.STATE_COND1
        connector = CohortFilterOps.OR
        for condition in conditions:
            if type(condition) == list:
                if type(condition[0]) == list:
                    block = self._conditions_to_raiutils_filters(condition)
                else:
                    block = self._build_raiutils_simple_block(condition[0], condition[1], condition[2])
                if state == self.STATE_COND1:
                    part1 = block
                    state = self.STATE_AND_OR
                elif state == self.STATE_COND2:
                    block = {CohortJsonConst.COMPOSITE_FILTERS: [part1, block], CohortJsonConst.OPERATION: connector}
                    part1 = block
                    state = self.STATE_COND1
            elif type(condition) == str:
                if condition in CohortFilters.CONDITION_CONNECTORS:
                    connector = CohortFilterOps.OR
                    if condition == CohortFilters.AND:
                        connector = CohortFilterOps.AND
                    state = self.STATE_COND2

        return block

    # -----------------------------------
    def _convert_conditions_to_raiutils_filters(self):
        """
        Convert a condition list (used by the raimitigations) to the JSON format used
        to represent the filters of a cohort in the raiutils library.

        :return: a dictionary representing the composite filter passed through the
            ``conditions`` parameter. The dictionary thfollows the format used in raiutils
            when saving a cohort to a JSON.
        :rtype: dict
        """
        if self.conditions is None or self.conditions == []:
            raise ValueError("ERROR: can't save conditions from a cohort without conditions.")
        raiutils_filters = [self._conditions_to_raiutils_filters(self.conditions)]
        return raiutils_filters

    # -----------------------------------
    def save(self, json_file: str):
        """
        Saves the conditions used by the cohort into a JSON file.

        :param json_file: the path of the JSON file to be saved.
        """
        filters = self._convert_conditions_to_raiutils_filters()
        cht_json = {self.JSON_NAME_FIELD: self.name, self.JSON_FILTER_FIELD: filters}
        with open(json_file, "w") as file:
            json.dump(cht_json, file, indent=4)

    # -----------------------------------
    def _validate_json(self, json_dict: dict, json_file: str):
        """
        Simple validation of the JSON file being loaded.

        :param json_dict: the dictionary loaded from the json file;
        :param json_file: the name of the json file.
        """
        required_fields = [self.JSON_NAME_FIELD, self.JSON_FILTER_FIELD]
        for field in required_fields:
            if field not in json_dict.keys():
                raise ValueError(
                    f"ERROR: the json file {json_file} does not contain the mandatory '{field}' key in it."
                )

    # -----------------------------------
    def _filter_dict_to_single_condition(self, single_filter: dict):
        """
        Converts a simple dictionary filter used in the JSON file format adopted
        by the raiutils to a simple condition list used in raimitigations.

        :param single_filter: a dictionary representing a simple filter in the
            json file;
        :return: a simple condition list (comprised of column, operator, and value);
        :rtype: list
        """
        column = single_filter[CohortJsonConst.COLUMN]
        if column in CohortJsonConst.INVALID_TERMS:
            raise ValueError(
                f"ERROR: the CohortDefinition class does not support the use of column {column}, "
                + "which is used for special cases by the raiutils library. Use a different column name."
            )

        operator = single_filter[CohortJsonConst.METHOD]
        value = single_filter[CohortJsonConst.ARG][0]
        if operator == CohortFilterMethods.METHOD_GREATER:
            operator = CohortFilters.GREATER
        elif operator == CohortFilterMethods.METHOD_GREATER_AND_EQUAL:
            operator = CohortFilters.GREATER_EQUAL
        elif operator == CohortFilterMethods.METHOD_LESS:
            operator = CohortFilters.LESS
        elif operator == CohortFilterMethods.METHOD_LESS_AND_EQUAL:
            operator = CohortFilters.LESS_EQUAL
        elif operator in [CohortFilterMethods.METHOD_EQUAL, CohortFilterMethods.METHOD_INCLUDES]:
            if operator == CohortFilterMethods.METHOD_INCLUDES:
                value = single_filter[CohortJsonConst.ARG]
            operator = CohortFilters.EQUAL
        elif operator == CohortFilterMethods.METHOD_EXCLUDES:
            operator = CohortFilters.DIFFERENT
            value = single_filter[CohortJsonConst.ARG]
        elif operator == CohortFilterMethods.METHOD_RANGE:
            operator = CohortFilters.RANGE
            value = single_filter[CohortJsonConst.ARG]

        return [column, operator, value]

    # -----------------------------------
    def _convert_raiutils_filters_to_conditions(self, filters: list, connector: int = None):
        """
        Converts the filters saved in a JSON file (using raiutils' format) to a conditions
        list used by raimitigations. This method is called recursively for each sub-filter
        found in the JSON file.

        :param filters: a list of filters in the JSON file;
        :param connector: the connector ('and' or 'or') to be used when concatanating two
            or more conditions. The default connector used is 'and';
        :return: a list of conditions formatted according to raimitigations' standards;
        :rtype: list
        """
        if connector is None:
            connector = CohortFilters.AND
        conditions = []

        state = self.RAIUTILS_STATE_COND1
        for single_filter in filters:
            if CohortJsonConst.METHOD in single_filter:
                single_condition = self._filter_dict_to_single_condition(single_filter)
                if state == self.RAIUTILS_STATE_COND1:
                    state = self.RAIUTILS_STATE_COND2
                elif state == self.RAIUTILS_STATE_COND2:
                    conditions.append(connector)
                conditions.append(single_condition)
            else:
                composite_filter_list = single_filter[CohortJsonConst.COMPOSITE_FILTERS]
                conector = CohortFilters.AND
                if single_filter[CohortJsonConst.OPERATION] == CohortFilterOps.OR:
                    conector = CohortFilters.OR
                composite_condition = self._convert_raiutils_filters_to_conditions(
                    composite_filter_list, connector=conector
                )
                if state == self.RAIUTILS_STATE_COND1:
                    state = self.RAIUTILS_STATE_COND2
                elif state == self.RAIUTILS_STATE_COND2:
                    conditions.append(connector)
                conditions.append(composite_condition)

        return conditions

    # -----------------------------------
    def _load(self, json_file: str):
        """
        Loads the conditions contained in a JSON file.

        :param json_file: the path to the JSON file that should be loaded.
        """
        with open(json_file, "r") as file:
            cht_json = json.load(file)

        self._validate_json(cht_json, json_file)
        name = cht_json[self.JSON_NAME_FIELD]
        filters = cht_json[self.JSON_FILTER_FIELD]
        conditions = self._convert_raiutils_filters_to_conditions(filters)

        return name, conditions

    # -----------------------------------
    def get_cohort_subset(
        self, X: pd.DataFrame, y: pd.DataFrame = None, index_used: list = None, return_index_list: bool = False
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

        :param X: a data frame containing the features of a dataset
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
        self.check_valid_df(X)
        if self.require_remaining_index():
            if index_used is None:
                raise ValueError(
                    "ERROR: when calling the get_cohort_subset() method for a cohort with no conditions, "
                    "it is necessary to provide the 'index_used' parameter so that the remaining indexes "
                    "are used."
                )
            missing_index = [index for index in X.index if index not in index_used]
            subset = X.filter(items=missing_index, axis=0)
        else:
            subset = X.query(self.query)

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
