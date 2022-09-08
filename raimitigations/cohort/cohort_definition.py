from typing import List
import itertools

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.spatial import distance


'''
* CohortDefinition(condition_dict or json file)
* builds the query for the cohort
* filter(df)
    - given a dataframe, filter it and return a sub_df that contains only instances
    from the cohort
*

TODO:
    - in the constructor, allow the cohort_definition parameter to be a json file
    - create a method that saves the cohort definition into a json file
'''

class CohortDefinition():

    # -----------------------------------
    def __init__(self, cohort_definition: list):
        self.cohort_definition = cohort_definition
        self.query = None
        self._conditions_from_list()
        self._set_cohort_query()
        print(self.conditions)
        print(self.query)


    # -----------------------------------
    def _conditions_from_list(self):
        """
        Reeconstructs the list of conditions provided by the user for a
        given cohort through the cohort_definition parameter.
        """
        conditions = self.cohort_definition
        self.conditions = []
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
                self.conditions.append(condition)


    # -----------------------------------
    def _set_cohort_query(self):
        """
        Generates and saves the string that represents the query
        used to filter the original full dataset to a dataset
        containing only the instances considered by the current
        cohort. The string follows the patterns followed by the
        method .query() from the pandas library.
        """
        if self.query is not None:
            return

        def is_number(value):
            try:
                _ = float(value)
                return True
            except:
                return False

        query = ""
        for cond in self.conditions:
            if query != "":
                query += " or "
            query += "("
            for j, col in enumerate(cond.keys()):
                if is_number(cond[col]):
                    value = float(cond[col])
                    if np.isnan(value):
                        query += f"`{col}`.isnull()"
                    else:
                        query += f"`{col}`=={cond[col]}"
                else:
                    query += f"`{col}`=='{cond[col]}'"
                if j != len(cond.keys()) - 1:
                    query += " and "
            query += ")"
        self.query = query