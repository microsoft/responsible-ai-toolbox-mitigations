import numpy as np
import re
from .error_module import ErrorModule

class PuncErrorModule(ErrorModule):

    """
    This module detects attributes that are only punctuation, whitespace, etc.
    """
    # -----------------------------------
    def __init__(self):
        pass

    # -----------------------------------
    def predict(self, strings):
        """
        Predicts and returns a set of the subset of a domain that is potentially
        erroneous.

        :param strings: a list of string values to predict punctuation errors on;
        """
        erroneous_vals = set()

        vals_set = set(strings)  # get unique values

        for s in vals_set:
            sstrip = re.sub(r'\W+', '', s.lower())
            cleaned_string = sstrip.lower().strip()
            if len(cleaned_string) == 0:
                erroneous_vals.add(s)

        return erroneous_vals

    # -----------------------------------
    def get_erroneous_rows_in_col(self, erroneous_vals, dataset, col):
        """
        Given the error set found by predict, this method maps the errors to particular rows
        in the column, returning a list of erroneous row indices.

        :param erroneous_vals: a set of errors predicted by the predict function;

        :param dataset: dataset containing the column of data evaluated for errors;

        :param col: name or index of column that has been evaluated for errors;
        """
        col_vals = np.array(dataset[col])  # TODO: doublecheck this returns column not row values even with 2d lists which data[row][col], not[col][row]
        erroneous_indices = []
        for e_val in erroneous_vals:
            erroneous_indices.extend(list(np.where(col_vals == e_val)[0]))

        '''
        erroneous_indices = []
        eset = set(erroneous_vals)

        for i, d in enumerate(dataset):
            val = d[col]
            if val in eset:
                erroneous_indices.append(i)
        '''

        return erroneous_indices

    # -----------------------------------
    def description(self):
        """
        Returns a description of the error.
        """
        return "PunctuationError: An attribute was found with no alpha numeric characeters."

    # -----------------------------------
    def get_available_types(self):
        """
        Returns a list of data types available for prediction using this error detection module.
        """
        return ['categorical', 'string']
