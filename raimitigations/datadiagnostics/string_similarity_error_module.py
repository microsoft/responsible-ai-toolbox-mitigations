from gensim.models.word2vec import Word2Vec
import numpy as np
from .error_module import ErrorModule


class StringSimilarityErrorModule(ErrorModule):

    """
    This module detects values that do not belong in a string-valued column. It fine-tunes Word2Vec on the given set of data and compares the score of likelihood of input values within the set using standard deviation to predict possibly erroneous values.

    :param thresh: a standard deviation count threshold to determine how many stds can a non-erroneous string's likelihood score be beyond the dataset's mean. This parameter defaults at 3.5;
    """

    # -----------------------------------

    def __init__(self, thresh=3.5):
        self.thresh = thresh
        self.module_name = "StringSimilarityErrorModule"

    # -----------------------------------
    def _predict(self, strings):
        """
        Predicts and returns a set of the subset of a domain that is potentially
        erroneous.

        :param strings: a list of string values to predict string-similarity errors on;
        """
        strings = [x for x in strings if str(x) != "nan"]
        self.model = Word2Vec(
            [s.lower().split() for s in strings], hs=1, negative=0, min_count=1
        )  # train a word2vec model using the input word strings

        erroneous_vals = set()

        # for each val in the column
        string_scores = []
        scoredict = {}

        for s in strings:
            cleaned_string = s.lower().split()
            if len(cleaned_string) == 0:
                erroneous_vals.add(s)
            else:
                score = np.squeeze(self.model.score([cleaned_string])) / len(cleaned_string)  # sentence average score
                string_scores.append(score)
                scoredict[s] = score

        score_std = np.std(string_scores)
        score_median = np.mean(string_scores)

        for s in scoredict:
            if np.abs(score_median - scoredict[s]) > self.thresh * score_std:
                erroneous_vals.add(s)

        return erroneous_vals

    # -----------------------------------
    def get_erroneous_rows_in_col(self, col_vals):
        """
        Given the error set found by predict, this method maps the errors to particular rows
        in the column, returning a list of erroneous row indices.

        :param col_vals: a list of string values to predict string-similarity errors on;

        :return:
        :rtype:
        """
        erroneous_vals = self._predict(col_vals)
        erroneous_indices = []
        for e_val in erroneous_vals:
            erroneous_indices.extend(list(np.where(col_vals == e_val)[0]))

        return erroneous_indices

    # -----------------------------------
    def description(self):
        """
        Returns a description of the error.
        """
        return f"StringSimilarityError: A string was not well predicted by a finetuned word2vec model. Its likelihood score was found to be greater than > {str(self.thresh)} stds beyond the mean likelihood."

    # -----------------------------------
    def get_available_types(self):
        """
        Returns a list of data types available for prediction using this error detection module.
        """
        return ["string"]
