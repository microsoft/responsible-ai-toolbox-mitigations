from gensim.models.word2vec import Word2Vec
import numpy as np
from .error_module import ErrorModule


class StringSimilarityErrorModule(ErrorModule):
    """
    This module predicts values that do not belong in a string-valued column. It fine-tunes Word2Vec on the given set of data and compares the score of likelihood of input values within the set using standard deviation to predict possibly erroneous values.

    :param thresh: a standard deviation count threshold to determine how many stds can a non-erroneous string's likelihood score be beyond the dataset's mean. This parameter defaults at 3.5;
    """
    # -----------------------------------
    def __init__(self, thresh: float = 3.5):
        self.thresh = thresh
        self.module_name = "StringSimilarityErrorModule"

    # -----------------------------------
    def _predict(self, strings: list) -> set:
        """
        Predicts and returns a set of the subset of a domain that is potentially
        erroneous.

        :param strings: a list of string values to predict string-similarity errors on;

        :return: a set of predicted erroneous values;
        :rtype: a set.
        """
        strings = [x for x in strings if str(x) != "nan"]

        #train a word2vec model using the input word strings
        self.model = Word2Vec(
            [s.lower().split() for s in strings], hs=1, negative=0, min_count=1
        )

        erroneous_vals = set()

        # for each val in the column
        string_scores = []
        scoredict = {}

        for s in strings:
            cleaned_string = s.lower().split()
            if len(cleaned_string) == 0:
                erroneous_vals.add(s)
            else:
                score = np.squeeze(self.model.score([cleaned_string])) / len(cleaned_string)
                string_scores.append(score)
                scoredict[s] = score

        score_std = np.std(string_scores)
        score_median = np.mean(string_scores)

        for s in scoredict:
            if np.abs(score_median - scoredict[s]) > self.thresh * score_std:
                erroneous_vals.add(s)

        return erroneous_vals

    # -----------------------------------
    def _description(self) -> str:
        """
        Returns a description of the error.
        """
        return f"StringSimilarityError: A string was not well predicted by a fine-tuned word2vec model. Its likelihood score was found to be greater than > {str(self.thresh)} stds beyond the mean likelihood."

    # -----------------------------------
    def _get_available_types(self) -> list:
        """
        Returns a list of data types available for prediction using this error prediction module.
        """
        return ["string"]
