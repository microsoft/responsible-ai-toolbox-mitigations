from gensim.models.word2vec import Word2Vec
from gensim.parsing.preprocessing import STOPWORDS
import numpy as np
import re
from scipy import stats
from .error_module import ErrorModule
import os.path

class SemanticErrorModule(ErrorModule):
    """
    This module predicts values that do not belong in a categorical column, it does so by using Word2Vec architecture.

    :param thresh: a float similarity threshold to determine when a value doesn't belong. This parameter defaults at 3.5;
    :param fail_thresh: an int representing the minimum required number of tokens found in the corpus before short-circuiting. This parameter defaults at 5;
    """
    # -----------------------------------
    def __init__(self, thresh: float = 3.5, fail_thresh: int = 5):
        self.corpus = os.path.join(os.path.abspath(os.path.dirname(__file__)), "corpora/text8")
        self.model = Word2Vec.load(self.corpus + "-pretrained.bin")

        # l2-normalizes vectors, replace=True replaces original vectors.
        self.model.init_sims(replace=True)
        self.thresh = thresh
        self.fail_thresh = fail_thresh
        self.module_name = "SemanticErrorModule"

    # -----------------------------------
    def _predict(self, vals: list) -> set:
        """
        Predicts and returns a set of the subset of a domain that is potentially
        erroneous.

        :param vals: a list of values to predict semantic errors on;

        :return: a set of predicted erroneous values;
        :rtype: a set.
        """
        # current status: find erroneous tokens and if val has an erroneous token (or no tokens) then it's an erroneous val.

        vals = [x for x in vals if str(x) != "nan"]
        vals_set = set(vals)
        # model_vals_set = set(vals)
        modeled_tokens = set()
        erroneous_vals = set()

        vals_tokens = {}
        for val in vals_set:
            # tokenize categorical value
            tokens = [t.strip().lower() for t in re.findall(r"[\w']+", str(val))]
            vals_tokens[val] = tokens

            # if value has no tokens (words), add to error list
            if not tokens:
                erroneous_vals.add(val)
                # model_vals_set.remove(val)
            else:
                # match = False
                # iterate through tokens, find modeled tokens
                for t in tokens:
                    if t not in STOPWORDS and t in self.model.wv:
                        # match = True
                        modeled_tokens.add(t)
                # if not match:
                # model_vals_set.remove(val)

        # fails if not enough tokens are present in model
        if len(modeled_tokens) < self.fail_thresh:
            return erroneous_vals

        # calculate total similarity score of each token to other tokens.
        token_total_similarities = {}
        for token_i in modeled_tokens:
            total_i_similarity = 0
            for token_j in modeled_tokens:
                total_i_similarity += self.model.wv.similarity(token_i, token_j)
            token_total_similarities[token_i] = total_i_similarity

        # take MAD
        mad = stats.median_abs_deviation(list(token_total_similarities.values()))
        median = np.median(list(token_total_similarities.values()))
        erroneous_tokens = set()

        for token in list(token_total_similarities.keys()):
            if np.abs(median - token_total_similarities[token]) > self.thresh * mad:
                erroneous_tokens.add(token)

        for val in vals_set:
            if list(set(vals_tokens[val]).intersection(erroneous_tokens)):
                erroneous_vals.add(val)

        """
        # or: we use the average total similarity score for each val before calculating distribution and finding outliers.

        # find average similarity score for each input value
        val_avg_similarity = {}
        scores = []
        for val in model_vals_set:
            tokens = [t.strip().lower() for t in re.findall(r"[\w']+", str(val))] # noqa
            avg_score = np.mean(np.array([token_total_similarities.get(key) for key in tokens]))
            val_avg_similarity[val] = avg_score
            scores.append(avg_score)

        # calculate distribution of similarity scores
        mad = calculate_mad(scores)
        median = np.median(scores)

        for val in val_avg_similarity:
            if np.abs(median - val_avg_similarity[val]) > self.thresh * mad:
                erroneous_vals.add(val)

        # return erroneous_vals
        """
        return erroneous_vals

    # -----------------------------------
    def _description(self) -> str:
        """
        Returns a description of the error.
        """
        return f"SemanticError: A value was found with a word2vec similarity score greater than > {str(self.thresh)} stds beyond the mean similarity score of all values."

    # -----------------------------------
    def _get_available_types(self) -> list:
        """
        Returns a list of data types available for prediction using this error prediction module.
        """
        return ["categorical"]
