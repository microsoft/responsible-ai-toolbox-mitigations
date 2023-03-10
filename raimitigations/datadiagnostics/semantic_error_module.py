from gensim.models.word2vec import Word2Vec
from gensim.parsing.preprocessing import STOPWORDS
import numpy as np
import re
from .error_module import ErrorModule
from .utils import calculate_mad
import os.path


class SemanticErrorModule(ErrorModule):
    """
    This module detects values that do not belong in a categorical column, it does so by using Word2Vec architecture. Note that this module is relatively slow when using a large training corpus.

    :param corpus: a file path referring to a corpus of text;
    :param thresh: a float similarity threshold to determine when a value doesn't belong. The higher this parameter is, the less sensitive the similarity metric. This parameter defaults at 3.5;
    :param fail_thresh: an int representing the fraction of tokens not found in the corpus before short-circuiting. This parameter defaults at 5;
    """

    # -----------------------------------
    def __init__(
        self,
        corpus: str = os.path.join(os.path.abspath(os.path.dirname(__file__)), "corpora/text8"),
        thresh: float = 3.5,
        fail_thresh: int = 5,
    ):
        self.model = Word2Vec.load(corpus + "-pretrained.bin")

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
        :return: returns erroneous values;
        :rtype: set.
        """
        # current status: find erroneous tokens and if val has an erroneous token (or no tokens) then it's an erroneous val.

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

        # take MAD to filter corpus
        mad = calculate_mad(list(token_total_similarities.values()))
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
    def get_erroneous_rows_in_col(self, col_vals):
        """
        Given the error set found by predict, this method maps the errors to particular rows
        in the column, returning a list of erroneous row indices.

        :param col_vals: a list of values to predict semantic errors on;
        :return:
        :rtype:
        """
        erroneous_vals = self._predict(col_vals)
        print(list(erroneous_vals))
        erroneous_indices = []
        for e_val in erroneous_vals:
            erroneous_indices.extend(list(np.where(col_vals == e_val)[0]))

        return erroneous_indices

    # -----------------------------------
    def description(self):
        """
        Returns a description of the error.
        """
        return f"SemanticError: A value was found with a word2vec similarity score greater than > {str(self.thresh)} stds beyond the mean similarity score of all values."

    # -----------------------------------
    def get_available_types(self):
        """
        Returns a list of data types available for prediction using this error detection module.
        """
        return ["categorical"]
