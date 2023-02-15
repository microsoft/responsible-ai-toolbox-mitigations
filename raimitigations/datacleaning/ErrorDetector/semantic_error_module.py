from gensim.models.word2vec import Word2Vec
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import word2vec
import numpy as np
import re
from .error_module import ErrorModule
from .utils import mad
import os.path

class SemanticErrorModule(ErrorModule):
    """
    This module detects values that do not belong in a categorical column, it does so by using Word2Vec architecture. Note that this module is relatively slow when using a large training corpus.

    :param corpus: a file path referring to a corpus of text;
    :param thresh: a float similarity threshold to determine when a value doesn't belong. The higher this parameter is, the less sensitive the similarity metric. This parameter defaults at 3.5;
    :param fail_thresh: an int representing the fraction of tokens not found in the corpus before short-circuiting. This parameter defaults at 5; #TODO: fix, what is this?
    """

    # -----------------------------------
    def __init__(self, corpus: str ='corpora/text8', thresh: float = 3.5, fail_thresh: int = 5):
        #compiles the corpus first time
        if os.path.isfile(corpus + '-pretrained.bin'):
            self.model = Word2Vec.load(corpus + '-pretrained.bin')
        else:
            sentences = word2vec.LineSentence(corpus)
            self.model = word2vec.Word2Vec(sentences)
            self.model.save(corpus + '-pretrained.bin')

        # l2-normalizes vectors, replace=True replaces original vectors.
        self.model.init_sims(replace=True) 
        self.thresh = thresh
        self.fail_thresh = fail_thresh

    # -----------------------------------
    def predict(self, vals: list) -> set:
        """
        Predicts and returns a set of the subset of a domain that is potentially
        erroneous.

        :param vals: a list of values to predict semantic errors on;
        :return: returns erroneous values;
        :rtype: set.
        """
        #TODO: current status: we use the average total similarity score for each val before calculating distribution and finding outliers. Their way: find erroneous values and basically if val has that token (or no tokens) then it's erroneous, they return erroneous tokens/no taken vals in predict and check what erroneous vals are (that have these tokens or no tokens) in the next function, instead if we do this, we should do that check here and the next function remains for indices of the erroneous vals only.
        
        vals_set = set(vals)
        erroneous_vals = set()
        modeled_tokens = set()

        for val in vals_set:
            # tokenize categorical value
            tokens = [t.strip().lower() for t in re.findall(r"[\w']+", val)]

            # if value has no tokens (words), add to error list
            if not tokens:
                erroneous_vals.add(val)
                vals_set.remove(val)
            else:
                match = False
                # iterate through tokens, find modeled tokens
                for t in tokens:
                    if t not in STOPWORDS and t in self.model:
                        match = True
                        modeled_tokens.add(token)
                if not match:
                    vals_set.remove(val)

        # fails if not enough tokens are present in model
        if len(modeled_tokens) < self.fail_thresh:
            return erroneous_vals
        
        # calculate total similarity score of each token to other tokens.
        token_total_similarities = {}
        for token_i in modeled_tokens:
            total_i_similarity = 0
            for token_j in modeled_tokens:
                total_i_similarity += self.model.similarity(token_i, token_j)
            token_total_similarities[token_i] = total_i_similarity
            

        # find average similarity score for each input value
        val_avg_similarity = {}
        scores = []
        for val in vals_set:
            tokens = [t.strip().lower() for t in re.findall(r"[\w']+", val)]
            avg_score = np.mean(np.array([token_total_similarities.get(key) for key in tokens]))
            val_avg_similarity[val] = avg_score
            scores.append(avg_score)

        # calculate distribution of similarity scores
        mad = mad(scores)
        median = np.median(scores)

        for val in val_avg_similarity:
            if np.abs(median - val_avg_similarity[val]) > self.thresh * mad:
                erroneous_vals.add(val)

        # return erroneous_vals
        return list(erroneous_vals)

    # -----------------------------------
    def get_erroneous_rows_in_col(self, erroneous_vals, dataset, col):
        """
        Given the error set found by predict, this method maps the errors to particular rows
        in the column, returning a list of erroneous row indices.

        :param erroneous_vals: a set of errors predicted by the predict function;
        :param dataset: dataset containing the column of data evaluated for errors;
        :param col: name or index of column that has been evaluated for errors;
        """
        col_vals = np.array(dataset[col]) #TODO: double-check this returns column not row values even with 2d lists which data[row][col], not[col][row]
        erroneous_indices = [] 
        for e_val in erroneous_vals: #TODO: what about new vals that weren't in predict, do we check here for all no tokens vals for example? (to include new values?), maybe not, should have been part of predict, will be how I set it up.
            erroneous_indices.extend(list(np.where(col_vals == e_val)[0]))

        '''
        erroneous_rows = []
        indices = []

        for row_idx, row in enumerate(dataset):
            val = d[col]
            tokens = [t.strip().lower() for t in re.findall(r"[\w']+", val)] #TODO: they need to do this for correct comparison, for comparing tokens rather than vals

            match = False

            for e in errors:

                if e in tokens or \ #TODO: if token in val or val has no tokens and is the error itself -> then val is erroneous?
                    (len(tokens) == 0 and e in val):
                    match = True
                    break

            if match:
                erroneous_rows.append(row)
                indices.append(row_idx)
        '''
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
        return ['categorical']
