# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.


import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors._base import KNeighborsMixin
from numbers import Integral
from sklearn.base import clone


_MESSAGE_DATASET_NOT_PROVIDED = "A panda dataframe dataset is a required parameter."
_MESSAGE_TARGET_NOT_PROVIDED = "the dataset target is a required parameter."
_MESSAGE_INDEX_NOT_FOUND = "Index is not found."
_MESSAGE_N_SAMPLE_LESS_K_NEIGHBOR = "The minimum number of samples (n_sample) in your data is equal to 1.  Solutions-1: You can exclude the classes in your data that are equal to 1.  Solution-2: Use the RandomOverSampler from ImbalancedLearn which does not have a similar restriction"


class Rebalance:

    """
    :param dataset: the dataframe that the user will rebalance  
    :type dataset: pd.DataFrame
    :param target: the name of the target column or the index of the target column
    :type target: str, int
    :param sampling_strategy: optional parameter for the strategy to use for the smote tomek resampling
    :type sampling_strategy: string 
    :param random_state: seed to control the randomization of the algorithm
    :type random_state: int
    :param smote: optional parameter to include a SMOTE object
    :type smote: imblearn.SMOTE
    :param tomek: optional parameter to include Tomek object
    :type tomek: imblearn.Tomek
    :param smote_tomek: optional parameter to include SmoteTomek object
    :type smote_tomek: imblearn.SmoteTomek
   
    """

    def __init__(
        self,
        dataset,
        target,
        sampling_strategy="auto",
        random_state=None,
        smote_tomek=None,
        smote=None,
        tomek=None,
    ):

        if dataset.empty:
            raise ValueError(_MESSAGE_DATASET_NOT_PROVIDED)
        else:
            self.dataset = dataset

        if target is None:
            raise ValueError(_MESSAGE_TARGET_NOT_PROVIDED)
        elif type(target) is int:
            features = dataset.columns.values.tolist()
            self.target = features[target]
            self.target_index = target
        else:
            self.target_index = dataset.columns.get_loc(target)
            self.target = target

        self.sampling_strategy = sampling_strategy

        # set random seed if it is not provided by the caller
        if random_state is None:
            self.random_state = np.random.randint(1, 100)
        else:
            self.random_state = random_state

        # create object if it is not provided by the caller
        self.smote_tomek = smote_tomek
        self.smote = smote
        self.tomek = tomek

        # make sure the min number of class samples is equal or higher to the k_neighbors
        self.min_neighbors = (
            self.dataset.iloc[:, self.target_index].value_counts().min()
        )
        nn_k = None
        if isinstance(self.min_neighbors, Integral):
            nn_k = NearestNeighbors(n_neighbors=self.min_neighbors)
        elif isinstance(self.min_neighbors, KNeighborsMixin):
            nn_k = clone(self.min_neighbors)
        else:
            raise ValueError(
                f"{nn_k} has to be one of {[int, KNeighborsMixin]}. "
                f"Got {type(self.min_neighbors)} instead."
            )

        # nn_k = NearestNeighbors(n_neighbors=self.min_neighbors)
        if self.smote is not None:
            if self.min_neighbors < self.smote.k_neighbors:
                if nn_k.n_neighbors == 1:  # special case, not supported by SMOTE
                    raise ValueError(_MESSAGE_N_SAMPLE_LESS_K_NEIGHBOR)
                self.smote.k_neighbors = nn_k

    # Split the target and features
    def _split_target(self):
        try:
            self.target_index = self.dataset.columns.get_loc(self.target)
            X = self.dataset.drop(self.dataset.columns[self.target_index], axis=1)
            y = self.dataset.iloc[:, self.target_index]
            return X, y
        except IndexError:
            raise IndexError(_MESSAGE_INDEX_NOT_FOUND)

    # Combine over- and under-sampling using imblearn SMOTETomek
    def _rebalance_smotetomek(self):
        X_original, y_original = self._split_target()
        X, y = self.smote_tomek.fit_resample(X_original, y_original)

        X.insert(self.target_index, self.target, y, True)
        return X

    #  Under-sampling using imblearn by removing Tomek’s links
    # and then
    # Perform over-sampling using imblearn SMOTE

    def _rebalance_smote_tomek(self):
        X_original, y_original = self._split_target()
        X, y = self.smote_tomek.fit_resample(X_original, y_original)

        X.insert(self.target_index, self.target, y, True)
        return X

    # Perform over-sampling using imblearn SMOTE

    def _rebalance_smote(self):
        X_original, y_original = self._split_target()
        X, y = self.smote.fit_resample(X_original, y_original)

        X.insert(self.target_index, self.target, y, True)
        return X

    # Under-sampling using imblearn by removing Tomek’s links.

    def _rebalance_tomeklinks(self):
        X_original, y_original = self._split_target()
        X, y = self.tomek.fit_resample(X_original, y_original)

        X.insert(self.target_index, self.target, y, True)
        return X

    def rebalance(self):
        if self.smote_tomek is not None:
            return self._rebalance_smotetomek()
        elif self.smote is not None and self.tomek is not None:
            self.smote_tomek = SMOTETomek(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                smote=self.smote,
                tomek=self.tomek,
            )
            return self._rebalance_smote_tomek()
        elif self.smote is not None:
            return self._rebalance_smote()
        elif self.tomek is not None:
            return self._rebalance_tomeklinks()
        else:
            nn_k = NearestNeighbors(n_neighbors=self.min_neighbors)
            smote = SMOTE(random_state=self.random_state, k_neighbors=nn_k)
            self.smote_tomek = SMOTETomek(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                smote=smote,
            )
            return self._rebalance_smotetomek()
