# Copyright (c) Microsoft Corporation and ErrorsMitigation contributors.


import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors


_MESSAGE_DATASET_NOT_PROVIDED = "A panda dataframe dataset is a required parameter."
_MESSAGE_TARGET_NOT_PROVIDED = "the dataset target is a required parameter."
_MESSAGE_INDEX_NOT_FOUND = "Index is not found."


class DataRebalance:

    r"""

    Parameters
    ----------
        dataset - Panda Data Frame. 
        target - The target column name or index (zero base)
        train_size – The training data split size.  
        random_state – Control the randomization of the algorithm. 
            ‘None’: the random number generator is the RandomState instance used by np.random.  
        categorical_features – A Boolean flag to indicates the presence of categorical features. It defaults to true. 

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

        # X_transform = self.dataset.iloc[:,self.transform_features]
        # make sure the min number of class samples is equal or higher to the k_neighbors
        self.min_neighbors = (
            self.dataset.iloc[:, self.target_index].value_counts().min()
        )
        nn_k = NearestNeighbors(n_neighbors=self.min_neighbors)
        if self.smote is not None:
            if self.min_neighbors < self.smote.k_neighbors:
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

    def Rebalance(self):
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
