# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


_MESSAGE_DATASET_NOT_PROVIDED = "A panda dataframe dataset is a required parameter."
_MESSAGE_TARGET_NOT_PROVIDED = "the dataset target is a required parameter."
_MESSAGE_INDEX_NOT_FOUND = "Index is not found."
_MESSAGE_TRAIN_SIZE_NOT_PROVIDED = "An numeric percentage value representing the Training data size ( between 0 and 1) ."


class Split:

    r"""

    :param dataset: the dataframe that the user will apply splitting on
    :type dataset: pd.Dataframe
    :param target: the name of the target column or the index of the target column
    :type target: str, int
    :param train_size: the ratio of the training data split
    :type train_size: float
    :param random_state: seed to control the randomization of the algorithm
    :type random_state: int
    :param categorical_features: a flag indicating the presence of categorical features in the dataset
    :type categorical_features: boolean
    :param stratify: if not None, data is split in a stratified fashion, using this as the class targets
    :type stratify: array-like
    
    """

    def __init__(
        self,
        dataset,
        target,
        train_size,
        random_state=None,
        categorical_features=True,
        drop_null=True,
        drop_duplicates=False,
        stratify=False,
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

        self.train_size = train_size

        if train_size is None:
            raise ValueError(_MESSAGE_TRAIN_SIZE_NOT_PROVIDED)
        else:
            self.train_size = train_size

        # set random seed if it is not provided by the caller
        if random_state is None:
            self.random_state = np.random.randint(1, 100)
        else:
            self.random_state = random_state

        self.categorical_features = categorical_features
        self.drop_null = drop_null
        self.drop_duplicates = drop_duplicates
        self.stratify = stratify

    # Split the target and features
    def _split_target(self):
        try:
            self.target_index = self.dataset.columns.get_loc(self.target)
            X = self.dataset.drop(self.dataset.columns[self.target_index], axis=1)
            y = self.dataset.iloc[:, self.target_index]
            return X, y
        except IndexError:
            raise IndexError(_MESSAGE_INDEX_NOT_FOUND)

    def _split(self):
        # handle duplicates
        if self.drop_duplicates:
            self.dataset = self.dataset.drop_duplicates()

        # handle null values
        if self.drop_null:
            self.dataset.dropna(axis=0, inplace=True)
        else:
            self.dataset.fillna(self.dataset.mean(), inplace=True)

        # handle the stratify option
        if self.categorical_features:
            # OneHotEncoder for categorical features
            self.dataset = pd.get_dummies(
                self.dataset, dummy_na=True, drop_first=False, prefix_sep="-"
            )

        # set random seed if it is not provided by the caller
        if self.random_state == None:
            self.random_state = np.random.randint(1, 100)

        # handle the stratify option
        if self.stratify:
            X, y_stratify = self._split_target()
        else:
            y_stratify = None

        # split data into training and testing datasets
        train, test = train_test_split(
            self.dataset,
            train_size=self.train_size,
            random_state=self.random_state,
            stratify=y_stratify,
        )

        return train, test

    def split(self):
        return self._split()
