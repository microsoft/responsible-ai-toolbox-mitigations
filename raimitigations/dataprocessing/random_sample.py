# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


_MESSAGE_DATASET_NOT_PROVIDED = "A panda dataframe dataset is a required parameter."
_MESSAGE_TARGET_NOT_PROVIDED = "the dataset target is a required parameter."
_MESSAGE_INDEX_NOT_FOUND = "Index is not found."
_MESSAGE_SAMPLE_SIZE_NOT_PROVIDED = (
    "An numeric percentage value representing the sample size ( between 0 and 1) ."
)


class RandomSample:

    """

    :param dataset: the dataframe that the user will apply sampling on
    :type dataset: pd.Dataframe
    :param target: the name of the target column or the index of the target column
    :type target: str, int
    :param sample_size: the number of samples to include in the random sample
    :type sample_size: int
    :param stratify: if not None, data is split in a stratified fashion, using this as the class targets
    :type stratify: array-like

    """

    def __init__(
        self,
        dataset,
        target,
        sample_size,
        categorical_features=True,
        drop_null=True,
        drop_duplicates=False,
        stratify=False,
    ):
        if dataset is None or dataset.empty:
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

        if sample_size is None:
            raise ValueError(_MESSAGE_SAMPLE_SIZE_NOT_PROVIDED)
        else:
            self.sample_size = sample_size

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

    def _random_sample(self):

        # handle duplicates
        if self.drop_duplicates:
            self.dataset = self.dataset.drop_duplicates()

        # handle null values
        if self.drop_null:
            self.dataset.dropna(axis=0, inplace=True)
        else:
            self.dataset.fillna(self.dataset.mean(), inplace=True)

        # OneHotEncoder for categorical features
        if self.categorical_features:
            self.dataset = pd.get_dummies(self.dataset, dummy_na=True, drop_first=False)

        # set random seed
        random_state = np.random.randint(1, 100)

        # handle the stratify option
        if self.stratify:
            X, y_stratify = self._split_target()
        else:
            y_stratify = None

        # split data and return a random data sample
        data_sample, _ = train_test_split(
            self.dataset,
            train_size=self.sample_size,
            random_state=random_state,
            stratify=y_stratify,
        )

        return data_sample

    def random_sample(self):
        """
        Returns a random sample of the dataset
        """
        return self._random_sample()
