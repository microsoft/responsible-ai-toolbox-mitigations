# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.


import pandas as pd
import numpy as np
from enum import Enum

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import Normalizer

_MESSAGE_DATASET_NOT_PROVIDED = "A panda dataframe dataset is a required parameter."
_MESSAGE_TARGET_NOT_PROVIDED = "The dataset target is a required parameter."
_MESSAGE_INDEX_NOT_FOUND = "Index is not found."


class Transformer:

    r"""
    :param dataset: the dataframe that the user will apply splitting on
    :type dataset: pd.Dataframe
    :param target: the name of the target column or the index of the target column
    :type target: str, int
    :param transformer_type: the type of transformer to apply to the data
    :type transformer_type:  TransformerType 
    :param transform_features: the features to be transformer using those transformers
    :type transform_features: array-like
    :param random_state: seed to control the randomization of the algorithm
    :param method: method of transformation for the PowerTransformer
    :param output_distribution: marginal distribution for the transformed data

    """

    class TransformerType(Enum):
        StandardScaler = 1
        MinMaxScaler = 2
        RobustScaler = 3
        PowerTransformer = 4
        QuantileTransformer = 5
        Normalizer = 6

    def __init__(
        self,
        dataset,
        target,
        transformer_type,
        transform_features=None,
        random_state=None,
        method="yeo-johnson",
        output_distribution="uniform",
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

        self.transform_features = transform_features

        # set random seed if it is not provided by the caller
        if random_state is None:
            self.random_state = np.random.randint(1, 100)
        else:
            self.random_state = random_state

        self.transformer_type = transformer_type
        self.method = method
        self.output_distribution = output_distribution

    # Split the target and features
    def _split_target(self):
        try:
            X = self.dataset.drop(self.dataset.columns[self.target_index], axis=1)
            y = self.dataset.iloc[:, self.target_index]
            return X, y
        except IndexError:
            raise IndexError(_MESSAGE_INDEX_NOT_FOUND)

    # Split features
    def _split_features(self):
        try:
            if all(isinstance(x, int) for x in self.transform_features):
                X_no_transform = self.dataset.drop(
                    self.dataset.columns[self.transform_features], axis=1
                )
                X_transform = self.dataset.iloc[:, self.transform_features]
            else:
                X_no_transform = self.dataset.drop(self.transform_features, axis=1)
                X_transform = self.dataset[self.transform_features]

            return X_transform, X_no_transform
        except IndexError:
            raise IndexError(_MESSAGE_INDEX_NOT_FOUND)

    # scaler = MinMaxScaler(feature_range=(1, 2))
    def _transform_data(self, X):
        pipe_cfg = {"num_cols": X.dtypes[X.dtypes != "object"].index.values.tolist()}

        if self.transformer_type.name == self.TransformerType.StandardScaler.name:
            num_pipe = Pipeline(
                [
                    ("num_imputer", SimpleImputer(strategy="median")),
                    ("num_scaler", StandardScaler()),
                ]
            )
        elif self.transformer_type.name == self.TransformerType.MinMaxScaler.name:
            power = PowerTransformer(method=self.method)
            minmax_scaler = MinMaxScaler()
            num_pipe = Pipeline(steps=[("s", minmax_scaler), ("p", power)])

        elif self.transformer_type.name == self.TransformerType.RobustScaler.name:
            power = PowerTransformer(method=self.method)
            robust_scaler = RobustScaler(quantile_range=(25, 75))
            num_pipe = Pipeline(steps=[("s", robust_scaler), ("p", power)])

        elif self.transformer_type.name == self.TransformerType.PowerTransformer.name:
            power = PowerTransformer(method=self.method)
            minmax_scaler = MinMaxScaler()
            if self.method == "yeo-johnson":
                num_pipe = Pipeline(steps=[("p", power)])
            else:
                num_pipe = Pipeline(steps=[("s", minmax_scaler), ("p", power)])

        elif (
            self.transformer_type.name == self.TransformerType.QuantileTransformer.name
        ):
            num_pipe = Pipeline(
                [
                    ("num_imputer", SimpleImputer(strategy="median")),
                    ("quantile", QuantileTransformer()),
                ]
            )
        elif self.transformer_type.name == self.TransformerType.Normalizer.name:
            num_pipe = Pipeline(
                [
                    ("num_imputer", SimpleImputer(strategy="median")),
                    ("Normalizer", Normalizer()),
                ]
            )

        feat_pipe = ColumnTransformer([("num_pipe", num_pipe, pipe_cfg["num_cols"])])

        return feat_pipe.fit_transform(X)

    def transform(self):

        if self.transform_features is not None:
            # Split features
            X_transform, X_no_transform = self._split_features()
            # OneHotEncoder for categorical features
            X = pd.get_dummies(X_transform, drop_first=False)
            col_names = list(X_no_transform.columns) + list(X.columns)
            # Transform data
            X = self._transform_data(X)
            # merge transformed and non-transformed data
            X_data = np.concatenate([X_no_transform, X], axis=1)
        else:
            # split features and target
            X_transform, y_original = self._split_target()
            # OneHotEncoder for categorical features
            X = pd.get_dummies(X_transform, drop_first=False)
            # save columns
            col_names = X.columns
            # Transform data
            X = self._transform_data(X)
            # insert target data to the beginning of the dataset
            y = y_original.to_numpy()
            X_data = np.insert(X, 0, y, axis=1)
            col_names = [self.target] + list(col_names)
        X_df = pd.DataFrame(X_data, columns=col_names)
        return X_df
