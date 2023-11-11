import numpy as np
from functools import partial
from flaml import tune
from typing import Union, List
from pandas import DataFrame
from sklearn.base import BaseEstimator

from .searchspacebuilder import SearchSpaceBuilder
from .evaluator import Evaluator


class AutoMitigator(BaseEstimator):
    """
    AutoMitigator is a class for automatically building and tuning a pipeline
    for mitigating bias in a dataset.

    :param int max_mitigations: The maximum number of mitigations to be applied to the dataset.
    :param int num_samples: The number of samples to be generated for each hyperparameter configuration.
    :param int time_budget_s: The time budget in seconds for the hyperparameter search.
    :param bool use_ray: Whether to use Ray for parallelism.
    :param dict tune_args: Keyword arguments to be passed to the tune.run method.
    :param dict automl_args: Keyword arguments to be passed to the AutoML constructor.
    """

    def __init__(
        self,
        max_mitigations: int = 1,
        num_samples=5,
        time_budget_s=None,
        use_ray: bool = True,
        tune_args: dict = {},
        automl_args: dict = {},
    ):
        self.max_mitigations = max_mitigations
        self.num_samples = num_samples
        self.time_budget_s = time_budget_s
        self.use_ray = use_ray
        self.tune_args = tune_args
        self.automl_args = automl_args

    def fit(self, X_train, y_train):
        """
        Fit the model to the training data.

        :param X_train: The training input samples.
        :param y_train: The target values.

        :return: Automitigator object.
        :rtype: raimitigations.automitigator.AutoMitigator

        :raises ValueError: If the number of mitigations is less than 1.
        :raises ValueError: If the number of samples is less than 1.
        :raises ValueError: If it is not able to fit a model with the mitigations applied.
        """
        if self.max_mitigations < 1:
            raise ValueError("At least one mitigation is necessary")
        if self.num_samples < 1:
            raise ValueError("num_samples should be at least 1")

        if self.automl_args is None or len(self.automl_args) == 0:
            self.automl_args = {"task": "classification", "time_budget": 30, "metric": "log_loss", "early_stop": True}

        task = self.automl_args["task"] if "task" in self.automl_args else "classification"
        search_space = SearchSpaceBuilder(self.max_mitigations, task).build()
        evaluator = Evaluator(automl_args=self.automl_args)

        analysis = tune.run(
            partial(evaluator.evaluate, X_train, y_train),
            config=search_space,
            metric="loss",
            mode="min",
            num_samples=self.num_samples,
            time_budget_s=self.time_budget_s,
            search_alg="BlendSearch",
            use_ray=self.use_ray,
            **self.tune_args,
        )

        if analysis is None or analysis.best_trial is None:
            raise ValueError("Failed to fit. Try adjusting the parameters.")

        self._automl = analysis.best_result["automl"]
        self._pipeline = analysis.best_result["pipeline"]
        self._search_space = analysis.best_result["search_space"]

        return self

    def predict(
        self,
        X: Union[np.array, DataFrame, List[str], List[List[str]]],
        **pred_kwargs,
    ):
        """
        Predict the class for each sample in X.

        :param Union[np.array, DataFrame, List[str], List[List[str]]] X: The input samples.
        :param dict pred_kwargs: Keyword arguments to be passed to the predict method of the pipeline.

        :return: The predicted classes.

        :raises ValueError: If model has not been fit before.
        """
        if not hasattr(self, "_pipeline"):
            raise ValueError("You must fit the model before predicting")

        return self._pipeline.predict(X, **pred_kwargs)

    def predict_proba(
        self,
        X: Union[np.array, DataFrame, List[str], List[List[str]]],
        **pred_kwargs,
    ):
        """
        Predict the probability of each class for each sample in X.

        :param X: The input samples.
        :param dict pred_kwargs: Keyword arguments to be passed to the predict method of the pipeline.

        :return: The predicted probabilities.

        :raises ValueError: If model has not been fit before.
        """
        if not hasattr(self, "_pipeline"):
            raise ValueError("You must fit the model before predicting")

        return self._pipeline.predict_proba(X, **pred_kwargs)
