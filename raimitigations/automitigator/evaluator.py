import math
import traceback
import raimitigations.dataprocessing as dp
from flaml import AutoML
from imblearn.pipeline import Pipeline

from .mitigation_actions import MitigationActions
from .automitigator_definitions import AutoMitigatorDefinitions as amd

class Evaluator:
    """
    Evaluates a given set of mitigations on a given dataset.
    """

    def __init__(self, automl_args=None) -> None:
        self.automl_args = automl_args
        self.pipeline_steps = []
        self.pipeline = None

    def _pipeline_append(self, step):
        """
        Append a step to the pipeline.
        Note: This is done to support older versions of sklearn < 1.1 (required to support python 3.7)
        which doesn't support an empty pipeline to be initialized nor appending a step to an existing pipeline
        without an estimator at the end.

        :param sklearn.pipeline.Pipeline pipeline: The pipeline to add the step to
        :param step: The step to add
        """
        self.pipeline_steps.append(step)

    def evaluate(self, train_x, train_y, search_config):
        """
        Evaluates a given set of mitigations on a given dataset.

        :param train_x: The training data
        :param train_y: The training labels
        :param dict search_config: The search configuration

        :return: The results of the evaluation
        :rtype: dict
        """
        # Sample config
        # config: {'search_space':
        #           {'cohort': 'all',
        #            'mitigations':
        #               {'action0': {'type': 0, 'strategy': 0, 'name': 'rebalancer'}}}}

        self.pipeline = None
        search_space = search_config[amd.search_space_key]
        cohort = search_space[amd.cohort_key]
        if cohort == amd.all_cohort:
            return self.mitigate_full_dataset(train_x, train_y, search_space)

    def _process_feature_selector(self, selector_type):
        """
        Process the feature selector

        :param dict selector_type: The feature selector configuration

        :raises ValueError: If the feature selector is unknown
        """
        selector_name = selector_type[amd.selector_name_key]
        if selector_name == amd.sequential_selector:
            self._pipeline_append((amd.sequential_selector, dp.SeqFeatSelection()))
        elif selector_name == amd.correlated_feature_selector:
            self._pipeline_append(
                (
                    amd.correlated_feature_selector,
                    dp.CorrelatedFeatures(
                        num_corr_th=selector_type[amd.cfs_num_corr_th_key],
                        num_pvalue_th=selector_type[amd.cfs_num_pvalue_th_key],
                        cat_corr_th=selector_type[amd.cfs_cat_corr_th_key],
                        cat_pvalue_th=selector_type[amd.cfs_cat_pvalue_th_key],
                    ),
                )
            )
        elif selector_name == amd.catboost_selector:
            self._pipeline_append(
                (
                    amd.catboost_selector,
                    dp.CatBoostSelection(
                        test_size=selector_type[amd.cs_test_size_key],
                        algorithm=selector_type[amd.cs_algorithm_key],
                        steps=selector_type[amd.cs_steps_key],
                        verbose=False,
                    ),
                )
            )

    def mitigate_full_dataset(self, train_x, train_y, search_space):
        """
        Evaluates a given set of mitigations on a given dataset.

        :param train_x: The training data
        :param train_y: The training labels
        :param dict search_space: The search configuration

        :return: The results of the evaluation
        :rtype: dict
        """

        mitigation_set = set()
        for mitigation in search_space[amd.mitigations_key]:
            config = search_space[amd.mitigations_key][mitigation]
            mitigation_name = config[amd.mitigation_name_key]

            # Skip if we've already seen this mitigation, except if it's nomitigation to allow for
            # combinations with fewer mitigations to be evaluated
            if (mitigation_name == amd.no_mitigation) or (mitigation_name not in mitigation_set):
                mitigation_set.add(mitigation_name)
            else:
                continue

            if mitigation_name == amd.synthesizer:
                self._pipeline_append(
                    (amd.synthesizer, MitigationActions.get_synthesizer(config[amd.synthesizer_epochs_key], config[amd.synthesizer_model_key]))
                )
            elif mitigation_name == amd.rebalancer:
                self._pipeline_append(
                    (amd.rebalancer, MitigationActions.get_rebalancer(config[amd.mitigation_type_key], config[amd.rebalancer_strategy_key]))
                )
            elif mitigation_name == amd.scaler:
                self._process_scaler(config[amd.mitigation_type_key])
            elif mitigation_name == amd.imputer:
                self._process_imputer(config[amd.mitigation_type_key])
            elif mitigation_name == amd.feature_selector:
                self._process_feature_selector(config[amd.mitigation_type_key])
            elif mitigation_name == amd.no_mitigation:
                continue

        fit_results = self._fit_model(train_x, train_y)
        fit_results["search_space"] = search_space
        return fit_results

    def _process_imputer(self, imputer_type):
        """
        Process the imputer

        :param dict imputer_type: The imputer configuration

        :raises ValueError: If the imputer is unknown
        """
        imputer_name = imputer_type[amd.imputer_name_key]
        if imputer_name == amd.basic_imputer:
            self._pipeline_append((amd.basic_imputer, dp.BasicImputer()))
        elif imputer_name == amd.iterative_imputer:
            self._pipeline_append((amd.iterative_imputer, dp.IterativeDataImputer()))
        elif imputer_name == amd.knn_imputer:
            self._pipeline_append((amd.knn_imputer, dp.KNNDataImputer()))

    def _process_scaler(self, scaler_type):
        """
        Process the scaler

        :param dict scaler_type: The scaler configuration

        :raises ValueError: If the scaler is unknown
        """
        scaler_name = scaler_type[amd.scaler_name_key]
        if scaler_name == amd.standard_scaler:
            self._pipeline_append((amd.standard_scaler, dp.DataStandardScaler()))
        elif scaler_name == amd.robust_scaler:
            self._pipeline_append((amd.robust_scaler, dp.DataRobustScaler()))
        elif scaler_name == amd.quantile_scaler:
            self._pipeline_append((amd.quantile_scaler, dp.DataQuantileTransformer()))
        elif scaler_name == amd.power_scaler:
            self._pipeline_append((amd.power_scaler, dp.DataPowerTransformer()))
        elif scaler_name == amd.normalize_scaler:
            self._pipeline_append((amd.normalize_scaler, dp.DataNormalizer()))
        elif scaler_name == amd.minmax_scaler:
            self._pipeline_append((amd.minmax_scaler, dp.DataMinMaxScaler()))

    def _fit_model(self, train_x, train_y):
        """
        Fit the model

        :param train_x: The training data
        :param train_y: The training labels

        :return: The results of the evaluation
        :rtype: dict

        :raises ValueError: If the scaler is unknown

        :return: Dictionary containing best loss, automl object and pipeline used
        :rtype: dict
        """
        automl = AutoML(**self.automl_args)
        self._pipeline_append(("automl", automl))
        self.pipeline = Pipeline(self.pipeline_steps)

        try:
            self.pipeline.fit(train_x, train_y)
            loss = automl.best_loss
        except Exception as ex:
            print(f"Evaluating pipeline {self.pipeline} caused error {ex} with trace {traceback.format_exc()}")
            loss = math.inf

        return {amd.results_loss_key: loss, amd.results_automl_key: automl, amd.results_pipeline_key: self.pipeline}
