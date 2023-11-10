import math
import traceback
import raimitigations.dataprocessing as dp
from flaml import AutoML
from imblearn.pipeline import Pipeline

from .mitigation_actions import MitigationActions

class Evaluator:
    """
    Evaluates a given set of mitigations on a given dataset.
    """
    def __init__(self, automl_args = None) -> None:
        self.automl_args = automl_args
        
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

        search_space = search_config['search_space']
        cohort = search_space['cohort']
        if cohort == "all":
            return self.mitigate_full_dataset(train_x, train_y, search_space)
        else:
            raise ValueError(f'Unknown cohort type {cohort}')

    def _process_feature_selector(self, pipeline, selector_type):
        """
        Process the feature selector

        :param pipeline: The pipeline to add the feature selector to
        :param dict selector_type: The feature selector configuration

        :raises ValueError: If the feature selector is unknown
        """
        selector_name = selector_type["selector_name"]
        if selector_name == "sequential_selector":
            pipeline.steps.append(('sequential_selector', dp.SeqFeatSelection()))
        elif selector_name == "correlated_feature_selector":
            pipeline.steps.append(('correlated_feature_selector', dp.CorrelatedFeatures(
                num_corr_th=selector_type['num_corr_th'],
                num_pvalue_th=selector_type['num_pvalue_th'],
                cat_corr_th=selector_type['cat_corr_th'],
                cat_pvalue_th=selector_type['cat_pvalue_th']
            )))
        elif selector_name == "catboost_selector":
            pipeline.steps.append(('catboost_selector', dp.CatBoostSelection(
                test_size=selector_type['test_size'],
                algorithm=selector_type['algorithm'],
                steps=selector_type['steps'],
                verbose=False
            )))

    def mitigate_full_dataset(self, train_x, train_y, search_space):
        """
        Evaluates a given set of mitigations on a given dataset.
        
        :param train_x: The training data
        :param train_y: The training labels
        :param dict search_space: The search configuration
        
        :return: The results of the evaluation
        :rtype: dict
        """

        pipeline = Pipeline([])

        mitigation_set = set()
        for mitigation in search_space['mitigations']:
            config = search_space['mitigations'][mitigation]
            mitigation_name = config['name']

            # Skip if we've already seen this mitigation, except if it's nomitigation to allow for 
            # combinations with fewer mitigations to be evaluated
            if (mitigation_name == "nomitigation") or (mitigation_name not in mitigation_set):
                mitigation_set.add(mitigation_name)
            else:
                continue

            if mitigation_name == "synthesizer":
                pipeline.steps.append(('synthesizer', MitigationActions.get_synthesizer(config["epochs"], config["model"])))
            elif mitigation_name == "rebalancer":
                pipeline.steps.append(('rebalancer', MitigationActions.get_rebalancer(config["type"], config["strategy"])))
            elif mitigation_name == "scaler":
                self._process_scaler(pipeline, config["type"])
            elif mitigation_name == "imputer":
                self._process_imputer(pipeline, config["type"])
            elif mitigation_name == "feature_selector":
                self._process_feature_selector(pipeline, config["type"])
            elif mitigation_name == "nomitigation":
                continue
            else:
                raise ValueError(f'Unknown mitigation {mitigation_name}')

        fit_results = self._fit_model(pipeline, train_x, train_y)
        fit_results["search_space"] = search_space
        return fit_results

    def _process_imputer(self, pipeline, imputer_type):
        """
        Process the imputer
        
        :param pipeline: The pipeline to add the imputer to
        :param dict imputer_type: The imputer configuration
        
        :raises ValueError: If the imputer is unknown
        
        :return: The pipeline with the imputer added
        :rtype: sklearn.pipeline.Pipeline
        """
        imputer_name = imputer_type["imputer_name"]
        if imputer_name == "basic":
            pipeline.steps.append(('basic_imputer', dp.BasicImputer()))
        elif imputer_name == "iterative":
            pipeline.steps.append(('iterative_imputer', dp.IterativeDataImputer()))
        elif imputer_name == "knn":
            pipeline.steps.append(('knn_imputer', dp.KNNDataImputer()))
        else:
            raise ValueError(f"Unknown imputer {imputer_name}")

    def _process_scaler(self, pipeline, scaler_type):
        """
        Process the scaler

        :param sklearn.pipeline.Pipeline pipeline: The pipeline to add the scaler to
        :param dict scaler_type: The scaler configuration
        
        :raises ValueError: If the scaler is unknown
        
        :return: The pipeline with the scaler added
        :rtype: sklearn.pipeline.Pipeline
        """
        scaler_name = scaler_type["scaler_name"]
        if scaler_name == "standard_scaler":
            pipeline.steps.append(('standard_scaler', dp.DataStandardScaler()))
        elif scaler_name == "robust_scaler":
            pipeline.steps.append(('robust_scaler', dp.DataRobustScaler()))
        elif scaler_name == "quantile_scaler":
            pipeline.steps.append(('quantile_scaler', dp.DataQuantileTransformer()))
        elif scaler_name == "power_scaler":
            pipeline.steps.append(('power_scaler', dp.DataPowerTransformer()))
        elif scaler_name == "normalize_scaler":
            pipeline.steps.append(('normalize_scaler', dp.DataNormalizer()))
        elif scaler_name == "minmax_scaler":
            pipeline.steps.append(('minmax_scaler', dp.DataMinMaxScaler()))
        else:
            raise ValueError(f"Unknown scaler {scaler_name}")

    def _fit_model(self, pipeline, train_x, train_y):
        """
        Fit the model
        
        :param sklearn.pipeline.Pipeline pipeline: The pipeline to add the scaler to
        :param train_x: The training data
        :param train_y: The training labels

        :return: The results of the evaluation
        :rtype: dict
        
        :raises ValueError: If the scaler is unknown
        
        :return: The pipeline with the scaler added
        :rtype: sklearn.pipeline.Pipeline
        """
        automl = AutoML(**self.automl_args)
        pipeline.steps.append(('automl', automl))

        try:
            pipeline.fit(train_x, train_y)
            loss = automl.best_loss
        except Exception as ex:
            print(f'Evaluating pipeline {pipeline} caused error {ex} with trace {traceback.format_exc()}')
            loss = math.inf

        return {
            "loss": loss,
            "automl": automl,
            "pipeline": pipeline
        }