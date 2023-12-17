from flaml import tune
from .automitigator_definitions import AutoMitigatorDefinitions as amd


class SearchSpaceBuilder:
    """
    SearchSpaceBuilder is a class for building the search space for the AutoML
    hyperparameter search.
    All potential mitigations are included in the search space. The search space is modeled
    as a hyperparameter optimization problem, where the hyperparameters are each of the mitigation's
    parameters and the values are the possible values for each parameter.

    :param int max_mitigations: The maximum number of mitigations to be applied to the dataset.
    :param str task: The task to be performed by the AutoML.
    """

    def __init__(self, max_mitigations: int = 1, task: str = "classification") -> None:
        self._max_mitigations = max_mitigations
        self._task = task

    def build(self):
        """
        Build the search space.

        :return: The search space.
        :rtype: dict
        """

        synthesizer = {
            amd.mitigation_name_key: amd.synthesizer,
            amd.synthesizer_epochs_key: tune.randint(lower=200, upper=700),
            amd.synthesizer_model_key: tune.randint(
                lower=0, upper=4
            ),  # 0 - ctgan, 1 - tvae, 2 - copula, 3 - copula_gan
        }

        rebalancer = {
            amd.mitigation_name_key: amd.rebalancer,
            amd.mitigation_type_key: tune.randint(lower=0, upper=3),  # 0 - oversample, 1 - undersample, 2 - both
            amd.rebalancer_strategy_key: tune.randint(
                lower=0, upper=4
            ),  # 0 - majority, 1 - not minority, 2 - not majority, 3 - all
        }

        standard_scaler = {amd.scaler_name_key: amd.standard_scaler}

        robust_scaler = {amd.scaler_name_key: amd.robust_scaler}

        quantile_scaler = {amd.scaler_name_key: amd.quantile_scaler}

        power_scaler = {amd.scaler_name_key: amd.power_scaler}

        normalize_scaler = {amd.scaler_name_key: amd.normalize_scaler}

        minmax_scaler = {amd.scaler_name_key: amd.minmax_scaler}

        scaler = {
            amd.mitigation_name_key: amd.scaler,
            amd.mitigation_type_key: tune.choice(
                [standard_scaler, robust_scaler, quantile_scaler, power_scaler, normalize_scaler, minmax_scaler]
            ),
        }

        basic_imputer = {amd.imputer_name_key: amd.basic_imputer}

        iterative_imputer = {amd.imputer_name_key: amd.iterative_imputer}

        knn_imputer = {amd.imputer_name_key: amd.knn_imputer}

        imputer = {
            amd.mitigation_name_key: amd.imputer,
            amd.mitigation_type_key: tune.choice([basic_imputer, iterative_imputer, knn_imputer]),
        }

        sequential_selector = {
            amd.selector_name_key: amd.sequential_selector,
        }

        correlated_feature_selector = {
            amd.selector_name_key: amd.correlated_feature_selector,
            amd.cfs_num_corr_th_key: tune.uniform(lower=0.7, upper=0.9),
            amd.cfs_num_pvalue_th_key: tune.uniform(lower=0.01, upper=0.10),
            amd.cfs_cat_corr_th_key: tune.uniform(lower=0.7, upper=0.9),
            amd.cfs_cat_pvalue_th_key: tune.uniform(lower=0.01, upper=0.10),
        }

        catboost_selector = {
            amd.selector_name_key: amd.catboost_selector,
            amd.cs_test_size_key: tune.uniform(lower=0.1, upper=0.7),
            amd.cs_algorithm_key: tune.choice(["predict", "loss", "shap"]),
            amd.cs_steps_key: tune.randint(lower=1, upper=100),
        }

        feature_selector = {
            amd.mitigation_name_key: amd.feature_selector,
            "type": tune.choice([sequential_selector, correlated_feature_selector, catboost_selector]),
        }

        nomitigation = {amd.mitigation_name_key: amd.no_mitigation}

        mitigations = {}
        mitigation_choices = [feature_selector, imputer, scaler, synthesizer, nomitigation]
        if self._task == "classification":
            mitigation_choices.append(rebalancer)

        for i in range(self._max_mitigations):
            mitigations[f"action{i}"] = tune.choice(mitigation_choices)

        all_search_space = {amd.cohort_key: amd.all_cohort, amd.mitigations_key: mitigations}

        # Add additional search space options like cohort based ones to the list
        search_space_list = [all_search_space]

        return {amd.search_space_key: tune.choice(search_space_list)}
