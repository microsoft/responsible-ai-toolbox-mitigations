from flaml import tune


class SearchSpaceBuilder:
    """
    SearchSpaceBuilder is a class for building the search space for the AutoML
    hyperparameter search.

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
            "name": "synthesizer",
            "epochs": tune.randint(lower=200, upper=700),
            "model": tune.randint(lower=0, upper=4),  # 0 - ctgan, 1 - tvae, 2 - copula, 3 - copula_gan
        }

        rebalancer = {
            "name": "rebalancer",
            "type": tune.randint(lower=0, upper=3),  # 0 - oversample, 1 - undersample, 2 - both
            "strategy": tune.randint(lower=0, upper=4),  # 0 - majority, 1 - not minority, 2 - not majority, 3 - all
        }

        standard_scaler = {"scaler_name": "standard_scaler"}

        robust_scaler = {"scaler_name": "robust_scaler"}

        quantile_scaler = {"scaler_name": "quantile_scaler"}

        power_scaler = {"scaler_name": "power_scaler"}

        normalize_scaler = {"scaler_name": "normalize_scaler"}

        minmax_scaler = {"scaler_name": "minmax_scaler"}

        scaler = {
            "name": "scaler",
            "type": tune.choice(
                [standard_scaler, robust_scaler, quantile_scaler, power_scaler, normalize_scaler, minmax_scaler]
            ),
        }

        basic_imputer = {"imputer_name": "basic"}

        iterative_imputer = {"imputer_name": "iterative"}

        knn_imputer = {"imputer_name": "knn"}

        imputer = {"name": "imputer", "type": tune.choice([basic_imputer, iterative_imputer, knn_imputer])}

        sequential_selector = {
            "selector_name": "sequential_selector",
        }

        correlated_feature_selector = {
            "selector_name": "correlated_feature_selector",
            "num_corr_th": tune.uniform(lower=0.7, upper=0.9),
            "num_pvalue_th": tune.uniform(lower=0.01, upper=0.10),
            "cat_corr_th": tune.uniform(lower=0.7, upper=0.9),
            "cat_pvalue_th": tune.uniform(lower=0.01, upper=0.10),
        }

        catboost_selector = {
            "selector_name": "catboost_selector",
            "test_size": tune.uniform(lower=0.1, upper=0.7),
            "algorithm": tune.choice(["predict", "loss", "shap"]),
            "steps": tune.randint(lower=1, upper=100),
        }

        feature_selector = {
            "name": "feature_selector",
            "type": tune.choice([sequential_selector, correlated_feature_selector, catboost_selector]),
        }

        nomitigation = {"name": "nomitigation"}

        mitigations = {}
        mitigation_choices = [feature_selector, imputer, scaler, synthesizer, nomitigation]
        if self._task == "classification":
            mitigation_choices.append(rebalancer)

        for i in range(self._max_mitigations):
            mitigations[f"action{i}"] = tune.choice(mitigation_choices)

        all_search_space = {"cohort": "all", "mitigations": mitigations}

        # Add additional search space options like cohort based ones to the list
        search_space_list = [all_search_space]

        return {"search_space": tune.choice(search_space_list)}
