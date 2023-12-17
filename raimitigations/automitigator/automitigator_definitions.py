class AutoMitigatorDefinitions:
    search_space_key = "search_space"

    mitigations_key = "mitigations"
    mitigation_name_key = "name"
    mitigation_type_key = "type"

    cohort_key = "cohort"
    all_cohort = "all"

    synthesizer = "synthesizer"
    synthesizer_epochs_key = "epochs"
    synthesizer_model_key = "model"

    rebalancer = "rebalancer"
    rebalancer_strategy_key = "strategy"

    scaler = "scaler"
    scaler_name_key = "scaler_name"
    standard_scaler = "standard_scaler"
    robust_scaler = "robust_scaler"
    quantile_scaler = "quantile_scaler"
    power_scaler = "power_scaler"
    normalize_scaler = "normalize_scaler"
    minmax_scaler = "minmax_scaler"

    imputer = "imputer"
    imputer_name_key = "imputer_name"
    basic_imputer = "basic"
    iterative_imputer = "iterative"
    knn_imputer = "knn"

    feature_selector = "feature_selector"
    selector_name_key = "selector_name"
    sequential_selector = "sequential_selector"
    correlated_feature_selector = "correlated_feature_selector"
    cfs_num_corr_th_key = "num_corr_th"
    cfs_num_pvalue_th_key = "num_pvalue_th"
    cfs_cat_corr_th_key = "cat_corr_th"
    cfs_cat_pvalue_th_key = "cat_pvalue_th"
    catboost_selector = "catboost_selector"
    cs_test_size_key = "test_size"
    cs_algorithm_key = "algorithm"
    cs_steps_key = "steps"

    no_mitigation = "nomitigation"

    results_loss_key = "loss"
    results_automl_key = "automl"
    results_pipeline_key = "pipeline"
