from raimitigations.utils import split_data
from raimitigations.automitigator import AutoMitigator
import raimitigations.dataprocessing as dp
import pytest

def _prepare_data(df, label_col):
    # Set the order that the ordinal encoder should use
    age_order = df['age'].unique()
    age_order.sort()
    tumor_size_order = df['tumor-size'].unique()
    tumor_size_order.sort()
    inv_nodes_order = df['inv-nodes'].unique()
    inv_nodes_order.sort()

    # Encode 'tumor-size', 'Class', and 'inv-nodes' using ordinal encoding
    enc_ord = dp.EncoderOrdinal(col_encode=["age", "tumor-size", "inv-nodes", "Class"],
                                categories={"age":age_order,
                                            "tumor-size":tumor_size_order, 
                                            "inv-nodes":inv_nodes_order}
                            )
    enc_ord.fit(df)
    proc_df = enc_ord.transform(df)
 
    # Encode the remaining categorical columns using One-Hot Encoding
    enc_ohe = dp.EncoderOHE()
    enc_ohe.fit(proc_df)
    proc_df = enc_ohe.transform(proc_df)
    return split_data(proc_df, label_col, test_size=0.25)

def _test_mitigation(train_x, train_y, test_x, mitigation):
    config = {
        'search_space':
        {
            'cohort': 'all',
            'mitigations': { 'action0': mitigation}
        }
    }

    tune_args = {'points_to_evaluate':[config]}
    autoMitigator = AutoMitigator(max_mitigations=1, num_samples=1, use_ray=False, tune_args=tune_args)
    autoMitigator.fit(train_x, train_y)
    _ = autoMitigator.predict(test_x)
    _ = autoMitigator.predict_proba(test_x)

    assert autoMitigator._pipeline is not None

# -----------------------------------
def test_duplicate_mitigations(df_breast_cancer, label_name_bc):
    train_x, test_x, train_y, test_y = _prepare_data(df_breast_cancer, label_name_bc)

    config = {
        'search_space':
        {
            'cohort': 'all',
            'mitigations': { 'action0': {'type': 0, 'strategy': 0, 'name': 'rebalancer'},
                             'action1': {'type': 0, 'strategy': 0, 'name': 'rebalancer'}}
        }
    }

    tune_args = {'points_to_evaluate':[config]}
    autoMitigator = AutoMitigator(max_mitigations=2, num_samples=1, use_ray=False, tune_args=tune_args)
    autoMitigator.fit(train_x, train_y)
    _ = autoMitigator.predict(test_x)

    assert autoMitigator._pipeline is not None
    
# -----------------------------------
def test_auto_mitigator_starting_points(df_breast_cancer, label_name_bc):
    train_x, test_x, train_y, test_y = _prepare_data(df_breast_cancer, label_name_bc)

    _test_mitigation(train_x, train_y, test_x, {'name': 'nomitigation'})

    _test_mitigation(train_x, train_y, test_x, {'type': 0, 'strategy': 0, 'name': 'rebalancer'})
    _test_mitigation(train_x, train_y, test_x, {'type': 1, 'strategy': 0, 'name': 'rebalancer'})

    _test_mitigation(train_x, train_y, test_x, {'epochs': 200, 'model': 0, 'name': 'synthesizer'})

    _test_mitigation(train_x, train_y, test_x, {"name": "scaler", "type": {"scaler_name": "standard_scaler"}})
    _test_mitigation(train_x, train_y, test_x, {"name": "scaler", "type": {"scaler_name": "robust_scaler"}})
    _test_mitigation(train_x, train_y, test_x, {"name": "scaler", "type": {"scaler_name": "quantile_scaler"}})
    _test_mitigation(train_x, train_y, test_x, {"name": "scaler", "type": {"scaler_name": "power_scaler"}})
    _test_mitigation(train_x, train_y, test_x, {"name": "scaler", "type": {"scaler_name": "normalize_scaler"}})
    _test_mitigation(train_x, train_y, test_x, {"name": "scaler", "type": {"scaler_name": "minmax_scaler"}})

    _test_mitigation(train_x, train_y, test_x, {"name": "imputer", "type": {"imputer_name": "basic"}})
    _test_mitigation(train_x, train_y, test_x, {"name": "imputer", "type": {"imputer_name": "iterative"}})
    _test_mitigation(train_x, train_y, test_x, {"name": "imputer", "type": {"imputer_name": "knn"}})

    _test_mitigation(train_x, train_y, test_x, {"name": "feature_selector", "type": {"selector_name": "sequential_selector"}})
    _test_mitigation(train_x, train_y, test_x, {"name": "feature_selector", "type": {"selector_name": "sequential_selector"}})
    _test_mitigation(train_x, train_y, test_x, {"name": "feature_selector", "type": 
        {"selector_name": "correlated_feature_selector", "num_corr_th": 0.7, "num_pvalue_th": 0.01, 
            "cat_corr_th": 0.7, "cat_pvalue_th": 0.01}})
    _test_mitigation(train_x, train_y, test_x, {"name": "feature_selector", "type": 
        {"selector_name": "catboost_selector", "test_size": 0.1, "algorithm": "predict", "steps": 1}})

# -----------------------------------
def test_auto_mitigator(df_breast_cancer, label_name_bc):
    train_x, test_x, train_y, test_y = _prepare_data(df_breast_cancer, label_name_bc)

    autoMitigator = AutoMitigator(max_mitigations=1, num_samples=2, use_ray=False)
    autoMitigator.fit(train_x, train_y)
    _ = autoMitigator.predict(test_x)
    _ = autoMitigator.predict_proba(test_x)

    assert autoMitigator._pipeline is not None

# -----------------------------------
def test_errors(df_breast_cancer, label_name_bc):
    train_x, test_x, train_y, test_y = _prepare_data(df_breast_cancer, label_name_bc)

    with pytest.raises(ValueError):
        automitigator = AutoMitigator(max_mitigations=0, num_samples=5)
        automitigator.fit(train_x, train_y)

    with pytest.raises(ValueError):
        automitigator = AutoMitigator(max_mitigations=1, num_samples=0)
        automitigator.fit(train_x, train_y)

    with pytest.raises(ValueError):
        automitigator = AutoMitigator(max_mitigations=1, num_samples=5)
        automitigator.predict(test_x)

    with pytest.raises(ValueError):
        automitigator = AutoMitigator(max_mitigations=1, num_samples=5)
        automitigator.predict_proba(test_x)
