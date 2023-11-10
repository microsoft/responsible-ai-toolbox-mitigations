from raimitigations.utils import split_data
from raimitigations.automitigator import AutoMitigator

# -----------------------------------
def test_other_errors(df_breast_cancer, label_name_bc):
    df = df_breast_cancer
    train_x, test_x, train_y, test_y = split_data(df_breast_cancer, label_name_bc, test_size=0.25)

    automl_args = {"task": "classification", "time_budget": 20, "metric": "log_loss", "verbose": False, "early_stop": True}
    autoMitigator = AutoMitigator(max_mitigations=1, num_samples=20, use_ray=False, automl_args=automl_args) #, cohort_def=["c1", "c2", "c3"])
    autoMitigator.fit(train_x, train_y)
    _ = autoMitigator.predict(test_x)
    _ = autoMitigator.predict_proba(test_x)

    assert autoMitigator._pipeline is not None