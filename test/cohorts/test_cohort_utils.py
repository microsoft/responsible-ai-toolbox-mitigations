import pytest
import xgboost as xgb
from sklearn.pipeline import Pipeline

import raimitigations.dataprocessing as dp
from raimitigations.cohort import fetch_cohort_results

# -----------------------------------
def _get_model_class():
    model = xgb.XGBClassifier(
            objective="binary:logistic",
            learning_rate=0.1,
            n_estimators=30,
            max_depth=10,
            colsample_bytree=0.7,
            alpha=0.0,
            reg_lambda=10.0,
            nthreads=4,
            verbosity=0,
            use_label_encoder=False,
        )
    return model


# -----------------------------------
def _get_model_reg():
    model = xgb.XGBRegressor()
    return model


# -----------------------------------
def test_results_class(df_full, label_col_name):
    df = df_full
    X = df.drop(columns=[label_col_name])
    y = df[label_col_name]

    pipe = Pipeline([
            ("encoder", dp.EncoderOHE(verbose=False)),
            ("estimator", _get_model_class()),
        ])
    pipe.fit(X, y)
    pred = pipe.predict_proba(X)
    _ = fetch_cohort_results(X, y, pred, cohort_col=["CN_0_num_0"])

    c1 = [ ['CN_0_num_0', '==', 'val0_1'], 'and', ['num_0', '>', 0.0] ]
    c2 = [ ['CN_0_num_0', '==', 'val0_0'], 'and', ['num_0', '>', 0.0] ]
    c3 = None
    _, th_dict = fetch_cohort_results(X, y, pred, cohort_def=[c1, c2, c3], return_th_dict=True)
    _ = fetch_cohort_results(X, y, pred, cohort_def=[c1, c2, c3], fixed_th=th_dict)
    _ = fetch_cohort_results(X, y, pred, cohort_def=[c1, c2, c3], fixed_th=0.5)
    _ = fetch_cohort_results(X, y, pred, cohort_def=[c1, c2, c3], fixed_th=0.5, shared_th=True)

    with pytest.raises(Exception):
        _ = fetch_cohort_results(X, y, pred)
    with pytest.raises(Exception):
        _ = fetch_cohort_results(X, y, pred, cohort_def=[c1, c2, c3], fixed_th="a")
    with pytest.raises(Exception):
        _ = fetch_cohort_results(X, y, pred, cohort_def=[c1, c2, c3], fixed_th={"all":0.5})


# -----------------------------------
def test_results_reg(df_regression, label_col_name):
    df = df_regression
    X = df.drop(columns=[label_col_name])
    y = df[label_col_name]

    pipe = Pipeline([
            ("encoder", dp.EncoderOHE(verbose=False)),
            ("estimator", _get_model_reg()),
        ])
    pipe.fit(X, y)
    pred = pipe.predict(X)
    _ = fetch_cohort_results(X, y, pred, cohort_col=["CN_0_num_0"], regression=True)
