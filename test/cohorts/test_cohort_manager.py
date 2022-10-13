import os
import pytest
import conftest as utils
import xgboost as xgb
from sklearn.pipeline import Pipeline

import raimitigations.dataprocessing as dp
from raimitigations.cohort.cohort_manager import CohortManager

# -----------------------------------
def _get_model():
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
def _get_object_list(df=None, label_col=None, X=None, y=None, use_index=True):
    utils.check_valid_input(df, label_col, X, y)

    cht_list = []

    if use_index:
        c1 = [ ['2', '==', 'val0_1'], 'and', ['0', '>', 0.0] ]
        c2 = [ ['2', '==', 'val0_0'], 'and', ['0', '>', 0.0] ]
        c3 = None
        cht_col = ['2', '3']
    else:
        c1 = [ ['CN_0_num_0', '==', 'val0_1'], 'and', ['num_0', '>', 0.0] ]
        c2 = [ ['CN_0_num_0', '==', 'val0_0'], 'and', ['num_0', '>', 0.0] ]
        c3 = None
        cht_col = ['CN_0_num_0', 'CN_1_num_1']

    cohort_pipeline = [
        dp.BasicImputer(verbose=False),
        dp.DataMinMaxScaler(verbose=False),
    ]
    cohort_set = CohortManager(
        df=df, label_col=label_col, X=X, y=y,
        transform_pipe=cohort_pipeline,
        cohort_def=[c1, c2, c3]
    )
    cht_list.append(cohort_set)

    cohort_pipeline = [
        dp.DataMinMaxScaler(verbose=False),
        dp.EncoderOrdinal(verbose=False),
        _get_model()
    ]
    cohort_set = CohortManager(
        df=df, label_col=label_col, X=X, y=y,
        transform_pipe=cohort_pipeline,
        cohort_def=[c1, c2, c3]
    )
    cht_list.append(cohort_set)

    c1_pipe = [dp.DataMinMaxScaler(verbose=False)]
    c2_pipe = dp.DataQuantileTransformer(verbose=False)
    c3_pipe = None
    cohort_set = CohortManager(
        df=df, label_col=label_col, X=X, y=y,
        transform_pipe=[c1_pipe, c2_pipe, c3_pipe],
        cohort_def=[c1, c2, c3]
    )
    cht_list.append(cohort_set)

    cohort_set = CohortManager(
        df=df, label_col=label_col, X=X, y=y,
        transform_pipe=dp.DataMinMaxScaler(verbose=False),
        cohort_def=[c1, c2, c3]
    )
    cht_list.append(cohort_set)

    cohort_set = CohortManager(
        df=df, label_col=label_col, X=X, y=y,
        cohort_col=cht_col
    )
    cht_list.append(cohort_set)


    return cht_list


# -----------------------------------
def _run_main_commands(X, y, cht_manager, X_in_fit=True):
    X = X.copy()
    if X_in_fit:
        cht_manager.fit(X=X, y=y)
    else:
        cht_manager.fit()

    _ = cht_manager.transform(X)
    if cht_manager._pipe_has_predict:
        _ = cht_manager.predict(X)
    if cht_manager._pipe_has_predict_proba:
        _ = cht_manager.predict_proba(X, split_pred=True)
    _ = cht_manager.get_subsets(X)
    _ = cht_manager.get_subsets(X, y, apply_transform=True)
    _ = cht_manager.get_queries()

    cht_manager.save_conditions("cht.json")
    _ = CohortManager(
        cohort_def="cht.json"
    )
    os.remove("cht.json")

# -----------------------------------
def test_df(df_full_cohort, label_col_name):
    df = df_full_cohort
    X = df.drop(columns=[label_col_name])
    y = df[label_col_name]

    obj_list = _get_object_list(df, label_col_name, use_index=False)
    for obj in obj_list:
        _run_main_commands(X, y, obj, X_in_fit=False)


# -----------------------------------
def test_xy(df_full_cohort, label_col_name):
    df = df_full_cohort
    X = df.drop(columns=[label_col_name])
    y = df[label_col_name]

    obj_list = _get_object_list(X=X, y=y, use_index=False)
    for obj in obj_list:
        _run_main_commands(X, y, obj, X_in_fit=False)


# -----------------------------------
def test_col_name(df_full_cohort, label_col_name):
    df = df_full_cohort
    X = df.drop(columns=[label_col_name])
    y = df[label_col_name]

    obj_list = _get_object_list(use_index=False)
    for obj in obj_list:
        _run_main_commands(X, y, obj, X_in_fit=True)


# -----------------------------------
def test_no_col_name(df_full_cohort, label_col_index_cohort):
    df = df_full_cohort
    label_col_name = df.columns[label_col_index_cohort]
    X = df.drop(columns=[label_col_name])
    y = df[label_col_name]

    X.columns = [i for i in range(X.shape[1])]
    y.columns = [0]
    obj_list = _get_object_list(use_index=True)
    for obj in obj_list:
        _run_main_commands(X, y, obj, X_in_fit=True)

# -----------------------------------
def test_rebalance(df_full_cohort, label_col_name):
    df = df_full_cohort
    X = df.drop(columns=[label_col_name])
    y = df[label_col_name]

    rebalance_cohort = CohortManager(
        transform_pipe=dp.Rebalance(verbose=False),
        cohort_col=["CN_0_num_0"]
    )
    new_X, new_y = rebalance_cohort.fit_resample(X, y)
    new_df = rebalance_cohort.fit_resample(df=df, rebalance_col=label_col_name)
    _ = rebalance_cohort.get_subsets(new_X, new_y, apply_transform=False)

# -----------------------------------
def test_sklearn_pipe(df_full_cohort, label_col_name):
    df = df_full_cohort
    X = df.drop(columns=[label_col_name])
    y = df[label_col_name]

    cohort_pipeline = [
        dp.DataMinMaxScaler(verbose=False),
        _get_model()
    ]
    cohort_set = CohortManager(
        transform_pipe=cohort_pipeline,
        cohort_col=["CN_0_num_0"]
    )
    skpipe = Pipeline([
        ("encoder", dp.EncoderOrdinal(verbose=False)),
        ("model", cohort_set)
    ])
    skpipe.fit(X, y)
    _ = skpipe.predict_proba(X)

# -----------------------------------
def test_errors_cohorts(df_full_cohort, label_col_name):
    cht_pipes_err = [
        [   dp.BasicImputer(verbose=False),
            dp.DataMinMaxScaler(verbose=False),
            dp.Rebalance(verbose=False) ],
        [   dp.BasicImputer(verbose=False),
            dp.DataMinMaxScaler(verbose=False),
            dp.Synthesizer(verbose=False) ],
        [   dp.BasicImputer(verbose=False),
            _get_model(),
            dp.DataMinMaxScaler(verbose=False), ],
        [ [dp.BasicImputer(verbose=False)] ]
    ]
    for cht_pipe in cht_pipes_err:
        cohort_set = CohortManager(
                transform_pipe=cht_pipe,
                cohort_col=["CN_0_num_0"]
            )
        with pytest.raises(Exception):
            cohort_set.fit(df=df_full_cohort, label_col=label_col_name)

    cohort_set = CohortManager(cohort_col=["CN_0_num_0"])
    with pytest.raises(Exception):
        cohort_set.get_queries()
    with pytest.raises(Exception):
        cohort_set.save_conditions("cht.json")
    with pytest.raises(Exception):
        cohort_set.fit_resample(df=df_full_cohort, rebalance_col=label_col_name)
    with pytest.raises(Exception):
        cohort_set.predict(df_full_cohort)
    with pytest.raises(Exception):
        cohort_set.predict_proba(df_full_cohort)

    with pytest.raises(Exception):
        cohort_set = CohortManager(
                    cohort_def=[['num_0', '>', 0.0], None],
                    cohort_col=["CN_0_num_0"]
                )
    with pytest.raises(Exception):
        cohort_set = CohortManager()
    with pytest.raises(Exception):
        cohort_set = CohortManager(cohort_def=10)
    with pytest.raises(Exception):
        cohort_set = CohortManager(cohort_col=[])
    with pytest.raises(Exception):
        cohort_set = CohortManager(cohort_def=[[['num_0', '>', 0.0]], None, None])

# -----------------------------------
def test_errors_cohorts_special_cases(df_full_cohort, label_col_name):

    class DummyClass1():
        def __init__(self):
            pass
        def fit(self):
            pass
        def transform(self):
            pass
        def predict(self):
            pass

    class DummyClass2():
        def __init__(self):
            pass
        def transform(self):
            pass
        def predict(self):
            pass

    c1 = [ ['CN_0_num_0', '==', 'val0_1'] ]
    c2 = None

    pipe = [DummyClass1(), dp.DataMinMaxScaler(verbose=False)]
    with pytest.raises(Exception):
        _ = CohortManager(transform_pipe=pipe, cohort_def=[c1, c2])

    pipe = [DummyClass2(), dp.DataMinMaxScaler(verbose=False)]
    with pytest.raises(Exception):
        _ = CohortManager(transform_pipe=pipe, cohort_def=[c1, c2])

    c1 = [ ['CN_0_num_0', '==', 'val0_1'], 'and', ['num_0', '>', 0.0] ]
    c2 = [ ['CN_0_num_0', '==', 'val0_0'], 'and', ['num_0', '>', 0.0] ]
    cht_set = CohortManager(
        transform_pipe=dp.DataMinMaxScaler(verbose=False),
        cohort_def=[c1, c2]
    )
    with pytest.raises(Exception):
        cht_set.fit(df=df_full_cohort, label_col=label_col_name)

    c1 = [ ['num_0', '>', 0.0] ]
    c2 = [ ['num_0', '>', 0.5] ]
    c3 = None
    cht_set = CohortManager(
        transform_pipe=dp.DataMinMaxScaler(verbose=False),
        cohort_def=[c1, c2, c3]
    )
    with pytest.raises(Exception):
        cht_set.fit(df=df_full_cohort, label_col=label_col_name)

    c1 = [ ['num_0', '>', 90.0] ]
    c2 = None
    cht_set = CohortManager(
        transform_pipe=dp.DataMinMaxScaler(verbose=False),
        cohort_def=[c1, c2]
    )
    with pytest.raises(Exception):
        cht_set.fit(df=df_full_cohort, label_col=label_col_name)