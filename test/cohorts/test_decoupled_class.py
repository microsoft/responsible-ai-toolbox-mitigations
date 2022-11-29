import pytest
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import conftest as utils
from sklearn.svm import LinearSVC

import raimitigations.dataprocessing as dp
from raimitigations.cohort import DecoupledClass
from raimitigations.cohort.decoupled_class.decoupled_cohort import _DecoupledCohort



# -----------------------------------
def _get_classifier():
    model = DecisionTreeClassifier(max_features="sqrt")
    return model


# -----------------------------------
def _get_regressor():
    model = DecisionTreeRegressor()
    return model


# -----------------------------------
def _get_object_list_bin_class(df=None, label_col=None, X=None, y=None, use_index=True):
    utils.check_valid_input(df, label_col, X, y)

    dec_class_list = []
    model = _get_classifier()
    model_no_proba = LinearSVC()
    preprocessing = [dp.EncoderOrdinal(verbose=False)]

    if use_index:
        cohort_cols1 = [1, 2]
        cohort_cols2 = [8]
        cohorts = {
            "cohort_1": [ ['1', '==', ['20-29', '30-39']] ],
            "cohort_2": [ ['1', '==', ['40-49', '50-59']] ],
            "cohort_3": None
        }
    else:
        cohort_cols1 = ["age", "menopause"]
        cohort_cols2 = ["breast-quad"]
        cohorts = {
            "cohort_1": [ ['age', '==', ['20-29', '30-39']] ],
            "cohort_2": [ ['age', '==', ['40-49', '50-59']] ],
            "cohort_3": None
        }

    dec_class = DecoupledClass(
        df=df,
        label_col=label_col,
        X=X,
        y=y,
        cohort_col=cohort_cols1,
        min_cohort_pct=0.2,
        minority_min_rate=0.15,
        estimator=model_no_proba,
        transform_pipe=preprocessing,
    )
    dec_class_list.append(dec_class)

    dec_class = DecoupledClass(
        df=df,
        label_col=label_col,
        X=X,
        y=y,
        cohort_col=cohort_cols2,
        theta=[0.3, 0.6, 0.9],
        min_fold_size_theta=10,
        default_theta=0.5,
        min_cohort_pct=0.2,
        minority_min_rate=0.15,
        estimator=model,
        transform_pipe=preprocessing,
    )
    dec_class_list.append(dec_class)

    dec_class = DecoupledClass(
        df=df,
        label_col=label_col,
        X=X,
        y=y,
        theta=False,
        cohort_def=cohorts,
        estimator=model,
        min_cohort_size=10,
        transform_pipe=preprocessing,
    )
    dec_class_list.append(dec_class)

    return dec_class_list


# -----------------------------------
def _run_main_commands(df, label_col, dec_class, df_in_fit=True, predict_prob=False):
    df = df.copy()
    if df_in_fit:
        dec_class.fit(df=df, label_col=label_col)
    else:
        dec_class.fit()

    dec_class.print_cohorts()

    if type(label_col) == int:
        test_dummy = df.copy().drop(df.columns[label_col], axis=1)
    else:
        test_dummy = df.copy().drop(columns=[label_col])
    if predict_prob and hasattr(dec_class.estimator, "predict_proba"):
        predict = dec_class.predict_proba(test_dummy)
    else:
        predict = dec_class.predict(test_dummy)
        test_dummy["pred"] = predict


# -----------------------------------
def test_df_const(df_breast_cancer, label_name_bc):
    df = df_breast_cancer

    obj_list = _get_object_list_bin_class(df, label_name_bc, use_index=False)
    for obj in obj_list:
        _run_main_commands(df, label_name_bc, obj, df_in_fit=False, predict_prob=True)


# -----------------------------------
def test_xy_const(df_breast_cancer, label_name_bc):
    df = df_breast_cancer

    X = df.drop(columns=[label_name_bc])
    y = df[label_name_bc]
    obj_list = _get_object_list_bin_class(X=X, y=y, use_index=False)
    for obj in obj_list:
        _run_main_commands(df, label_name_bc, obj, df_in_fit=False)


# -----------------------------------
def test_col_name(df_breast_cancer, label_name_bc):
    df = df_breast_cancer

    obj_list = _get_object_list_bin_class(use_index=False)
    for obj in obj_list:
        _run_main_commands(df, label_name_bc, obj, df_in_fit=True)


# -----------------------------------
def test_no_col_name(df_breast_cancer, label_index_bc):
    df = df_breast_cancer

    df.columns = [i for i in range(df.shape[1])]
    obj_list = _get_object_list_bin_class(use_index=True)
    for obj in obj_list:
        _run_main_commands(df, label_index_bc, obj, df_in_fit=True)


# -----------------------------------
def _get_object_list_multiclass():
    dec_class_list = []
    model = _get_classifier()
    preprocessing = [dp.EncoderOHE(verbose=False)]

    dec_class = DecoupledClass(
        cohort_col=["CN_1_num_1"],
        theta=True,
        minority_min_rate=0.02,
        estimator=model,
        transform_pipe=preprocessing,
    )
    dec_class_list.append(dec_class)

    return dec_class_list


# -----------------------------------
def test_multiclass(df_multiclass1, label_col_name):
    df = df_multiclass1

    obj_list = _get_object_list_multiclass()
    for obj in obj_list:
        _run_main_commands(df, label_col_name, obj, df_in_fit=True)


# -----------------------------------
def _get_object_list_regression():
    dec_class_list = []
    model = _get_regressor()
    preprocessing = [dp.EncoderOHE(verbose=False)]

    dec_class = DecoupledClass(
        cohort_col=["CN_0_num_0"],
        theta=False,
        min_cohort_size=20,
        min_cohort_pct=0.01,
        minority_min_rate=0.05,
        estimator=model,
        transform_pipe=preprocessing,
    )
    dec_class_list.append(dec_class)

    dec_class = DecoupledClass(
        cohort_col=["CN_1_num_1"],
        theta=True,
        min_cohort_size=80,
        min_fold_size_theta=10,
        transform_pipe=preprocessing,
    )
    dec_class_list.append(dec_class)

    return dec_class_list


# -----------------------------------
def test_regression(df_regression, label_col_name):
    df = df_regression

    obj_list = _get_object_list_regression()
    for obj in obj_list:
        _run_main_commands(df, label_col_name, obj, df_in_fit=True)


# -----------------------------------
def test_instantiation_errors_bin_class():
    # ERROR: at least one of the following parameters must be provided: cohorts or cohort_cols
    with pytest.raises(Exception):
        _ = DecoupledClass()

    # ERROR: cohort_cols must be a list of column names...
    with pytest.raises(Exception):
        _ = DecoupledClass(cohort_col=4)

    # ERROR: cohort_dict must be a dictionary that defines...
    with pytest.raises(Exception):
        _ = DecoupledClass(cohort_def=4)

    # ERROR: the 'min_cohort_size' parameter must be an integer
    with pytest.raises(Exception):
        _ = DecoupledClass(cohort_col=["age"], min_cohort_size=0.2)

    # ERROR: Invalid value for the 'theta' parameter...
    with pytest.raises(Exception):
        _ = DecoupledClass(cohort_col=["age"], theta="hello")

    # ERROR: Invalid parameter 'min_fold_size_theta'...
    with pytest.raises(Exception):
        _ = DecoupledClass(cohort_col=["age"], min_fold_size_theta=0.1)

    # "ERROR: Invalid parameter 'valid_k_folds_theta'...
    with pytest.raises(Exception):
        _ = DecoupledClass(cohort_col=["age"], valid_k_folds_theta=0.1)
    with pytest.raises(Exception):
        _ = DecoupledClass(cohort_col=["age"], valid_k_folds_theta=[-1])

    # ERROR: Expected 'estimator' to be a SKLearn classifier
    with pytest.raises(Exception):
        _ = DecoupledClass(estimator="estimator")

    # ERROR: the 'label_col' parameter must be a pd.Series containing the...
    with pytest.raises(Exception):
        _ = _DecoupledCohort(label_col=[1, 2, 3])

    # ERROR: invalid estimator provided. When using transfer learning...
    with pytest.raises(Exception):
        dec_class = DecoupledClass(
            cohort_col=["menopause"],
            theta=0.6,
            estimator=QuadraticDiscriminantAnalysis(),
        )


# -----------------------------------
def _get_list_obj_error_fit_bin_class():
    obj_list = []
    model = _get_classifier()
    preprocessing = [dp.EncoderOrdinal(verbose=False)]

    # ERROR: Cannot use transfer learning over cohorts with skewed distributions
    dec_class = DecoupledClass(
        cohort_col=["breast-quad", "age", "menopause"],
        theta=True,
        min_cohort_pct=0.2,
        minority_min_rate=0.15,
        transform_pipe=preprocessing,
    )
    dec_class.print_cohorts()
    obj_list.append(dec_class)

    # ERROR: the cohorts created doesn't include all instances...
    cohorts = {
        "cohort_1": [["age", "==", "40-49"], "and", ["menopause", "==", "premeno"]],
        "cohort_2": [
            [["age", "==", "60-69"], "and", ["menopause", "==", "ge40"]],
            "or",
            [["age", "==", "30-39"], "and", ["menopause", "==", "premeno"]]
        ],
    }
    dec_class = DecoupledClass(
        cohort_def=cohorts,
        min_cohort_pct=0.2,
        minority_min_rate=0.15,
        estimator=model,
        transform_pipe=preprocessing,
    )
    obj_list.append(dec_class)

    # ERROR: Could not create more than 1 valid cohort...
    cohorts = {
        "cohort_1": [["age", "==", "40"], "and", ["menopause", "==", "p"]],
        "cohort_2": [["age", "==", "60"], "and", ["menopause", "==", "g"]]
    }
    dec_class = DecoupledClass(cohort_def=cohorts, transform_pipe=preprocessing)
    obj_list.append(dec_class)

    # ERROR: Could not create more than 1 cohort with the following restrinctions:...
    dec_class = DecoupledClass(
        cohort_col=["age", "menopause", "tumor-size"],
        theta=False,
        min_cohort_pct=0.4,
        minority_min_rate=0.5,
        transform_pipe=preprocessing
    )
    obj_list.append(dec_class)

    # ERROR: could not find any outside cohort with a compatible label distribution...
    dec_class = DecoupledClass(
        cohort_def=cohorts,
        theta=0.1,
        cohort_dist_th=0.1,
        transform_pipe=preprocessing
    )
    obj_list.append(dec_class)

    # ERROR: one of the cohorts used is too small, or with a skewed label...
    cohorts = {
        "cohort_1": [["age", "==", ["20-29", "70-79"]]],
        "cohort_2": None
    }
    dec_class = DecoupledClass(
        cohort_def=cohorts,
        theta=[0.1, 0.5, 0.8],
        min_fold_size_theta=20,
        valid_k_folds_theta=[3, 4, 5],
        transform_pipe=preprocessing
    )
    obj_list.append(dec_class)

    return obj_list


# -----------------------------------
def test_fit_errors_bin_class(df_breast_cancer, label_name_bc):
    df = df_breast_cancer
    obj_list = _get_list_obj_error_fit_bin_class()

    for obj in obj_list:
        with pytest.raises(Exception):
            obj.fit(df=df, label_col=label_name_bc)


# -----------------------------------
def _get_list_obj_error_fit_regression():
    obj_list = []
    model = _get_regressor()
    preprocessing = [dp.EncoderOHE(verbose=False)]

    # ERROR: the 'regression' parameter must a boolean value.
    dec_class = DecoupledClass(
        cohort_col=["CN_1_num_1"],
        regression=[10],
        min_cohort_size=10,
        min_cohort_pct=0.01,
        minority_min_rate=0.05,
        estimator=model,
        transform_pipe=preprocessing,
    )
    obj_list.append(dec_class)

    # ERROR: Expected a regression model (regression = True), but instead...
    dec_class = DecoupledClass(
        cohort_col=["CN_1_num_1"],
        regression=True,
        estimator=_get_classifier(),
        transform_pipe=preprocessing,
    )
    obj_list.append(dec_class)

    # ERROR: Expected a classification model (regression = False), but instead...
    dec_class = DecoupledClass(
        cohort_col=["CN_1_num_1"],
        regression=False,
        min_cohort_size=10,
        min_cohort_pct=0.01,
        minority_min_rate=0.05,
        estimator=DecisionTreeRegressor(),
        transform_pipe=preprocessing,
    )
    obj_list.append(dec_class)

    return obj_list


# -----------------------------------
def test_fit_errors_regression(df_regression, label_col_name):
    df = df_regression
    obj_list = _get_list_obj_error_fit_regression()

    for obj in obj_list:
        with pytest.raises(Exception):
            obj.fit(df=df, label_col=label_col_name)


# -----------------------------------
def test_predict_errors_bin_class(df_breast_cancer, label_name_bc):
    df = df_breast_cancer
    df_train = df.query("`deg-malig`!=1")
    df_pred = df.query("`deg-malig`==1")

    preprocessing = [dp.EncoderOrdinal(verbose=False)]
    dec_class = DecoupledClass(
        cohort_col=["deg-malig"],
        theta=False,
        transform_pipe=preprocessing
    )
    dec_class.fit(df=df_train, label_col=label_name_bc)

    # ERROR: a subset of the instances passed to the transform(), predict()...
    with pytest.raises(Exception):
        _ = dec_class.predict_proba(df_pred)

    dec_class = DecoupledClass(
        cohort_col=["breast"],
        transform_pipe=preprocessing,
        estimator=LinearSVC()
    )
    dec_class.fit(df=df, label_col=label_name_bc)
    X = df.drop(columns=[label_name_bc])
    _ = dec_class.predict(X)
    with pytest.raises(Exception):
        _ = dec_class.predict_proba(X)


# -----------------------------------
def test_predict_errors_regression(df_regression, label_col_name):
    df = df_regression
    dec_class = DecoupledClass(
        cohort_col=["CN_0_num_0"],
        theta=False,
        min_cohort_size=10,
        min_cohort_pct=0.01,
        minority_min_rate=0.05,
        transform_pipe=[dp.EncoderOHE(verbose=False)],
    )
    dec_class.fit(df=df, label_col=label_col_name)
    with pytest.raises(Exception):
        _ = dec_class.predict_proba(df)


# -----------------------------------
def test_other_errors(df_breast_cancer, label_name_bc):
    df = df_breast_cancer

    # ERROR: only the last cohort is allowed to have a condition list assigned to 'None'
    dec = DecoupledClass(cohort_def=[[["age", "==", "40-49"]], None, None])
    with pytest.raises(Exception):
        dec.fit(df=df, label_col=label_name_bc)

