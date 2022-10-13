from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pytest

from raimitigations.utils import split_data
from raimitigations.dataprocessing import (
    EncoderOHE,
    EncoderOrdinal,
    BasicImputer,
    DataMinMaxScaler,
    CatBoostSelection,
)


# -----------------------------------
def _get_pipelines():
    pipeline_list = []

    pipe = Pipeline(
        [
            ("encoder", EncoderOHE(col_encode=["CN_0_num_0", "CN_1_num_1"])),  # dataprocessing
            ("imputer", SimpleImputer(strategy="constant", fill_value=-100)),  # sklearn
            ("scaler", DataMinMaxScaler(include_cols=[0, 1])),  # dataprocessing
            ("std", StandardScaler()),  # sklearn
        ]
    )
    pipeline_list.append({"name": "p1", "pipe": pipe, "numpy": False, "error": False})

    pipe = Pipeline(
        [
            ("encoder", EncoderOHE(col_encode=["CN_0_num_0", "CN_1_num_1"])),  # dataprocessing
            ("imputer", BasicImputer()),  # sklearn
            ("scaler", DataMinMaxScaler(include_cols=["num_0", "num_1"])),  # dataprocessing
            ("std", StandardScaler()),  # sklearn
        ]
    )
    pipeline_list.append({"name": "p2", "pipe": pipe, "numpy": False, "error": False})

    pipe = Pipeline(
        [
            ("encoder", EncoderOHE(col_encode=[8, 9])),  # dataprocessing
            ("imputer", BasicImputer()),  # sklearn
            ("scaler", DataMinMaxScaler(include_cols=["num_0", "num_1"])),  # dataprocessing
            ("std", StandardScaler()),  # sklearn
        ]
    )
    pipeline_list.append({"name": "p3", "pipe": pipe, "numpy": False, "error": False})

    pipe = Pipeline(
        [
            ("imputer", BasicImputer()),
            ("scaler", DataMinMaxScaler(include_cols=[0, 1])),
            ("encoder", EncoderOHE()),
            ("std", StandardScaler()),
        ]
    )
    pipeline_list.append({"name": "p4", "pipe": pipe, "numpy": True, "error": False})

    pipe = Pipeline(
        [
            ("encoder", EncoderOHE(col_encode=["CN_0_num_0", "CN_1_num_1"])),  # dataprocessing
            ("imputer", SimpleImputer(strategy="constant", fill_value=-100)),  # sklearn
            ("scaler", DataMinMaxScaler(include_cols=["num_0", "num_1"])),  # dataprocessing
            ("std", StandardScaler()),  # sklearn
        ]
    )
    pipeline_list.append({"name": "p5", "pipe": pipe, "numpy": False, "error": True})

    pipe = Pipeline(
        [
            ("imputer", BasicImputer()),
            ("encoder", EncoderOrdinal()),
            ("std", StandardScaler()),
            ("scaler", DataMinMaxScaler(include_cols=["num_0", "num_1"])),
            ("model", SVC()),
        ]
    )
    pipeline_list.append({"name": "p6", "pipe": pipe, "numpy": False, "error": False})

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=-100)),
            ("encoder", EncoderOrdinal()),
            ("std", StandardScaler()),
            ("scaler", DataMinMaxScaler(include_cols=[0, 1])),
            ("model", SVC()),
        ]
    )
    pipeline_list.append({"name": "p7", "pipe": pipe, "numpy": True, "error": False})

    pipe = Pipeline(
        [
            ("encoder", EncoderOHE(col_encode=["CN_0_num_0", "CN_1_num_1"])),  # dataprocessing
            ("imputer", SimpleImputer(strategy="constant", fill_value=-100)),  # sklearn
            ("scaler", DataMinMaxScaler(include_cols=["num_0", "num_1"])),  # dataprocessing
            ("std", StandardScaler()),  # sklearn
        ]
    )
    pipeline_list.append({"name": "p8", "pipe": pipe, "numpy": False, "error": True})

    pipe = Pipeline(
        [
            ("imputer", BasicImputer()),
            ("encoder", EncoderOrdinal()),
            ("std", StandardScaler()),
            ("scaler", DataMinMaxScaler(include_cols=["num_0", "num_1"])),
            ("model", KMeans()),
        ]
    )
    pipeline_list.append({"name": "p9", "pipe": pipe, "numpy": False, "error": True})

    pipe = Pipeline(
        [
            ("imputer", BasicImputer()),
            ("encoder", EncoderOrdinal()),
            ("scaler", DataMinMaxScaler(include_cols=["num_0", "num_1"])),
            ("model", KMeans()),
        ]
    )
    pipeline_list.append({"name": "p10", "pipe": pipe, "numpy": False, "error": False})

    pipe = Pipeline(
        [
            ("imputer", BasicImputer()),
            ("encoder", EncoderOrdinal()),
            ("feat_sel", CatBoostSelection(fixed_cols=["num_0", "num_1"], catboost_log=False)),
            ("scaler", DataMinMaxScaler(include_cols=["num_0", "num_1"])),
            ("model", KMeans()),
        ]
    )
    pipeline_list.append({"name": "p11", "pipe": pipe, "numpy": False, "error": False})

    return pipeline_list


# -----------------------------------
def _run_main_commands(pipe, X, y, fit_transform_count, predict_proba_count):
    # test pipelines with fit, transform, and fit_transform
    if hasattr(pipe, "transform"):
        if fit_transform_count % 2 == 0:
            pipe.fit(X, y)
            _ = pipe.transform(X)
        else:
            _ = pipe.fit_transform(X, y)
        fit_transform_count += 1
    # test pipelines with fit_predict
    elif hasattr(pipe, "fit_predict"):
        _ = pipe.fit_predict(X, y)
    # test pipelines with fit, predict, and predict_proba
    elif hasattr(pipe, "predict_proba"):
        pipe.fit(X, y)
        if predict_proba_count % 2 == 0:
            _ = pipe.predict(X)
        else:
            _ = pipe.predict_proba(X)
        predict_proba_count += 1


# -----------------------------------
def test_transform_pipelines(df_full, label_col_name):
    train_x, test_x, train_y, test_y = split_data(df_full, label_col_name, test_size=0.2)
    train_x_np = train_x.to_numpy()
    train_y_np = train_y.to_numpy()

    pipeline_list = _get_pipelines()

    fit_transform_count = 0
    predict_proba_count = 0
    for pipe_info in pipeline_list:
        pipe = pipe_info["pipe"]
        name = pipe_info["name"]
        print(f"Running pipeline {name}")

        X = train_x.copy()
        y = train_y.copy()
        if pipe_info["numpy"]:
            X = train_x_np
            y = train_y_np

        if pipe_info["error"]:
            with pytest.raises(Exception):
                _run_main_commands(pipe, X, y, fit_transform_count, predict_proba_count)
        else:
            _run_main_commands(pipe, X, y, fit_transform_count, predict_proba_count)
