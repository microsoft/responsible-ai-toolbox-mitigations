import os
import pytest
import pickle
import warnings

import conftest as utils
from raimitigations.dataprocessing import Synthesizer, EncoderOrdinal, BasicImputer

warnings.filterwarnings('ignore')


# -----------------------------------
def _get_object_list(df=None, label_col=None, X=None, y=None):
    utils.check_valid_input(df, label_col, X, y)

    synth_list = []

    imputer = BasicImputer()
    encoder = EncoderOrdinal()
    synth = Synthesizer(
        df=df,
        label_col=label_col,
        X=X,
        y=y,
        transform_pipe=[imputer, encoder],
        model="ctgan",
        epochs=2,
        load_existing=False,
    )
    synth_list.append(synth)

    synth = Synthesizer(df=df, label_col=label_col, X=X, y=y, model="copula", epochs=1, load_existing=True)
    synth_list.append(synth)

    synth = Synthesizer(df=df, label_col=label_col, X=X, y=y, model="copula_gan", epochs=1, load_existing=True)
    synth_list.append(synth)

    synth = Synthesizer(df=df, label_col=label_col, X=X, y=y, model="tvae", epochs=1, load_existing=True)
    synth_list.append(synth)

    return synth_list


# -----------------------------------
def _run_main_commands(df, label_col, transf, df_in_fit=True):
    df = df.copy()
    if df_in_fit:
        transf.fit(df=df, label_col=label_col)
    else:
        transf.fit()

    conditions = {"CN_1_num_1": "val1_0", label_col: 1}

    try:
        _ = transf.sample(10)
    except Exception as error:
        error_msg = str(error)
        if "Unable to sample any rows for the given conditions" not in error_msg:
            raise ValueError(f"ERROR: the following error occured while generating synthetic data: {error_msg}")

    try:
        _ = transf.transform(df=df)
    except Exception as error:
        error_msg = str(error)
        if "valid rows" not in error_msg:
            raise ValueError(f"ERROR: the following error occured while generating synthetic data: {error_msg}")

    try:
        _ = transf.transform(df=df, n_samples=10)
    except Exception as error:
        error_msg = str(error)
        if "Unable to sample any rows for the given conditions" not in error_msg:
            raise ValueError(f"ERROR: the following error occured while generating synthetic data: {error_msg}")

    X = df.drop(columns=[label_col])
    y = df[label_col]
    try:
        _ = transf.transform(X=X, y=y, n_samples=5, conditions=conditions)
    except Exception as error:
        error_msg = str(error)
        if "Unable to sample any rows for the given conditions" not in error_msg:
            raise ValueError(f"ERROR: the following error occured while generating synthetic data: {error_msg}")


# -----------------------------------
def test_df_const(df_full_nan, label_col_name):
    df = df_full_nan
    obj_list = _get_object_list(df, label_col_name)
    for obj in obj_list:
        _run_main_commands(df, label_col_name, obj, df_in_fit=False)


# -----------------------------------
def test_xy_const(df_full_nan, label_col_name):
    df = df_full_nan
    X = df.drop(columns=[label_col_name])
    y = df[label_col_name]
    obj_list = _get_object_list(X=X, y=y)
    for obj in obj_list:
        _run_main_commands(df, label_col_name, obj, df_in_fit=False)


# -----------------------------------
def test_col_name(df_full_nan, label_col_name):
    df = df_full_nan
    obj_list = _get_object_list()
    for obj in obj_list:
        _run_main_commands(df, label_col_name, obj, df_in_fit=True)
        os.remove(obj.save_file)


# -----------------------------------
def test_pickle(df_full_nan, label_col_name):
    df = df_full_nan
    synth = Synthesizer(model="ctgan", epochs=1)
    synth.fit(df=df, label_col=label_col_name)
    _ = synth.sample(20)

    file_writer = open("synth.obj", "wb")
    pickle.dump(synth, file_writer)
    file_writer.close()
    os.remove(synth.save_file)

    file_reader = open("synth.obj", "rb")
    synth_loaded = pickle.load(file_reader)
    file_reader.close()

    _ = synth_loaded.sample(20)
    os.remove("synth.obj")


# -----------------------------------
def test_errors(df_full_nan, label_col_name):
    df = df_full_nan
    with pytest.raises(Exception):
        obj = Synthesizer(model="a")
    with pytest.raises(Exception):
        obj = Synthesizer(df=df, label_col=6)
    with pytest.raises(Exception):
        obj = Synthesizer(save_file=10)

    df_err1 = df.copy()
    df_err1.rename(columns={"num_0":"new"}, inplace=True)
    df_err2 = df.copy()
    df_err2["new"] = df_err2["num_0"].values.tolist()
    conditions = {"CN_1_num_1": "val1_0", label_col_name: 1}

    obj = Synthesizer(epochs=1)
    obj.fit(df=df, label_col=label_col_name)
    with pytest.raises(Exception):
        obj.transform(strategy="a")
    with pytest.raises(Exception):
        obj.transform(strategy=10)
    with pytest.raises(Exception):
        obj.transform(df=df_err1)
    with pytest.raises(Exception):
        obj.transform(df=df_err2)
    with pytest.raises(Exception):
        obj.transform(conditions=conditions)
    os.remove(obj.save_file)
