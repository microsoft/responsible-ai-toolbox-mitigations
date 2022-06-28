import pytest
import pandas as pd
from raimitigations.dataprocessing import EncoderOHE, EncoderOrdinal


# -----------------------------------
def _get_object_list(unique_values, df=None, use_index=True):

    encoder_list = []

    if use_index:
        enc_col = [10]
        categories = {10: unique_values}
    else:
        enc_col = ["CN_1_num_1"]
        categories = {"CN_1_num_1": unique_values}

    encoder = EncoderOHE(df=df)
    encoder_list.append(encoder)

    encoder = EncoderOrdinal(df=df, unknown_value=-10)
    encoder_list.append(encoder)

    encoder = EncoderOrdinal(df=df, unknown_err=True)
    encoder_list.append(encoder)

    encoder = EncoderOrdinal(df=df, col_encode=enc_col, categories=categories)
    encoder_list.append(encoder)

    encoder = EncoderOrdinal(df=df, unknown_err=True)
    encoder_list.append(encoder)

    encoder = EncoderOHE(df=df, col_encode=enc_col)
    encoder_list.append(encoder)

    encoder = EncoderOHE(df=df, col_encode=enc_col, drop=False, unknown_err=True)
    encoder_list.append(encoder)

    return encoder_list


# -----------------------------------
def _run_main_commands(df, transf, df_in_fit=True):
    df = df.copy()
    if df_in_fit:
        transf.fit(df=df)
    else:
        transf.fit()
    new_df = transf.transform(df)
    org_df = transf.inverse_transform(new_df)
    if type(transf) == EncoderOrdinal:
        _ = transf.get_mapping()

    def lists_similar(l1, l2):
        if len(l1) != len(l2):
            return False
        l1.sort()
        l2.sort()
        if l1 == l2:
            return True
        else:
            return False

    if "CN_1_num_1" in df.columns.to_list():
        similar = lists_similar(df["CN_1_num_1"].tolist(), org_df["CN_1_num_1"].tolist())
    else:
        similar = lists_similar(df["10"].tolist(), org_df["10"].tolist())
    assert similar, "ERROR: the inverse_transform method didn't revert the column to its original value."


# -----------------------------------
def test_df_const(df_full):
    df = df_full

    unique_val = df["CN_1_num_1"].unique()
    unique_val.sort()
    obj_list = _get_object_list(unique_val, df, use_index=False)
    for obj in obj_list:
        _run_main_commands(df, obj, df_in_fit=False)


# -----------------------------------
def test_col_name(df_full):
    df = df_full

    unique_val = df["CN_1_num_1"].unique()
    unique_val.sort()
    obj_list = _get_object_list(unique_val, df=None, use_index=False)
    for obj in obj_list:
        _run_main_commands(df, obj, df_in_fit=True)
        _ = obj.get_encoded_columns()


# -----------------------------------
def test_col_index(df_full):
    df = df_full

    unique_val = df["CN_1_num_1"].unique()
    unique_val.sort()
    obj_list = _get_object_list(unique_val, df=None, use_index=True)
    for obj in obj_list:
        _run_main_commands(df, obj, df_in_fit=True)


# -----------------------------------
def test_no_col_name(df_full):
    df = df_full

    unique_val = df["CN_1_num_1"].unique()
    df.columns = [i for i in range(df.shape[1])]
    unique_val.sort()
    obj_list = _get_object_list(unique_val, df=None, use_index=True)
    for obj in obj_list:
        _run_main_commands(df, obj, df_in_fit=True)


# -----------------------------------


def test_special_case(df_full):
    df = df_full
    obj = EncoderOrdinal()
    obj.fit(df=df)
    new_df = obj.transform(df)

    obj = EncoderOHE()
    obj.fit(df=new_df)
    _ = obj.transform(new_df)


# -----------------------------------


def test_errors(df_full):
    df = df_full
    obj_list = [EncoderOrdinal(categories={18: ["1"]}), EncoderOrdinal(col_encode=18), EncoderOrdinal(col_encode=[6.0])]
    for obj in obj_list:
        df_cp = df.copy()
        with pytest.raises(Exception):
            obj.fit(df=df_cp)


# -----------------------------------


def test_errors_no_col_name(df_full):
    df = df_full
    df.columns = [i for i in range(df.shape[1])]
    obj_list = [
        EncoderOrdinal(categories=["value"]),
        EncoderOrdinal(categories={18: ["1"]}),
        EncoderOrdinal(categories={"CN_1_num_1": 1}),
        EncoderOrdinal(categories={10: ["1"]}),
    ]
    for obj in obj_list:
        df_cp = df.copy()
        with pytest.raises(Exception):
            obj.fit(df=df_cp)

    with pytest.raises(Exception):
        obj_list[0].transform(df=df_cp)

    obj = EncoderOrdinal()
    with pytest.raises(Exception):
        obj.fit(df=["a", "a", "b", "a", "b", "b"])
    with pytest.raises(Exception):
        obj.fit()


# -----------------------------------


def test_ordinal_unknown():
    df = pd.DataFrame()
    df["col1"] = [1, 2, 3, 4, 5, 6, 7]
    df["col2"] = ["UNKNOWN", "_unknown_", "UNK", "?", "UNKNOWN_0.0", "UNKNOWN_1.0", "UNKNOWN"]
    encoder = EncoderOrdinal()
    encoder.fit(df=df)
