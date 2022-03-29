import pytest
import sys
import copy
import pandas as pd
import zipfile

from raimitigations.dataprocessing import Split
from common_utils import (create_hr_promotion_data, create_hr_promotion_10_data, validate_rows, data_set_drop_null_dup)

# parameters for Split API
# data_split =  Split(dataset, target, train_size, random_state, categorical_features, drop_null, drop_duplicates, is_stratify)
# Parameters:
#           dataset
#           target
#           train_size
#           random_state = None
#           categorical_features = True
#           drop_null = True
#           drop_duplicates = False
#           is_stratify = False


hr_promotion = create_hr_promotion_data()
hr_promotion_10 = create_hr_promotion_10_data()

# unit test for the Split API

def test_data_split_rate():

    """Data Split test with 0.9 rate, no drop_null and no drop_dupicates. API call: Split(hr_data1, 12, 0.9, 42, True, False, False, False)"""

    train_size = 0.9
    data_split = Split(hr_promotion_10, 12, train_size, 42, True, False, False, False)
    train_data, test_data = data_split.split()

    assert train_data.shape[0] == validate_rows(hr_promotion_10, 0.9, False, False)
    assert test_data.shape[0] == validate_rows(hr_promotion_10, 0.1, False, False)


def test_data_split_remove_duplicate():

    """Data Split test with 0.9 rate, drop_dupicates but no drop_null. API call: Split(hr_data2, 12, 0.9, 42, True, False, True, False)"""

    data_split = Split(hr_promotion_10, 12, 0.6, 42, True, False, True, False)
    train_data, test_data = data_split.split()

    train_val = validate_rows(hr_promotion_10, 0.6, False, True)
    test_val = data_set_drop_null_dup(hr_promotion_10, False, True) - train_val

    assert train_data.shape[0] == train_val
    assert test_data.shape[0] == test_val



def test_data_split_drop_Null():

    """Data Split test with 0.9 rate, drop_null but no drop_dupicates. API call: Split(hr_data3, 12, 0.9, 42, True, True, False, False)"""

    data_split = Split(hr_promotion_10, 12, 0.8, 42, True, True, False, False)
    train_data, test_data = data_split.split()

    train_val = validate_rows(hr_promotion_10, 0.8, True, False)
    test_val = data_set_drop_null_dup(hr_promotion_10, True, False) - train_val

    assert train_data.shape[0] == train_val
    assert test_data.shape[0] == test_val


def test_data_split_stratifyOn():

    """Data Split test with 0.8 rate, stratify On. API call: Split(hr_data4, 12, 0.8, 42, True, False, False, True)"""

    data_split_strat_False = Split(hr_promotion, 12, 0.8, 42, True, False, False, False)
    train_dataOff, test_dataOff = data_split_strat_False.split()

    data_split = Split(hr_promotion, 12, 0.8, 42, True, False, False, True)
    train_dataOn, test_dataOn = data_split.split()

    df_trainOff = pd.DataFrame(train_dataOff)
    df_testOff = pd.DataFrame(test_dataOff)
    train_percent_stratOff = (
        df_trainOff.is_promoted.value_counts()[1]
        / df_trainOff.is_promoted.value_counts()[0]
    ) * 100
    test_percent_stratOff = (
        df_testOff.is_promoted.value_counts()[1]
        / df_testOff.is_promoted.value_counts()[0]
    ) * 100

    df_trainOn = pd.DataFrame(train_dataOn)
    df_testOn = pd.DataFrame(test_dataOn)
    train_percent_stratOn = (
        df_trainOn.is_promoted.value_counts()[1]
        / df_trainOn.is_promoted.value_counts()[0]
    ) * 100
    test_percent_stratOn = (
        df_testOn.is_promoted.value_counts()[1]
        / df_testOn.is_promoted.value_counts()[0]
    ) * 100

    assert abs(train_percent_stratOn - test_percent_stratOn) < abs(
        train_percent_stratOff - test_percent_stratOff
    )


def test_data_split_dropNulls_stratifyOff():

    """Data Split test with 0.8 rate, drop null, stratify Off. API call: Split(hr_data5, 12, 0.8, 42, True, True, False, False)"""

    data_split_strat_False = Split(hr_promotion, 12, 0.8, 42, True, True, False, False)
    train_dataOff, test_dataOff = data_split_strat_False.split()

    data_split = Split(hr_promotion, 12, 0.8, 42, True, True, False, True)
    train_dataOn, test_dataOn = data_split.split()

    df_trainOff = pd.DataFrame(train_dataOff)
    df_testOff = pd.DataFrame(test_dataOff)
    train_percent_stratOff = (df_trainOff.is_promoted.value_counts()[1] * 100) / (
        df_trainOff.is_promoted.value_counts()[0]
        + df_trainOff.is_promoted.value_counts()[1]
    )
    test_percent_stratOff = (df_testOff.is_promoted.value_counts()[1] * 100) / (
        df_testOff.is_promoted.value_counts()[0]
        + df_testOff.is_promoted.value_counts()[1]
    )

    df_trainOn = pd.DataFrame(train_dataOn)
    df_testOn = pd.DataFrame(test_dataOn)
    train_percent_stratOn = (df_trainOn.is_promoted.value_counts()[1] * 100) / (
        df_trainOn.is_promoted.value_counts()[0]
        + df_trainOn.is_promoted.value_counts()[1]
    )
    test_percent_stratOn = (df_testOn.is_promoted.value_counts()[1] * 100) / (
        df_testOn.is_promoted.value_counts()[0]
        + df_testOn.is_promoted.value_counts()[1]
    )

    assert abs(train_percent_stratOn - test_percent_stratOn) < abs(
        train_percent_stratOff - test_percent_stratOff
    )


def test_data_split_dropDups_stratifyOnOff():

    """Data Split test with 0.8 rate, drop null, stratify Off. API call: Split(hr_data5, 12, 0.5, 42, True, False, True, False)"""

    data_split_strat_False = Split(hr_promotion, 12, 0.5, 42, True, False, True, False)
    train_dataOff, test_dataOff = data_split_strat_False.split()

    data_split = Split(hr_promotion, 12, 0.5, 42, True, False, True, True)
    train_dataOn, test_dataOn = data_split.split()

    aproxOff = (
        train_dataOff.is_promoted.value_counts()[1]
        - test_dataOff.is_promoted.value_counts()[1]
    ) / train_dataOff.is_promoted.value_counts()[1]
    aproxOn = (
        train_dataOn.is_promoted.value_counts()[1]
        - test_dataOn.is_promoted.value_counts()[1]
    ) / train_dataOn.is_promoted.value_counts()[1]

    assert abs(aproxOff) > abs(aproxOn)


def test_data_split_dropNulls_dropdups_stratifyOnOff():

    """Data Split test with 0.8 rate, drop null, stratify Off. API call: Split(hr_data5, 12, 0.6, 42, True, True, True, False)"""

    hr_promotion_d = hr_promotion.drop_duplicates()

    data_split_strat_False = Split(hr_promotion_d, 12, 0.6, 42, True, True, True, False)
    train_dataOff, test_dataOff = data_split_strat_False.split()

    data_split = Split(hr_promotion_d, 12, 0.6, 42, True, True, True, True)
    train_dataOn, test_dataOn = data_split.split()

    df_trainOff = pd.DataFrame(train_dataOff)
    df_testOff = pd.DataFrame(test_dataOff)
    train_percent_stratOff = (df_trainOff.is_promoted.value_counts()[1] * 100) / (
        df_trainOff.is_promoted.value_counts()[0]
        + df_trainOff.is_promoted.value_counts()[1]
    )
    test_percent_stratOff = (df_testOff.is_promoted.value_counts()[1] * 100) / (
        df_testOff.is_promoted.value_counts()[0]
        + df_testOff.is_promoted.value_counts()[1]
    )

    df_trainOn = pd.DataFrame(train_dataOn)
    df_testOn = pd.DataFrame(test_dataOn)
    train_percent_stratOn = (df_trainOn.is_promoted.value_counts()[1] * 100) / (
        df_trainOn.is_promoted.value_counts()[0]
        + df_trainOn.is_promoted.value_counts()[1]
    )
    test_percent_stratOn = (df_testOn.is_promoted.value_counts()[1] * 100) / (
        df_testOn.is_promoted.value_counts()[0]
        + df_testOn.is_promoted.value_counts()[1]
    )

    on = abs(train_percent_stratOn - test_percent_stratOn)
    off = abs(train_percent_stratOff - test_percent_stratOff)

    assert abs(train_percent_stratOn - test_percent_stratOn) < abs(
        train_percent_stratOff - test_percent_stratOff
    )
