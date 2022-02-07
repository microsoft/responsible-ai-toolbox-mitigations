import pytest
import sys
import copy
import pandas as pd

sys.path.append("../../../responsible-ai-mitigations")
from raimitigations.dataprocessing import Split

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

pytest.global_variable_1 = "test/datasets/hr_promotion_test"
pytest.hr_promotion_TEST = pd.read_csv(
    "test/datasets/hr_promotion_test" + "/train.csv"
).drop(["employee_id"], axis=1)
pytest.global_variable_3 = "test/datasets/hr_promotion"
pytest.hr_promotion = pd.read_csv("test/datasets/hr_promotion" + "/train.csv").drop(
    ["employee_id"], axis=1
)


# unit test for the Split API

testdata = [(0.9, (45, 46, 5, 46)), (0.5, (25, 46, 25, 46))]


@pytest.mark.Functional
@pytest.mark.parametrize(
    "train_size, expected", testdata, ids=["train size=0.9", "train size=0.5"]
)
def test_data_split_rate_v0(train_size, expected):

    """Data Split test with 0.9 rate, no drop_null and no drop_dupicates. API call: Split(hr_data1, 12, 0.9, 42, True, False, False, False)"""

    hr_data1 = copy.deepcopy(pytest.hr_promotion_TEST)
    data_split = Split(hr_data1, 12, train_size, 42, True, False, False, False)
    train_data, test_data = data_split.split()

    assert train_data.shape[0] == expected[0]
    assert train_data.shape[1] == expected[1]
    assert test_data.shape[0] == expected[2]
    assert test_data.shape[1] == expected[3]


@pytest.mark.Functional
def test_data_split_remove_duplicate():

    """Data Split test with 0.9 rate, drop_dupicates but no drop_null. API call: Split(hr_data2, 12, 0.9, 42, True, False, True, False)"""

    hr_data2 = copy.deepcopy(pytest.hr_promotion_TEST)

    data_split = Split(hr_data2, 12, 0.9, 42, True, False, True, False)
    train_data, test_data = data_split.split()

    assert train_data.shape[0] == 43
    assert train_data.shape[1] == 46
    assert test_data.shape[0] == 5
    assert test_data.shape[1] == 46


@pytest.mark.Functional
def test_data_split_drop_Null():

    """Data Split test with 0.9 rate, drop_null but no drop_dupicates. API call: Split(hr_data3, 12, 0.9, 42, True, True, False, False)"""

    hr_data3 = copy.deepcopy(pytest.hr_promotion_TEST)

    data_split = Split(hr_data3, 12, 0.9, 42, True, True, False, False)
    train_data, test_data = data_split.split()

    assert train_data.shape[0] == 38
    assert train_data.shape[1] == 45
    assert test_data.shape[0] == 5
    assert test_data.shape[1] == 45


@pytest.mark.Functional
def test_data_split_stratifyOn():

    """Data Split test with 0.8 rate, stratify On. API call: Split(hr_data4, 12, 0.8, 42, True, False, False, True)"""

    hr_data4 = copy.deepcopy(pytest.hr_promotion)

    data_split_strat_False = Split(hr_data4, 12, 0.8, 42, True, False, False, False)
    train_dataOff, test_dataOff = data_split_strat_False.split()

    data_split = Split(hr_data4, 12, 0.8, 42, True, False, False, True)
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


@pytest.mark.Functional
def test_data_split_dropNulls_stratifyOff():

    """Data Split test with 0.8 rate, drop null, stratify Off. API call: Split(hr_data5, 12, 0.8, 42, True, True, False, False)"""

    hr_data5 = copy.deepcopy(pytest.hr_promotion)

    data_split_strat_False = Split(hr_data5, 12, 0.8, 42, True, True, False, False)
    train_dataOff, test_dataOff = data_split_strat_False.split()

    data_split = Split(hr_data5, 12, 0.8, 42, True, True, False, True)
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


@pytest.mark.Functional
def test_data_split_dropDups_stratifyOnOff():

    """Data Split test with 0.8 rate, drop null, stratify Off. API call: Split(hr_data5, 12, 0.5, 42, True, False, True, False)"""

    hr_data6 = copy.deepcopy(pytest.hr_promotion)

    data_split_strat_False = Split(hr_data6, 12, 0.5, 42, True, False, True, False)
    train_dataOff, test_dataOff = data_split_strat_False.split()

    data_split = Split(hr_data6, 12, 0.5, 42, True, False, True, True)
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


@pytest.mark.Functional
def test_data_split_dropNulls_dropdups_stratifyOnOff():

    """Data Split test with 0.8 rate, drop null, stratify Off. API call: Split(hr_data5, 12, 0.5, 42, True, True, True, False)"""

    hr_data5 = copy.deepcopy(pytest.hr_promotion)
    hr_data5 = hr_data5.drop_duplicates()

    data_split_strat_False = Split(hr_data5, 12, 0.6, 42, True, True, True, False)
    train_dataOff, test_dataOff = data_split_strat_False.split()

    data_split = Split(hr_data5, 12, 0.6, 42, True, True, True, True)
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
