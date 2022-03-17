import pytest
import sys
import copy
import pandas as pd

from pandas import read_csv
from common_utils import (create_hr_promotion_data, create_hr_promotion_10_data)

from raimitigations.dataprocessing import RandomSample


# data_random_sample = RandomSample(data_set, target, sample_size, stratify)
# Parameters:
#           dataset
#           target
#           sample_size,
#           stratify = False

pytest.hr_promotion = create_hr_promotion_data()
pytest.hr_promotion_10 = create_hr_promotion_10_data()

@pytest.fixture
def data_set():
    hr_data = copy.deepcopy(pytest.hr_promotion_10)
    return hr_data


@pytest.fixture
def data_large_set():
    hr_data = copy.deepcopy(pytest.hr_promotion)
    return hr_data


@pytest.fixture
def target_index_promoted(data_set):
    target = data_set.columns.get_loc("is_promoted")
    return target


@pytest.fixture
def target_index_promoted_largeDS(data_large_set):
    target = data_large_set.columns.get_loc("is_promoted")
    return target


@pytest.fixture
def target_index_no_trainings(data_set):
    target = data_set.columns.get_loc("no_of_trainings")
    return target


# tests for the DataSplit API


@pytest.mark.Functional
def test_data_randomSample_split_categorical(data_set, target_index_promoted):

    """Data RandomSample test with split rate 0.2, drop_null and categorical features. API call: RandomSample(hr_data, target_index, 0.2, True)"""

    data_sample = RandomSample(data_set, target_index_promoted, 0.2, True)

    random_sample_split = data_sample.random_sample()

    assert random_sample_split.shape[0] == validate_sampling(data_set, 0.2)


@pytest.mark.Functional
def test_data_randomSample_split_not_categorical(data_set, target_index_promoted):

    """Data RandomSample test with split rate 0.2, drop_null and not categorical features. API call: RandomSample(hr_data, target_index, 0.2, False)"""

    data_sample = RandomSample(data_set, target_index_promoted, 0.3, False)

    random_sample_split = data_sample.random_sample()

    assert random_sample_split.shape[0] == validate_sampling(data_set, 0.3)


@pytest.mark.Functional
def test_data_randomSample_default_dup(data_set, target_index_promoted):

    """Data RandomSample test with split rate 0.2, drop_null and drop_duplicates. API call: RandomSample(hr_data, target_index_promoted, 0.2, True, True)"""

    data_sample = RandomSample(data_set, target_index_promoted, 0.3, True, True)

    random_sample = data_sample.random_sample()

    assert random_sample.shape[0] == validate_sampling(data_set, 0.3)


@pytest.mark.Functional
def test_data_randomSample_split_dropNulDup(data_set, target_index_promoted):

    """Data RandomSample test with split rate 0.2, drop_null and drop_duplicates. API call: RandomSample(hr_data, target_index_promoted, 0.2, False, True, True, True)"""

    data_sample = RandomSample(data_set, target_index_promoted, 0.4, True, True, True)

    random_sample = data_sample.random_sample()

    assert random_sample.shape[0] == validate_sampling(data_set, 0.4)


@pytest.mark.Functional
def test_data_randomSample_split_dropDup(data_set, target_index_promoted):

    """Data RandomSample test with split rate 0.3, drop_null and drop_duplicates. API call: RandomSample(hr_data, target_index_promoted, 0.2, True, False, True)"""

    data_sample = RandomSample(data_set, target_index_promoted, 0.3, True, False, True)

    random_sample = data_sample.random_sample()

    assert random_sample.shape[0] == validate_sampling(data_set, 0.3)


@pytest.mark.Functional
def test_data_randomSample_split_noDrops(data_set, target_index_promoted):

    """Data RandomSample test with split rate 0.2, no drop_null and no drop_duplicates. API call: RandomSample(hr_data, target_index_promoted, 0.2, True, False, False, False)"""

    data_sample = RandomSample(
        data_set, target_index_promoted, 0.2, True, False, False, False
    )

    random_sample = data_sample.random_sample()

    assert random_sample.shape[0] == validate_sampling(data_set, 0.2)


@pytest.mark.Functional
def test_data_randomSample_stratifyOnOff(data_large_set, target_index_promoted_largeDS):

    """Data RandomSample test with split rate 0.5. API call: RandomSample(data_large_set, target_index_promoted, 0.5, True, False, False, False)"""

    data_sample1 = RandomSample(
        data_large_set, target_index_promoted_largeDS, 0.8, True, False, False, False
    )
    random_sample1 = data_sample1.random_sample()

    data_sample_stratOn1 = RandomSample(
        data_large_set, target_index_promoted_largeDS, 0.8, True, False, False, True
    )
    random_sample_stratOn1 = data_sample_stratOn1.random_sample()

    stratOff_random_1_over_0 = (
        random_sample1.is_promoted.value_counts()[1]
        / random_sample1.is_promoted.value_counts()[0]
    )
    stratOn_random_1_over_0 = (
        random_sample_stratOn1.is_promoted.value_counts()[1]
        / random_sample_stratOn1.is_promoted.value_counts()[0]
    )
    all_set_1_over_0 = (
        data_large_set.is_promoted.value_counts()[1]
        / data_large_set.is_promoted.value_counts()[0]
    )

    assert abs(all_set_1_over_0 - stratOn_random_1_over_0) < abs(
        all_set_1_over_0 - stratOff_random_1_over_0
    )


@pytest.mark.Functional
def test_data_randomSample_stratifyOnOff_dropnull(
    data_large_set, target_index_promoted_largeDS
):

    """Data RandomSample test with split rate 0.8, drop null and stratify. API call: RandomSample(hr_data, target_index_promoted, 0.8, True, True, False, False)"""

    data_sample2 = RandomSample(
        data_large_set, target_index_promoted_largeDS, 0.8, False, True, False, False
    )
    random_sample2 = data_sample2.random_sample()

    data_sample_stratOn2 = RandomSample(
        data_large_set, target_index_promoted_largeDS, 0.8, False, True, False, True
    )
    random_sample_stratOn2 = data_sample_stratOn2.random_sample()

    stratOff_random_1_over_0 = (
        random_sample2.is_promoted.value_counts()[1]
        / random_sample2.is_promoted.value_counts()[0]
    )
    stratOn_random_1_over_0 = (
        random_sample_stratOn2.is_promoted.value_counts()[1]
        / random_sample_stratOn2.is_promoted.value_counts()[0]
    )
    all_set_1_over_0 = (
        data_large_set.is_promoted.value_counts()[1]
        / data_large_set.is_promoted.value_counts()[0]
    )

    assert abs(all_set_1_over_0 - stratOn_random_1_over_0) < abs(
        all_set_1_over_0 - stratOff_random_1_over_0
    )


@pytest.mark.Functional
def test_data_randomSample_stratifyOnOff_dropNull_dropdup(
    data_large_set, target_index_promoted_largeDS
):

    """Data RandomSample test with split rate 0.8, drop null, frop duplicates, checking stratify On_Off. API call: RandomSample(hr_data, target_index_promoted, 0.7, True, True, True, False)"""

    data_sample3 = RandomSample(
        data_large_set, target_index_promoted_largeDS, 0.7, True, True, True, False
    )
    random_sample3 = data_sample3.random_sample()

    data_sample_stratOn3 = RandomSample(
        data_large_set, target_index_promoted_largeDS, 0.7, True, True, True, True
    )
    random_sample_stratOn3 = data_sample_stratOn3.random_sample()

    stratOff_random_1_over_0 = (
        random_sample3.is_promoted.value_counts()[1]
        / random_sample3.is_promoted.value_counts()[0]
    )
    stratOn_random_1_over_0 = (
        random_sample_stratOn3.is_promoted.value_counts()[1]
        / random_sample_stratOn3.is_promoted.value_counts()[0]
    )
    all_set_1_over_0 = (
        data_large_set.is_promoted.value_counts()[1]
        / data_large_set.is_promoted.value_counts()[0]
    )

    assert abs(all_set_1_over_0 - stratOn_random_1_over_0) < abs(
        all_set_1_over_0 - stratOff_random_1_over_0
    )


@pytest.mark.Functional
def test_data_randomSample_split_target_dropDup(data_set, target_index_no_trainings):

    """Data RandomSample test with split rate 0.4, target_index and drop_duplicates. API call: RandomSample(hr_data, target_index_no_trainings, 0.4, True, False, True)"""

    data_sample = RandomSample(
        data_set, target_index_no_trainings, 0.4, True, False, True
    )
    random_sample = data_sample.random_sample()

    assert random_sample.shape[0] == 19
    assert random_sample.shape[1] == 46

def validate_sampling (data_set, sample):

    num_rows = int(data_set.shape[0] * sample)

    return num_rows

def validate_sampling_drop (data_set, sample, drop_nul, drop_dup):

    if drop_dup:
      data_set = data_set.drop_duplicates()
      
    num_rows = int(data_set.shape[0] * sample)

    return num_rows