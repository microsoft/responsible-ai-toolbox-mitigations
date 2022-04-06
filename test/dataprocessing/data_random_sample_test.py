import pytest
import pandas as pd

from pandas import read_csv
from common_utils import (create_hr_promotion_data, create_hr_promotion_10_data, validate_rows)

from raimitigations.dataprocessing import RandomSample


# data_random_sample = RandomSample(data_set, target, sample_size, stratify)
# Parameters:
#           dataset
#           target
#           sample_size,
#           categorical_features = True, 
#           drop_null = True,
#           drop_duplicates = False,
#           stratify = False


hr_promotion = create_hr_promotion_data()
hr_promotion_10 = create_hr_promotion_10_data()
target_promoted = "is_promoted"
target_no_trainings = "no_of_trainings"

@pytest.fixture
def target_index_promoted():
    target = hr_promotion_10.columns.get_loc(target_promoted)
    return target


@pytest.fixture
def target_index_promoted_largeDS():
    target = hr_promotion.columns.get_loc(target_promoted)
    return target


@pytest.fixture
def target_index_no_trainings():
    target = hr_promotion_10.columns.get_loc(target_no_trainings)
    return target


# tests for the DataSplit API


def test_data_randomSample_split_categorical(target_index_promoted):

    """Data RandomSample test with split rate 0.2, drop_null and categorical features. API call: RandomSample(hr_data, target_index, 0.2, True)"""

    data_sample = RandomSample(hr_promotion, target_index_promoted, 0.2, True)
    random_sample_split = data_sample.random_sample()

    assert random_sample_split.shape[0] == validate_rows(hr_promotion, 0.2, True, False)


def test_data_randomSample_split_not_categorical(target_index_no_trainings):

    """Data RandomSample test with split rate 0.2, drop_null and not categorical features. API call: RandomSample(hr_data, target_index, 0.3, False)"""

    data_sample = RandomSample(hr_promotion, target_index_no_trainings, 0.3, False)
    random_sample_split = data_sample.random_sample()

    assert random_sample_split.shape[0] == validate_rows(hr_promotion, 0.3, True, False)


def test_data_randomSample_default_dup():

    """Data RandomSample test with split rate 0.2, drop_null and drop_duplicates. API call: RandomSample(hr_data, target_index_promoted, 0.4, True, True)"""

    data_sample = RandomSample(hr_promotion, target_promoted, 0.4, True, True)
    random_sample = data_sample.random_sample()

    assert random_sample.shape[0] == validate_rows(hr_promotion, 0.4, True, False)


def test_data_randomSample_split_dropNulDup(target_index_promoted):

    """Data RandomSample test with split rate 0.2, drop_null and drop_duplicates. API call: RandomSample(hr_data, target_index_promoted, 0.4, False, True, True, True)"""

    data_sample = RandomSample(hr_promotion_10, target_index_promoted, 0.4, True, True, True)
    random_sample = data_sample.random_sample()

    assert random_sample.shape[0] == validate_rows(hr_promotion_10, 0.4, True, True)


def test_data_randomSample_split_dropDup(target_index_promoted):

    """Data RandomSample test with split rate 0.3, drop_null and drop_duplicates. API call: RandomSample(hr_data, target_index_promoted, 0.3, True, False, True)"""

    data_sample = RandomSample(hr_promotion_10, target_index_promoted, 0.3, True, False, True)
    random_sample = data_sample.random_sample()

    assert random_sample.shape[0] == validate_rows(hr_promotion_10, 0.3, False, True)


def test_data_randomSample_split_target_dropDup(target_index_no_trainings):

    """Data RandomSample test with split rate 0.4, target_index and drop_duplicates. API call: RandomSample(hr_data, target_index_no_trainings, 0.4, True, False, True)"""

    data_sample = RandomSample(
        hr_promotion_10, target_index_no_trainings, 0.4, True, False, True
    )
    random_sample = data_sample.random_sample()

    assert random_sample.shape[0] == validate_rows(hr_promotion_10, 0.4, False, True)


def test_data_randomSample_split_noDrops(target_index_promoted):

    """Data RandomSample test with split rate 0.2, no drop_null and no drop_duplicates. API call: RandomSample(hr_data, target_index_promoted, 0.2, True, False, False, False)"""

    data_sample = RandomSample(
        hr_promotion_10, target_index_promoted, 0.2, True, False, False, False
    )
    random_sample = data_sample.random_sample()

    assert random_sample.shape[0] == validate_rows(hr_promotion_10, 0.2, False, False)


def test_data_randomSample_stratifyOnOff(target_index_promoted_largeDS):

    """Data RandomSample test with split rate 0.5. API call: RandomSample(data_large_set, target_index_promoted, 0.5, True, False, False, False)"""

    data_sample = RandomSample(
        hr_promotion, target_index_promoted_largeDS, 0.8, True, False, False, False
    )
    random_sample = data_sample.random_sample()

    data_sample_stratOn = RandomSample(
        hr_promotion, target_index_promoted_largeDS, 0.8, True, False, False, True
    )
    random_sample_stratOn1 = data_sample_stratOn.random_sample()

    stratOff_random_1_over_0 = (
        random_sample.is_promoted.value_counts()[1]
        / random_sample.is_promoted.value_counts()[0]
    )
    stratOn_random_1_over_0 = (
        random_sample_stratOn1.is_promoted.value_counts()[1]
        / random_sample_stratOn1.is_promoted.value_counts()[0]
    )
    all_set_1_over_0 = (
        hr_promotion.is_promoted.value_counts()[1]
        / hr_promotion.is_promoted.value_counts()[0]
    )

    assert abs(all_set_1_over_0 - stratOn_random_1_over_0) <= abs(
        all_set_1_over_0 - stratOff_random_1_over_0
    )


def test_data_randomSample_stratifyOnOff_dropnull(
    target_index_promoted_largeDS
    ):

    """Data RandomSample test with split rate 0.8, drop null and stratify. API call: RandomSample(hr_data, target_index_promoted, 0.8, True, True, False, False)"""

    hr_promotion.dropna(axis=0, inplace=True)

    data_sample = RandomSample(
        hr_promotion, target_index_promoted_largeDS, 0.8, False, True, False, False
    )
    random_sample = data_sample.random_sample()

    data_sample_stratOn = RandomSample(
        hr_promotion, target_index_promoted_largeDS, 0.8, False, True, False, True
    )
    random_sample_stratOn = data_sample_stratOn.random_sample()

    stratOff_random_1_over_0 = (
        random_sample.is_promoted.value_counts()[1]
        / random_sample.is_promoted.value_counts()[0]
    )
    stratOn_random_1_over_0 = (
        random_sample_stratOn.is_promoted.value_counts()[1]
        / random_sample_stratOn.is_promoted.value_counts()[0]
    )
    all_set_1_over_0 = (
        hr_promotion.is_promoted.value_counts()[1]
        / hr_promotion.is_promoted.value_counts()[0]
    )

    assert abs(all_set_1_over_0 - stratOn_random_1_over_0) <= abs(
        all_set_1_over_0 - stratOff_random_1_over_0
    )


def test_data_randomSample_stratifyOnOff_dropNull_dropdup(
    target_index_promoted_largeDS
):

    """Data RandomSample test with split rate 0.8, drop null, frop duplicates, checking stratify On_Off. API call: RandomSample(hr_data, target_index_promoted, 0.7, True, True, True, False)"""

    hr_promotion_drop = hr_promotion.drop_duplicates()
    hr_promotion_drop.dropna(axis=0, inplace=True)
    
    data_sample = RandomSample(
        hr_promotion, target_index_promoted_largeDS, 0.7, True, True, True, False
    )
    random_sample = data_sample.random_sample()

    data_sample_stratOn = RandomSample(
        hr_promotion, target_index_promoted_largeDS, 0.7, True, True, True, True
    )
    random_sample_stratOn = data_sample_stratOn.random_sample()

    stratOff_random_1_over_0 = (
        random_sample.is_promoted.value_counts()[1]
        / random_sample.is_promoted.value_counts()[0]
    )
    stratOn_random_1_over_0 = (
        random_sample_stratOn.is_promoted.value_counts()[1]
        / random_sample_stratOn.is_promoted.value_counts()[0]
    )
    all_set_1_over_0 = (
        hr_promotion_drop.is_promoted.value_counts()[1]
        / hr_promotion_drop.is_promoted.value_counts()[0]
    )

    assert abs(all_set_1_over_0 - stratOn_random_1_over_0) <= abs(
        all_set_1_over_0 - stratOff_random_1_over_0
    )



