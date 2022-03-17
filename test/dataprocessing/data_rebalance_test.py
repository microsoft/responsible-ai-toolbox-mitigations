from _pytest.compat import STRING_TYPES
import pytest
import sys
import copy
import pandas as pd

from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

from raimitigations.dataprocessing import Rebalance

from common_utils import (create_hr_promotion_data, create_hr_promotion_10_data)

# data_rebalance = Rebalance(dataset, target, sampling_strategy, random_state, smote_tomek, smote, tomek)
# Parameters:
#           dataset
#           target
#           sampling_strategy='auto'
#           random_state=None
#           smote_tomek=None
#           smote=None
#           tomek=None

# LOGGER = logging.getLogger(__name__)


""" pytest.hr_test_PATH = "test/datasets/hr_promotion_test"
pytest.hr_promotion_TEST = pd.read_csv(pytest.hr_test_PATH + "/train.csv").drop(
    ["employee_id"], axis=1
) """

hr_promotion = create_hr_promotion_data()
hr_promotion_10 = create_hr_promotion_10_data()

""" pytest.hr_PATH = "test/datasets/hr_promotion"
pytest.hr_promotion = pd.read_csv(pytest.hr_PATH + "/train.csv").drop(
    ["employee_id"], axis=1
) """
""" 
pytest.hr_10_PATH = "test/datasets/hr_promotion_10"
pytest.hr_promotion_10 = pd.read_csv(pytest.hr_10_PATH + "/train.csv").drop(
    ["employee_id"], axis=1
) """
""" 
pytest.hr_30_PATH = "test/datasets/hr_promotion_30"
pytest.hr_promotion_30 = pd.read_csv(pytest.hr_30_PATH + "/train.csv").drop(
    ["employee_id"], axis=1
)

pytest.hr_50_PATH = "test/datasets/hr_promotion_50"
pytest.hr_promotion_50 = pd.read_csv(pytest.hr_50_PATH + "/train.csv").drop(
    ["employee_id"], axis=1
) """

hr_data_set = copy.deepcopy(pytest.hr_promotion)
hr_data_set = hr_data_set.drop_duplicates()
hr_data_set = pd.get_dummies(hr_data_set, drop_first=False)
hr_data_set.dropna(inplace=True)

hr_data_Small = copy.deepcopy(pytest.hr_promotion_10)
hr_data_Small = hr_data_Small.drop_duplicates()
hr_data_Small = pd.get_dummies(hr_data_Small, drop_first=False)
hr_data_Small.dropna(inplace=True)

""" hr_data_10 = copy.deepcopy(pytest.hr_promotion_10)
hr_data_30 = copy.deepcopy(pytest.hr_promotion_30)
hr_data_50 = copy.deepcopy(pytest.hr_promotion_50) """

tomek = TomekLinks(sampling_strategy="auto")
smote = SMOTE(sampling_strategy="auto", random_state=42)
smote_tomek = SMOTETomek(sampling_strategy="auto", random_state=42)

target_index_promoted = "is_promoted"
target_index_KPI = "KPIs_met >80%"
target_index_previous_year_rating = "previous_year_rating"
target_index_gender_f = "gender_f"
target_index_no_of_trainings = "no_of_trainings"

rebal_data = [
    (hr_data_set, target_index_promoted, "majority", 42, None, smote, None, "equal"),
    (hr_data_set, target_index_promoted, "auto", 42, None, None, tomek, "lower"),
    (hr_data_set, target_index_promoted, "auto", 42, smote_tomek, None, None, "equal"),
    (hr_data_set, target_index_promoted, "auto", 42, None, smote, tomek, "higher"),
    (hr_data_set, target_index_promoted, "auto", 42, None, None, None, "equal"),
    (
        hr_data_set,
        target_index_promoted,
        "not minority",
        42,
        None,
        smote,
        None,
        "equal",
    ),
    (hr_data_set, target_index_promoted, "all", 42, None, smote, None, "equal"),
    (hr_data_set, target_index_promoted, "minority", 42, None, smote, None, "equal"),
    (hr_data_set, target_index_promoted, "majority", 42, None, smote, None, "equal"),
    (hr_data_set, target_index_promoted, "auto", 42, None, smote, None, "equal"),
    (hr_data_set, target_index_promoted, "auto", None, None, smote, None, "equal"),
    (hr_data_Small, target_index_promoted, "auto", None, None, smote, None, "equal"),
]


# unit test for the Rebalance API


@pytest.mark.Functional
@pytest.mark.parametrize(
    "hr_data_set, target_index, random_state, seed, smote_tomek, smote, tomek, expected",
    rebal_data,
)
def test_data_rebal_p(
    hr_data_set, target_index, random_state, seed, smote_tomek, smote, tomek, expected
):

    if target_index == "is_promoted":
        target_index_promoted = hr_data_set.columns.get_loc(target_index)
    else:
        target_index_n_trainings = hr_data_set.columns.get_loc(target_index)

    # testing random_state input
    assert type(random_state) == str
    assert random_state in [
        "not majority",
        "auto",
        "not minority",
        "all",
        "minority",
        "majority",
    ]

    random_state_str = str(random_state)
    seed_str = str(seed)

    if smote_tomek == None:
        smote_tomek_str = "None"
    else:
        smote_tomek_str = smote_tomek.__class__.__name__

    if smote == None:
        smote_str = "None"
    else:
        smote_str = smote.__class__.__name__

    if tomek == None:
        tomek_str = "None"
    else:
        tomek_str = tomek.__class__.__name__

    test_data_rebal_p.__doc__ = (
        "Data Rebalance test with the objects to use. API call: Rebalance (data set size = "
        + str(hr_data_set.shape[0])
        + " rows, target index = "
        + target_index
        + ", random state = "
        + random_state_str
        + ", seed = "
        + seed_str
        + ", smote_tomek = "
        + smote_tomek_str
        + ", smote = "
        + smote_str
        + ", tomek = "
        + tomek_str
        + ")"
    )

    data_rebal = Rebalance(
        hr_data_set, "is_promoted", random_state, seed, smote_tomek, smote, tomek
    )
    result_p = data_rebal.rebalance()

    result_p.shape

    initialPromoted = hr_data_set.is_promoted.value_counts()[1]
    initialNotPromoted = hr_data_set.is_promoted.value_counts()[0]
    promoted = result_p.is_promoted.value_counts()[1]
    notPromoted = result_p.is_promoted.value_counts()[0]

    if expected == "equal":
        assert promoted == notPromoted
    elif expected == "lower":
        assert notPromoted < initialNotPromoted
        assert initialPromoted == promoted
    elif expected == "higher":
        assert initialPromoted < promoted
        assert notPromoted == initialNotPromoted


@pytest.mark.Functional
def test_data_rebal_default():

    """Data Rebalance test with smote object to use. API call: Rebalance(hr_promotion_10, target_index, 'auto', seed)"""

    hr_data1 = copy.deepcopy(pytest.hr_promotion_10)

    # handle duplicates
    hr_data1 = hr_data1.drop_duplicates()

    # random state
    seed = 42

    # OneHotEncoder for categorical features
    hr_data11 = pd.get_dummies(hr_data1, drop_first=False)

    # handle null values
    hr_data11.dropna(inplace=True)

    target_index = hr_data11.columns.get_loc("is_promoted")
    data_rebalance = Rebalance(hr_data11, target_index, "auto", seed)
    result = data_rebalance.rebalance()

    result.shape

    promoted = result.is_promoted.value_counts()[1]
    notPromoted = result.is_promoted.value_counts()[0]

    assert promoted == notPromoted


@pytest.mark.Functional
def test_data_rebal_default1():

    """Data Rebalance test with smote and tomek object to use. API call: Rebalance(hr_promotion_10, target_index, 'auto', seed, None)"""

    hr_data2 = copy.deepcopy(pytest.hr_promotion_10)

    # handle duplicates
    hr_data21 = hr_data2.drop_duplicates()

    # random state
    seed = 42

    # OneHotEncoder for categorical features
    hr_data21 = pd.get_dummies(hr_data21, drop_first=False)

    # handle null values
    hr_data21.dropna(inplace=True)

    target_index = hr_data21.columns.get_loc("is_promoted")
    data_rebalance = Rebalance(hr_data21, target_index, "auto", seed, None)
    result = data_rebalance.rebalance()

    result.shape

    promoted = result.is_promoted.value_counts()[1]
    notPromoted = result.is_promoted.value_counts()[0]

    assert promoted == notPromoted


@pytest.mark.Functional
def test_data_rebal_default2():

    """Data Rebalance test with Smote object to use as default. API call: Rebalance(hr_promotion_10, target_index, 'auto', seed, None, None)"""

    hr_data3 = copy.deepcopy(pytest.hr_promotion_10)

    # handle duplicates
    hr_data31 = hr_data3.drop_duplicates()

    # random state
    seed = 42

    # OneHotEncoder for categorical features
    hr_data31 = pd.get_dummies(hr_data31, drop_first=False)

    # handle null values
    hr_data31.dropna(inplace=True)

    target_index = hr_data31.columns.get_loc("is_promoted")
    data_rebalance = Rebalance(hr_data31, target_index, "auto", seed, None, None)
    result = data_rebalance.rebalance()

    promoted = result.is_promoted.value_counts()[1]
    notPromoted = result.is_promoted.value_counts()[0]

    assert promoted == notPromoted


@pytest.mark.Functional
def test_data_rebal_smotetomekLargeDS():

    """Data Rebalance test with smote_tomek object to use against big dataset. API call: Rebalance(hr_promotion, target_index, 'auto', seed, smote_tomek)"""

    hr_data4 = copy.deepcopy(pytest.hr_promotion)

    # handle duplicates
    hr_data41 = hr_data4.drop_duplicates()

    # random state
    seed = 42

    # OneHotEncoder for categorical features
    hr_data41 = pd.get_dummies(hr_data41, drop_first=False)

    # handle null values
    hr_data41.dropna(inplace=True)

    smote_tomek = SMOTETomek(sampling_strategy="auto", random_state=seed)

    target_index = hr_data41.columns.get_loc("is_promoted")
    data_rebalance = Rebalance(hr_data41, target_index, "auto", seed, smote_tomek)

    result = data_rebalance.rebalance()

    promoted = result.is_promoted.value_counts()[1]
    notPromoted = result.is_promoted.value_counts()[0]

    assert promoted == notPromoted


@pytest.mark.Functional
@pytest.mark.xfail
def test_data_rebal_smotetomekSmallDS():

    """Data Rebalance test with smote_tomek object to use. API call: Rebalance(hr_promotion_10, target_index, 'auto', seed, smote_tomek)"""

    hr_data5 = copy.deepcopy(pytest.hr_promotion_10)

    # handle duplicates
    hr_data51 = hr_data5.drop_duplicates()

    # random state
    seed = 42

    # OneHotEncoder for categorical features
    hr_data51 = pd.get_dummies(hr_data51, drop_first=False)

    # handle null values
    hr_data51.dropna(inplace=True)

    smote_tomek = SMOTETomek(sampling_strategy="auto", random_state=seed)

    target_index = hr_data51.columns.get_loc("is_promoted")
    data_rebalance = Rebalance(hr_data51, target_index, "auto", seed, smote_tomek)
    with pytest.raises(ValueError) as excinfo:
        result = data_rebalance.rebalance()
    assert "Expected n_neighbors" in str(excinfo.value), "this is catch exception"


