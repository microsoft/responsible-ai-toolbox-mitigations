from _pytest.compat import STRING_TYPES
import pytest
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

seed = 42

hr_promotion = create_hr_promotion_data()
hr_promotion_10 = create_hr_promotion_10_data()

hr_data_set = hr_promotion.drop_duplicates()
hr_data_set = pd.get_dummies(hr_data_set, drop_first=False)
hr_data_set.dropna(inplace=True)

hr_data_Small = hr_promotion_10.drop_duplicates()
hr_data_Small = pd.get_dummies(hr_data_Small, drop_first=False)
hr_data_Small.dropna(inplace=True)

tomek = TomekLinks(sampling_strategy="auto")
smote = SMOTE(sampling_strategy="auto", random_state=42)
smote_tomek = SMOTETomek(sampling_strategy="auto", random_state=42)

target_index_promoted = "is_promoted"
target_index_KPI = "KPIs_met >80%"
target_index_previous_year_rating = "previous_year_rating"
target_index_gender_f = "gender_f"
target_index_no_of_trainings = "no_of_trainings"

rebal_data = [
    (hr_data_set, target_index_promoted, "majority", seed, None, smote, None, "equal"),
    (hr_data_set, target_index_promoted, "auto", seed, None, None, tomek, "lower"),
    (hr_data_Small, target_index_promoted, "auto", seed, smote_tomek, None, None, "equal"),
    (hr_data_set, target_index_promoted, "auto", seed, None, smote, tomek, "higher"),
    (hr_data_set, target_index_promoted, "auto", seed, None, None, None, "equal"),
    (hr_data_set, target_index_promoted, "not minority", seed, None, smote, None, "equal"),
    (hr_data_Small, target_index_promoted, "all", seed, None, smote, None, "equal"),
    (hr_data_set, target_index_promoted, "minority", seed, None, smote, None, "equal"),
    (hr_data_set, target_index_promoted, "majority", seed, None, smote, None, "equal"),
    (hr_data_set, target_index_promoted, "auto", seed, None, smote, None, "equal"),
    (hr_data_set, target_index_promoted, "auto", None, None, smote, None, "equal"),
    (hr_data_Small, target_index_promoted, "auto", None, None, smote, None, "equal"),
    (hr_data_Small, target_index_KPI, "auto", None, None, smote, None, "equal"),
    (hr_data_set, target_index_KPI, "auto", seed, smote_tomek, None, None, "equal"),
    (hr_data_Small, target_index_KPI, "auto", seed, None, smote, tomek, "higher"),
]


# unit test for the Rebalance API

@pytest.mark.parametrize(
    "hr_data_set, target_index, random_state, seed, smote_tomek, smote, tomek, expected",
    rebal_data,
)
def test_data_rebal_p(
    hr_data_set, target_index, random_state, seed, smote_tomek, smote, tomek, expected
):

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
        hr_data_set, target_index, random_state, seed, smote_tomek, smote, tomek
    )
    result_p = data_rebal.rebalance()

    result_p.shape

    initial_target_on = hr_data_set[target_index].value_counts()[1]
    initial_target_off = hr_data_set[target_index].value_counts()[0]
    rebal_target_on = result_p[target_index].value_counts()[1]
    rebal_target_off = result_p[target_index].value_counts()[0]

    if expected == "equal":
        assert rebal_target_on == rebal_target_off
    elif expected == "lower":
        assert rebal_target_off < initial_target_off
        assert initial_target_on == rebal_target_on
    elif expected == "higher":
        assert initial_target_on < rebal_target_on
        assert rebal_target_off == initial_target_off


def test_data_rebal_default():

    """Data Rebalance test with smote object to use. API call: Rebalance(hr_promotion_10, target_index, 'auto', seed)"""

    data_rebalance = Rebalance(hr_data_Small, target_index_promoted, "auto", seed)
    result = data_rebalance.rebalance()

    result.shape

    promoted = result.is_promoted.value_counts()[1]
    notPromoted = result.is_promoted.value_counts()[0]

    assert promoted == notPromoted


def test_data_rebal_default1():

    """Data Rebalance test with smote and tomek object to use. API call: Rebalance(hr_promotion_10, target_index, 'auto', seed, None)"""

    data_rebalance = Rebalance(hr_data_Small, target_index_promoted, "auto", seed, None)
    result = data_rebalance.rebalance()

    result.shape

    promoted = result.is_promoted.value_counts()[1]
    notPromoted = result.is_promoted.value_counts()[0]

    assert promoted == notPromoted


def test_data_rebal_default2():

    """Data Rebalance test with Smote object to use as default. API call: Rebalance(hr_promotion_10, target_index, 'auto', seed, None, None)"""

    data_rebalance = Rebalance(hr_data_Small, target_index_promoted, "auto", seed, None, None)
    result = data_rebalance.rebalance()

    promoted = result.is_promoted.value_counts()[1]
    notPromoted = result.is_promoted.value_counts()[0]

    assert promoted == notPromoted


def test_data_rebal_smotetomekLargeDS():

    """Data Rebalance test with smote_tomek object to use against big dataset. API call: Rebalance(hr_promotion, target_index, 'auto', seed, smote_tomek)"""

    smote_tomek = SMOTETomek(sampling_strategy="auto", random_state=seed)

    data_rebalance = Rebalance(hr_data_set, target_index_promoted, "auto", seed, smote_tomek)

    result = data_rebalance.rebalance()

    promoted = result.is_promoted.value_counts()[1]
    notPromoted = result.is_promoted.value_counts()[0]

    assert promoted == notPromoted


@pytest.mark.xfail
def test_data_rebal_smotetomekSmallDS():

    """Data Rebalance test with smote_tomek object to use. API call: Rebalance(hr_promotion_10, target_index, 'auto', seed, smote_tomek)"""

    smote_tomek = SMOTETomek(sampling_strategy="auto", random_state=seed)

    target_index = hr_data_Small.columns.get_loc("is_promoted")
    data_rebalance = Rebalance(hr_data_Small, target_index, "auto", seed, smote_tomek)
    with pytest.raises(ValueError) as excinfo:
        result = data_rebalance.rebalance()
    assert "Expected n_neighbors" in str(excinfo.value), "this is catch exception"


