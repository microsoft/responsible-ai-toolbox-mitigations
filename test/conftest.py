import random
from datetime import datetime
from py.xml import html
import pytest
import numpy as np
import torch
import uci_dataset as database

from raimitigations.utils import create_dummy_dataset
from raimitigations.dataprocessing import BasicImputer


SEED = 42


# -----------------------------------
def pytest_html_report_title(report):
    report.title = "The results for errors mitigation APIs testing"


# -----------------------------------
def pytest_html_results_table_header(cells):
    del cells[1]
    del cells[2]
    cells.insert(1, html.th("API"))
    cells.insert(2, html.th("Test"))
    cells.insert(3, html.th("Duration"))
    cells.insert(4, html.th("Description"))
    cells.pop()


# -----------------------------------
def pytest_html_results_table_row(report, cells):
    del cells[1]
    del cells[2]
    cells.insert(1, html.td(getattr(report, "api", "")))
    cells.insert(2, html.td(getattr(report, "test", "")))
    cells.insert(3, html.td(getattr(report, "duration", "")))
    cells.insert(4, html.td(getattr(report, "description", "")))
    cells.pop()


# -----------------------------------
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    reportApi = getattr(item.function, "__module__")[5:]
    reportApi = reportApi[:-5]
    report.api = f"{reportApi}"

    reportTest = getattr(item.function, "__name__")[5:]
    report.test = f"{reportTest}"

    report.description = getattr(item.function, "__doc__")


# -----------------------------------
def _set_seed():
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)


# -----------------------------------
@pytest.fixture
def df_num():
    _set_seed()
    df = create_dummy_dataset(
        samples=500,
        n_features=6,
        n_num_num=0,
        n_cat_num=0,
        n_cat_cat=0,
    )
    return df


# -----------------------------------
@pytest.fixture
def label_col_name():
    return "label"


# -----------------------------------
@pytest.fixture
def df_full():
    _set_seed()
    df = create_dummy_dataset(
        samples=500,
        n_features=6,
        n_num_num=2,
        n_cat_num=2,
        n_cat_cat=0,
        num_num_noise=[0.01, 0.02],
        pct_change=[0.03, 0.05],
    )
    return df


# -----------------------------------
@pytest.fixture
def df_regression():
    _set_seed()
    df = create_dummy_dataset(
        samples=500,
        n_features=6,
        n_num_num=2,
        n_cat_num=2,
        n_cat_cat=0,
        num_num_noise=[0.01, 0.02],
        pct_change=[0.03, 0.05],
        regression=True,
    )
    return df


# -----------------------------------
@pytest.fixture
def df_multiclass():
    _set_seed()
    df = create_dummy_dataset(
        samples=500,
        n_features=6,
        n_num_num=2,
        n_cat_num=2,
        n_cat_cat=0,
        num_num_noise=[0.01, 0.02],
        pct_change=[0.03, 0.05],
        n_classes=4
    )
    return df

# -----------------------------------
@pytest.fixture
def df_multiclass1():
    _set_seed()
    df = create_dummy_dataset(
        samples=5000,
        n_classes=3,
        n_features=6,
        n_num_num=3,
        n_cat_num=3,
        n_cat_cat=0,
        num_num_noise=[0.01, 0.02],
        pct_change=[0.06, 0.15],
    )
    return df


# -----------------------------------
@pytest.fixture
def df_multiclass2():
    _set_seed()
    df = create_dummy_dataset(
        samples=2000,
        n_classes=3,
        n_features=6,
        n_num_num=2,
        n_cat_num=2,
        n_cat_cat=0,
        num_num_noise=[0.01, 0.02],
        pct_change=[0.03, 0.05],
    )
    return df


# -----------------------------------
@pytest.fixture
def df_full_nan():
    _set_seed()
    def add_nan(vec, pct):
        vec = list(vec)
        nan_index = random.sample(range(len(vec)), int(pct * len(vec)))
        for index in nan_index:
            vec[index] = np.nan
        return vec

    df = create_dummy_dataset(
        samples=500,
        n_features=6,
        n_num_num=2,
        n_cat_num=2,
        n_cat_cat=2,
        num_num_noise=[0.01, 0.05],
        pct_change=[0.05, 0.1],
    )
    col_with_nan = ["num_0", "num_3", "num_4", "CN_0_num_0", "CC_1_num_1"]
    for col in col_with_nan:
        if col != "label":
            df[col] = add_nan(df[col], 0.1)
    return df

# -----------------------------------
@pytest.fixture
def df_breast_cancer():
    df = database.load_breast_cancer()
    df["Class"] = df["Class"].replace({"recurrence-events": 1, "no-recurrence-events": 0})
    imputer = BasicImputer(
        col_impute=["breast-quad"],
        specific_col={
            "breast-quad":{"missing_values": np.nan, "strategy": "most_frequent", "fill_value": None}
        },
        verbose=False
    )
    imputer.fit(df)
    df = imputer.transform(df)
    return df

# -----------------------------------
@pytest.fixture
def label_name_bc():
    return "Class"


# -----------------------------------
@pytest.fixture
def label_index_bc():
    return 0

# -----------------------------------
@pytest.fixture
def label_col_index():
    return 6

# -----------------------------------
@pytest.fixture
def df_full_cohort():
    _set_seed()
    df = create_dummy_dataset(
        samples=1000,
        n_features=2,
        n_num_num=0,
        n_cat_num=2,
        n_cat_cat=0,
        num_num_noise=[0.01, 0.05],
        pct_change=[0.05, 0.1],
    )
    df = df.sample(frac=1)
    return df

# -----------------------------------
@pytest.fixture
def label_col_index_cohort():
    return 2

# -----------------------------------
def check_valid_columns(final_list, selected, include_label=True):
    if include_label:
        size = len(final_list) == len(selected) + 1
    else:
        size = len(final_list) == len(selected)
    for v in selected:
        if v not in final_list:
            return False
        break
    return size


# -----------------------------------
def check_fixed_col(fixed, selected):
    for value in fixed:
        if type(value) == int:
            value = str(value)
        if value not in selected:
            return False
    return True


# -----------------------------------
def check_valid_input(df, label_col, X, y):
    if df is not None or label_col is not None:
        if df is None or label_col is None:
            raise ValueError("ERROR: please provide a valid (df, label_col) tuple.")
    elif X is not None or y is not None:
        if X is None or y is None:
            raise ValueError("ERROR: please provide a valid (X, y) tuple.")


# -----------------------------------
def check_cols_num(df, cols):
    for col in cols:
        if df.dtypes[col] == "object":
            return False
    return True