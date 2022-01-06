import os

import pytest
import pandas as pd


@pytest.fixture
def small_df():
    filepath = "test/databalanceanalysis/"
    return pd.read_csv(os.path.join(os.getcwd(), filepath + "test_df.csv"))
