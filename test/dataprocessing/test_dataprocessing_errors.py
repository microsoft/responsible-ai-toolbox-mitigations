import pytest
import numpy as np

import raimitigations.dataprocessing as dp
from raimitigations.cohort import CohortManager, DecoupledClass
from raimitigations.cohort.decoupled_class.decoupled_cohort import _DecoupledCohort



# -----------------------------------
def test_dataprocessing_errors(df_breast_cancer, label_name_bc):
    df = df_breast_cancer

    # ERROR: Expected 'estimator' to be a SKLearn classifier or regressor
    dec = DecoupledClass(cohort_col=["A"])
    dec.estimator = np.array([1])
    with pytest.raises(Exception):
        dec._set_estimator()

    # ERROR: calling the _get_df_subset method with an invalid col_list parameter
    encoder = dp.EncoderOrdinal(verbose=False)
    encoder.fit(df)
    with pytest.raises(Exception):
        encoder._get_df_subset(df, col_list="a")