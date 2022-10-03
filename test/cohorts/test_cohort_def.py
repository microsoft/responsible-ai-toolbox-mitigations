import pytest
import pandas as pd
from raimitigations.cohort.cohort_definition import CohortDefinition


# -----------------------------------
def test_cohort_def():
    pass

# -----------------------------------
def test_cohort_def_err():
    cond_err =  [
        [ ['num_0', '>', '0.0'], 'xor', ['num_3', '==', '0.0']  ],
        [ [ ['num_0', '>', 0.0], 'or', ['num_3', '==', 0.0] ], ['num_2', '<', 0.0] ],
        [ ['num_0', '>', '0.0'], 10, ['num_3', '==', '0.0']  ],
        [ [ ['num_0', '>', '0.0'], 'xor', ['num_3', '==', '0.0'] ], 'and' ],
        [ ['num_0', '>', '0.0'], 'xor', ['num_3', '==', '0.0'], 10 ],
        [ ['num_0', 'r', [0.0, 1.1]] ],
        [ ['num_0', 'range', []] ],
        [ ['num_0', '>', [0.0, 1.1]] ],
        [ ['num_0', 'range', [0.0, 1.1, 2.0]] ],
        [ ['num_0', 'range', 1.1] ],
        [ ['num_0', 'range', {}] ],
    ]
    for cond in cond_err:
        with pytest.raises(Exception):
            _ = CohortDefinition(cond)

