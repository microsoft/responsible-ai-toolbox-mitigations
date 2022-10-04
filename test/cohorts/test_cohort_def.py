import os
import pytest
import numpy as np
import pandas as pd
from raimitigations.cohort.cohort_definition import CohortDefinition


# -----------------------------------
def test_cohort_def():
    df = pd.DataFrame({
        "race":     ['elf', 'orc', 'halfling', 'human', 'halfling', 'orc', 'elf', 'orc', 'human', 'orc'],
        "height(m)":[1.6,   1.95,  1.40,       1.75,     1.53,      2.10,   1.85,  1.79,  1.65,   np.nan],
        "past_score":[85,   59,    19,          89,      91,        79,      45,   82,    47,     87  ],
        "score":    [90,    43,    29,          99,      85,        73,      58,   94,    37,     51]
    })

    conditions = [
                    [ ['race', '==', 'elf'], 'or', ['race', '==', 'orc'] ],
                    'and',
                    ['height(m)', '>=', 1.8]
                ]
    cht_def = CohortDefinition(conditions)
    _ = cht_def.get_cohort_subset(df)

    conditions = [ [ ['race', '==', ['elf', 'orc'] ] ] ]
    cht_def = CohortDefinition(conditions)
    _ = cht_def.get_cohort_subset(df)

    conditions = [ [ ['race', '!=', ['human', 'halfling'] ] ] ]
    cht_def = CohortDefinition(conditions)
    _ = cht_def.get_cohort_subset(df)

    conditions = [ [ ['race', '!=', ['human', 'halfling'] ] ] ]
    cht_def = CohortDefinition(conditions)
    _ = cht_def.get_cohort_subset(df)

    conditions = [ ['height(m)', '==', np.nan] ]
    cht_def = CohortDefinition(conditions)
    _ = cht_def.get_cohort_subset(df)

    conditions = [ [ ['height(m)', 'range', [1.1, 1.7]], 'and', ['race', '!=', 'halfling'] ] ]
    cht_def = CohortDefinition(conditions)
    _ = cht_def.get_cohort_subset(df)

    conditions = [  ['height(m)', '>', 1.5],
                    'and',
                    ['height(m)', '<', 1.99],
                    'and',
                    ['score', '<=', 70]
                ]
    cht_def = CohortDefinition(conditions)
    _ = cht_def.get_cohort_subset(df)

    conditions = [ ['score', '<=', 'past_score'] ]
    cht_def = CohortDefinition(conditions)
    _ = cht_def.get_cohort_subset(df)
    cht_def.save("single_cohort.json")

    new_cht = CohortDefinition("single_cohort.json")
    _ = new_cht.get_cohort_subset(df)

    if os.path.exists("single_cohort.json"):
        os.remove("single_cohort.json")

# -----------------------------------
def test_cohort_def_err(df_full):
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
        [ ['num_0', '>', 1.0, 2.0] ],
        {},
        [ ['num_0', '>', 0.0], 'or' ],
    ]
    for cond in cond_err:
        with pytest.raises(Exception):
            _ = CohortDefinition(cond)

    with pytest.raises(Exception):
        cht = CohortDefinition(None)
        _ = cht.get_cohort_subset(df_full)

    cht = CohortDefinition([ ['num_10', '>', 0.0] ])
    with pytest.raises(Exception):
        _ = cht.get_cohort_subset(df_full)

