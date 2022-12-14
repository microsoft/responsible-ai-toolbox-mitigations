import os
import pathlib
import collections
import pytest
import numpy as np
import pandas as pd
from raimitigations.cohort.cohort_definition import CohortDefinition


# -----------------------------------
def test_cohort_def():
    json_file = "single_cohort.json"

    df = pd.DataFrame({
        "race":     ['elf', 'orc', 'halfling', 'human', 'halfling', 'orc', 'elf', 'orc', 'human', 'orc'],
        "height(m)":[1.6,   1.95,  1.40,       1.75,     1.53,      2.10,   1.85,  1.79,  1.65,   np.nan],
        "past_score":[85,   59,    19,          89,      91,        79,      45,   82,    47,     87  ],
        "score":    [90,    43,    29,          99,      85,        73,      58,   94,    37,     51]
    })

    list_conditions = []
    conditions = [
                    [ ['race', '==', 'elf'], 'or', ['race', '==', 'orc'] ],
                    'and',
                    ['height(m)', '>=', 1.8]
                ]
    list_conditions.append(conditions)

    conditions = [ [ ['race', '==', ['elf', 'orc'] ] ] ]
    list_conditions.append(conditions)

    conditions = [ [ ['race', '!=', ['human', 'halfling'] ] ] ]
    list_conditions.append(conditions)

    conditions = [ [ ['race', '!=', ['human', 'halfling'] ] ] ]
    list_conditions.append(conditions)

    conditions = [ ['height(m)', '==', np.nan] ]
    list_conditions.append(conditions)

    conditions = [ ['height(m)', '==', [1.95, np.nan]] ]
    list_conditions.append(conditions)

    conditions = [ ['height(m)', '!=', [1.95, np.nan]] ]
    list_conditions.append(conditions)

    conditions = [ [ ['height(m)', 'range', [1.1, 1.7]], 'and', ['race', '!=', 'halfling'] ] ]
    list_conditions.append(conditions)

    conditions = [  ['height(m)', '>', 1.5],
                    'and',
                    ['height(m)', '<', 1.99],
                    'and',
                    ['score', '<=', 70]
                ]
    list_conditions.append(conditions)

    conditions = [ ['score', '<=', 'past_score'] ]
    list_conditions.append(conditions)

    for condition in list_conditions:
        cht_def = CohortDefinition(condition)
        cht_def.save(json_file)
        subset1 = cht_def.get_cohort_subset(df)
        new_cht = CohortDefinition(json_file)
        subset2 = new_cht.get_cohort_subset(df)
        assert collections.Counter(list(subset1.index)) == collections.Counter(list(subset2.index)), (
            "ERROR: the subsets encountered by the original cohort and the loaded cohort are different."
        )

    if os.path.exists(json_file):
        os.remove(json_file)

    current_path = pathlib.Path(__file__).parent.absolute()
    json_fld = f"{current_path}/json_files_test"
    json_files = [f"{json_fld}/cht_0.json", f"{json_fld}/cht_1.json"]
    for file_name in json_files:
        _ = CohortDefinition(file_name)

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
        [ ['num_0', 'range', [0.0, np.nan]] ],
    ]
    for cond in cond_err:
        with pytest.raises(Exception):
            _ = CohortDefinition(cond)

    with pytest.raises(Exception):
        cht = CohortDefinition(None)
        _ = cht.get_cohort_subset(df_full)

    cht = CohortDefinition(None)
    with pytest.raises(Exception):
        cht.save("test.json")

    cht = CohortDefinition([ ['num_10', '>', 0.0] ])
    with pytest.raises(Exception):
        _ = cht.get_cohort_subset(df_full)

    current_path = pathlib.Path(__file__).parent.absolute()
    json_fld = f"{current_path}/json_files_test"
    json_files = [f"{json_fld}/cht_err_1.json", f"{json_fld}/cht_err_2.json"]
    for file_name in json_files:
        with pytest.raises(Exception):
            _ = CohortDefinition(file_name)

