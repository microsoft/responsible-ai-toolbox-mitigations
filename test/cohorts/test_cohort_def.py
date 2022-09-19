import pandas as pd
from raimitigations.cohort.cohort_definition import CohortDefinition

df = pd.DataFrame({
    "race":     ['elf', 'orc', 'halfling', 'human', 'halfling', 'orc', 'elf', 'orc', 'human'],
    "height(m)":[1.6,   1.95,  1.40,       1.75,     1.53,      2.10,   1.85,  1.79,  1.65],
    "score":    [90,    43,    29,          99,      85,        73,      58,   94,    37]
})



conditions = [
                [ ['race', '==', 'elf'], 'or', ['race', '==', 'orc'] ],
                'and',
                ['height(m)', '>=', 1.8]
            ]

cht_def = CohortDefinition(conditions)
temp, _, _ = cht_def.get_cohort_subset(df)
cht_def.save("test.json")
print(temp)

conditions = [ [ ['height(m)', 'range', [1.1, 1.7]], 'and', ['race', '!=', 'halfling'] ] ]

cht_def = CohortDefinition(conditions)
temp, _, _ = cht_def.get_cohort_subset(df)
print(temp)

conditions = [ ['height(m)', '>', 1.5],
              'and',
              ['height(m)', '<', 1.99],
              'and',
              ['score', '<=', 70]
            ]

cht_def = CohortDefinition(conditions)
temp, _, _ = cht_def.get_cohort_subset(df)
print(temp)


try:
    conditions =  [ ['race', '==', 'elf'], 'xor', ['race', '==', 'orc']  ]
    cht_def = CohortDefinition(conditions)
except Exception as e:
    print(e)

try:
    conditions = [ [ ['race', '==', 'elf'], 'or', ['race', '==', 'orc'] ], ['height(m)', '>=', 1.8] ]
    cht_def = CohortDefinition(conditions)
except Exception as e:
    print(e)

try:
    conditions = [ ['race', '==', 'elf'], 10, ['race', '==', 'orc'] ]
    cht_def = CohortDefinition(conditions)
except Exception as e:
    print(e)

try:
    conditions = [ [ ['race', '==', 'elf'], 'or', ['race', '==', 'orc'] ], 'and' ]
    cht_def = CohortDefinition(conditions)
except Exception as e:
    print(e)

# ---------------------------------
try:
    conditions = [ ['height(m)', 'range', [1.1, 1.7], 10] ]
    cht_def = CohortDefinition(conditions)
except Exception as e:
    print(e)

try:
    conditions = [ ['height(m)', 'r', [1.1, 1.7]] ]
    cht_def = CohortDefinition(conditions)
except Exception as e:
    print(e)

try:
    conditions = [ ['height(m)', 'range', []] ]
    cht_def = CohortDefinition(conditions)
except Exception as e:
    print(e)

try:
    conditions = [ ['height(m)', '>', [1.1, 1.7]] ]
    cht_def = CohortDefinition(conditions)
except Exception as e:
    print(e)

try:
    conditions = [ ['height(m)', 'range', [1.1, 1.7, 1]] ]
    cht_def = CohortDefinition(conditions)
except Exception as e:
    print(e)

try:
    conditions = [ ['height(m)', 'range', 1.1] ]
    cht_def = CohortDefinition(conditions)
except Exception as e:
    print(e)

try:
    conditions = [ ['race', '<=', 'elf'] ]
    cht_def = CohortDefinition(conditions)
except Exception as e:
    print(e)

try:
    conditions = [ ['height(m)', 'range', {}] ]
    cht_def = CohortDefinition(conditions)
except Exception as e:
    print(e)

