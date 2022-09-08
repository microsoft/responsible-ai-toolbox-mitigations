from raimitigations.cohort.cohort_definition import CohortDefinition


conditions = [
                {'age': ['1', '2', '3'], 'can_read': 1},
                {'age': ['90', '91', '92'], 'can_read': 0}
            ]

cht_def = CohortDefinition(conditions)

