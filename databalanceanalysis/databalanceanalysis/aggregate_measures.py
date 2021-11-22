from databalanceanalysis.databalanceanalysis.constants import (
    aggregate_measures_to_func,
    aggregate_balance_measures,
)

'''
This class computes a set of aggregated balance measures that represents how balanced
the given dataframe is along the given sensitive features.
  
The output is a dictionary that maps the names of the different aggregate measures to their values:
    The following measures are computed:
    - Atkinson Index - https://en.wikipedia.org/wiki/Atkinson_index
    - Theil Index (L and T) - https://en.wikipedia.org/wiki/Theil_index
'''
class AggregateMeasures:
    def __init__(self, df, sensitive_cols):
        self._df = df
        self._sensitive_cols = sensitive_cols
        self._benefits = df.groupby(sensitive_cols).size() / df.shape[0]
        self._aggregate_measures = {}
        for measure in aggregate_balance_measures:
            func = aggregate_measures_to_func[measure]
            self._aggregate_measures[measure] = func(self._benefits)

    @property
    def measures(self):
        return self._aggregate_measures
