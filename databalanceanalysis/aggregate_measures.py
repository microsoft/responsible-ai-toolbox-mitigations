from databalanceanalysis.constants import (
    aggregate_measures_to_func,
    aggregate_balance_measures,
)
from databalanceanalysis.constants import Measures


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
    def aggregate_measures(self):
        return self._aggregate_measures
