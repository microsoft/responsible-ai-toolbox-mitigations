
from aggregate_functions import atkinson_index, thiel_l_index, thiel_t_index

class AggregateMeasures:
    def __init__(self, df, sensitive_cols):
        self._df = df
        self._sensitive_cols = sensitive_cols
        self._benefits = self._df[sensitive_cols]
        self._atkinson_index = atkinson_index(self._benefits, 1)
        self._thiel_t_index = thiel_t_index(self._benefits)
        self._thiel_l_index = thiel_l_index(self._benefits)

    @property
    def atkinson_index(self):
        return self._atkinson_index
    
    @property
    def thiel_l_index(self):
        return self._thiel_l_index
    
    @property
    def thiel_t_index(self):
        return self._thiel_t_index
