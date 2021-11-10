
import pandas as pd
from errorsmitigation.dataprocessing import DataRebalance

from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
# from sklearn.neighbors import NearestNeighbors

# unit test for the DataRebalance API
def test_data_rebalance():
    # load the training dataset
    data_dir = 'datasets/test/hr_promotion'
    seed =42

    dataset =  pd.read_csv(data_dir + '/train.csv').drop(['employee_id'], axis=1)
  
    # handle duplicates
    dataset = dataset.drop_duplicates()

    # handle null values
    dataset.dropna(axis=0, inplace=True)
    
    # OneHotEncoder for categorical features
    dataset = pd.get_dummies(dataset, drop_first=False)

    tomek = TomekLinks(sampling_strategy='auto')
    smote = SMOTE(sampling_strategy='auto', random_state= seed)
    smote_tomek = SMOTETomek(sampling_strategy='auto', random_state=seed)

    # nn_k = NearestNeighbors(n_neighbors=3)
    # smote = SMOTE(random_state=seed, k_neighbors=nn_k)
    #      dataset
    #      target
    #      sampling_strategy='auto'
    #      random_state=None
    #      smote_tomek=None
    #      smote=None
    #      tomek=None

    target_index = dataset.columns.get_loc('is_promoted')
    
    # over sampling
    data_rebalance =  DataRebalance(dataset, target_index, None, None, None, smote, None)
    X = data_rebalance.Rebalance()
    
    print('')

test_data_rebalance()