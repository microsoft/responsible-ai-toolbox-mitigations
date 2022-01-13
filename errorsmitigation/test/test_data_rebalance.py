
import pandas as pd
from errorsmitigation.dataprocessing import DataRebalance

from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
# from sklearn.neighbors import NearestNeighbors



def undummify(df, prefix_sep="_", col_list=None):
    if col_list == None:
        cols_to_collapse = {
            item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
        }
        series_list = []
        for col, needs_to_collapse in cols_to_collapse.items():
            if needs_to_collapse:
                undummified = (
                    df.filter(like=col)
                    .idxmax(axis=1)
                    .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                    .rename(col)
                )
                series_list.append(undummified)
            else:
                series_list.append(df[col])
        undummified_df = pd.concat(series_list, axis=1)
    else:
        series_list = []
        collapse_df = df
        for col in col_list:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
            collapse_df = collapse_df.loc[:, ~df.columns.str.startswith(col)]
        undummified_df = pd.concat(series_list, axis=1)
        undummified_df = pd.concat([undummified_df, collapse_df], axis=1)
    return undummified_df


# unit test for the DataRebalance API
def test_data_rebalance():
    # load the training dataset
    data_dir = 'datasets/test/hr_promotion'
    seed = 42

    dataset =  pd.read_csv(data_dir + '/train.csv'); #.drop(['employee_id'], axis=1)
  
    # # handle duplicates
    # dataset = dataset.drop_duplicates()

    # handle null values
    dataset.dropna(axis=0, inplace=True)
    
    # OneHotEncoder for categorical features
    # dataset = pd.get_dummies(dataset, drop_first=False) 
    dummy_df = pd.get_dummies(dataset, prefix_sep = "-")
    gender_df = undummify(dummy_df, prefix_sep = "-", col_list = ['gender'])

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

    # target_index = dataset.columns.get_loc('no_of_trainings')
    
        
    data_balance_smote =  DataRebalance(gender_df, 'gender', 'auto', 42, None, smote, None)
    smote_df = data_balance_smote.Rebalance()
    print(smote_df.shape)


    # over sampling
    # data_rebalance =  DataRebalance(dataset, target_index, 'auto', seed, None, smote, None)
    # X = data_rebalance.Rebalance()
    
    print('')

test_data_rebalance()