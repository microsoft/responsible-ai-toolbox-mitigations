
import pandas as pd
from errorsmitigation.dataprocessing import DataSplit

# unit test for the DataSplit API
def test_data_split():
    # load the training dataset
    data_dir = 'datasets/hr_promotion'
    # hr_data =  pd.read_csv(data_dir + '/train.csv').drop(['employee_id'], axis=1)
    hr_data =  pd.read_csv(data_dir + '/train.csv')
    
#                 dataset
#                 target
#                 train_size
#                 random_state = None
#                 categorical_features = True 
#                 drop_null = True
#                 drop_duplicates = False
#                 is_stratify = False

    print('hr_data, target_index, 0.2, True, True, True, True')
    data_split =  DataSplit(hr_data, 'is_promoted', 0.8, 42, True, True, True, True)
    train_data, test_data = data_split.Split()

    print('')

test_data_split()
