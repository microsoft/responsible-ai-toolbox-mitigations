import pandas as pd
from errorsmitigation.dataprocessing import DataSample

# unit test for the DataSample API
def test_data_random_sample():
    # load the dataset
    data_dir = 'datasets/hr_promotion'
    # hr_data =  pd.read_csv(data_dir + '/demo_aug1.csv').drop(['employee_id'], axis=1)
    hr_data =  pd.read_csv(data_dir + '/train.csv')
    
                # dataset
                # target
                # sample_size,
                # categorical_features = True, 
                # drop_null = True,
                # drop_duplicates = False,
                # stratify = False

    target_index = hr_data.columns.get_loc('is_promoted')
    data_sample =  DataSample(hr_data, target_index, 0.2, True, True, True, True)  
    random_sample = data_sample.RandomSample()

    print(random_sample.shape)
    print(random_sample.head())
   
    print('')

test_data_random_sample()
