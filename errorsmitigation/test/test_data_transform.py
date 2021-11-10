
import pandas as pd
from errorsmitigation.dataprocessing import DataTransformer


# unit test for the DataTransform API
def test_data_transform():
    # load the training dataset
    data_dir = 'datasets/test/hr_promotion'
    seed =42

    hr_data =  pd.read_csv(data_dir + '/train.csv')

    # dataset,
    # target, 
    # transformer_type,
    # transform_features = None,
    # random_state = None,
    # method ='yeo-johnson',
    # output_distribution  = 'uniform' 
    # transform_features = None
    
    # transform_features = ['department', 'region', 'education','gender','recruitment_channel']
    # transform_features = [0,1,2,3,4]

    target_index = hr_data.columns.get_loc('is_promoted')
    data_transform =  DataTransformer(hr_data, target_index, DataTransformer.TransformerType.StandardScaler, None, seed) 
    X = data_transform.Transform()

    print('')

test_data_transform()