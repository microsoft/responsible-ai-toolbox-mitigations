
import pandas as pd
from errorsmitigation.dataprocessing import DataSplit, DataTransformer, DataRebalance, DataSample

# unit test for the DataSplit API
def test_data_split():
    # load the training dataset
    data_dir = 'datasets/hr_promotion'
    # hr_data =  pd.read_csv(data_dir + '/demo_aug1.csv').drop(['employee_id'], axis=1)
    hr_data =  pd.read_csv(data_dir + '/demo_aug1.csv')
    seed = 42

    target_index = hr_data.columns.get_loc('is_promoted')
    data_sample =  DataSample(hr_data, target_index, 0.5, True)

    random_sample = data_sample.RandomSample()
    print(random_sample.shape)


    data_split =  DataSplit(random_sample, 'is_promoted', 0.9, 42, True, True, True, False)
    train_data, test_data = data_split.Split()    
    
    # data_split =  DataSplit(hr_data, 'is_promoted', 0.9, 42, True, True, True, False)
    # train_data, test_data = data_split.Split()  


    # data_rebalance =  DataRebalance(train_data, 0, 'auto', seed, None, smote)
    # data_train_rebalance =  DataRebalance(train_data, 'is_promoted', 'auto', seed)
    # X = data_train_rebalance.Rebalance()
    
    # # print('')
    # data_transform =  DataTransformer(X, 'is_promoted',DataTransformer.TransformerType.Normalizer, 42)    
    # X = data_transform.Transform()

    
    print(X.shape)
test_data_split()

