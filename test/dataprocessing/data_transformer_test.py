from asyncio.windows_events import NULL
import pytest
import sys
import numpy as np
import pickle as pk
import copy
import pandas as pd

from raimitigations.dataprocessing import Transformer

from common_utils import (
    PASS, 
    create_hr_promotion_data, 
    create_hr_promotion_10_data, 
    validate_categorical_trans, 
    verify_data_categorical_columns,
    verify_type_non_categorical_columns
    )


# data_transformer =  Transformer(dataset, target, transformer_type, transform_features, random_state, method, output_distribution)
# Parameters:
#           dataset,
#           target,
#           transformer_type,
#           transform_features = None,
#           random_state = None,
#           method ='yeo-johnson',
#           output_distribution  = 'uniform'

hr_promotion = create_hr_promotion_data()
hr_promotion_10 = create_hr_promotion_10_data()

seed = 42


@pytest.fixture
def target_index_promoted():
    target_index = hr_promotion_10.columns.get_loc("is_promoted")
    return target_index


@pytest.fixture
def target_index_previous_year_rating():
    target_index = hr_promotion_10.columns.get_loc("previous_year_rating")
    return target_index


# unit test for the Transform API


def test_data_transf_allfeatures(target_index_promoted):

    """Data Transformer test with default transform_features. API call: Transformer(hr_data_test, target_index_promoted, Transformer.TransformerType.StandardScaler)"""

    data_transform = Transformer(hr_promotion_10, target_index_promoted, Transformer.TransformerType.StandardScaler)
    result = data_transform.transform()

    categorical_columns = ['department', 'region', 'education', 'gender', 'recruitment_channel']
    non_categorical_columns = ['previous_year_rating', 'length_of_service', 'KPIs_met >80%', 'awards_won?', 'avg_training_score']

    categorical_trans = validate_categorical_trans (hr_promotion_10, result, categorical_columns)
    assert categorical_trans == None, f'Error in result table in column: {categorical_trans}'

    verify_categorical = verify_data_categorical_columns(categorical_columns, result, hr_promotion_10)
    assert verify_categorical  == None, f'Error in result table in column: {verify_categorical[0]} and row: {verify_categorical[1]}'
    
    verify_non_categorical = verify_type_non_categorical_columns(non_categorical_columns, result)
    assert verify_non_categorical == None, f'Error in result table in column: {verify_non_categorical}'


def test_data_transf_allfeatures_PowerTrans(target_index_promoted):

    """Data Transformer test with default transform_features. API call: Transformer(hr_data_test, target_index_promoted, Transformer.TransformerType.PowerTransformer)"""

    data_transform = Transformer(
        hr_promotion_10, 
        target_index_promoted, 
        Transformer.TransformerType.PowerTransformer
    )
    result = data_transform.transform()

    categorical_columns = ['department', 'region', 'education', 'gender', 'recruitment_channel']
    non_categorical_columns = ['previous_year_rating', 'length_of_service', 'KPIs_met >80%', 'awards_won?', 'avg_training_score']

    categorical_trans = validate_categorical_trans (hr_promotion_10, result, categorical_columns)
    assert categorical_trans == None, f'Error in result table in column: {categorical_trans}'

    verify_categorical = verify_data_categorical_columns(categorical_columns, result, hr_promotion_10)
    assert verify_categorical  == None, f'Error in result table in column: {verify_categorical[0]} and row: {verify_categorical[1]}'
    
    verify_non_categorical = verify_type_non_categorical_columns(non_categorical_columns, result)
    assert verify_non_categorical == None, f'Error in result table in column: {verify_non_categorical}'


def test_data_transf_allfeatures_diffTargetIndex(target_index_previous_year_rating):

    """Data Transformer test with default transform_features and previous year rating target index. API call: Transformer(hr_data_test, target_index_previous_year_rating, Transformer.TransformerType.StandardScaler)"""

    data_transform = Transformer(
        hr_promotion_10,
        target_index_previous_year_rating,
        Transformer.TransformerType.StandardScaler
    )
    result = data_transform.transform()

    categorical_columns = ['department', 'region', 'education', 'gender', 'recruitment_channel']
    non_categorical_columns = ['is_promoted', 'length_of_service', 'KPIs_met >80%', 'awards_won?', 'avg_training_score']

    categorical_trans = validate_categorical_trans (hr_promotion_10, result, categorical_columns)
    assert categorical_trans == None, f'Error in result table in column: {categorical_trans}'

    verify_categorical = verify_data_categorical_columns(categorical_columns, result, hr_promotion_10)
    assert verify_categorical  == None, f'Error in result table in column: {verify_categorical[0]} and row: {verify_categorical[1]}'
    
    verify_non_categorical = verify_type_non_categorical_columns(non_categorical_columns, result)
    assert verify_non_categorical == None, f'Error in result table in column: {verify_non_categorical}'
    


def test_data_transf_StandardScalerFeat_None(target_index_promoted):

    """Data Transformer test with all transform_features. API call: Transformer(hr_data_test, target_index_promoted, Transformer.TransformerType.StandardScaler, None, pytest.seed)"""

    transform_features = None

    data_transform = Transformer(
        hr_promotion_10, 
        target_index_promoted, 
        Transformer.TransformerType.StandardScaler, 
        transform_features, 
        seed
    )
    result = data_transform.transform()

    categorical_columns = [
        'department', 
        'region', 'education', 
        'gender', 
        'recruitment_channel'
    ]
    non_categorical_columns = [
        'previous_year_rating', 
        'length_of_service', 
        'KPIs_met >80%', 
        'awards_won?', 
        'avg_training_score'
    ]

    categorical_trans = validate_categorical_trans (hr_promotion_10, result, categorical_columns)
    assert categorical_trans == None, f'Error in result table in column: {categorical_trans}'

    verify_categorical = verify_data_categorical_columns(categorical_columns, result, hr_promotion_10)
    assert verify_categorical  == None, f'Error in result table in column: {verify_categorical[0]} and row: {verify_categorical[1]}'
    
    verify_non_categorical = verify_type_non_categorical_columns(non_categorical_columns, result)
    assert verify_non_categorical == None, f'Error in result table in column: {verify_non_categorical}'


def test_data_transf_StandardScaler(target_index_promoted):

    """Data Transformer test with subset of transform_features. API call: Transformer(hr_data_test, target_index_promoted, Transformer.TransformerType.StandardScaler, ['age', 'previous_year_rating', 'length_of_service', 'avg_training_score'], pytest.seed)"""

    transform_features = [
        "age",
        "previous_year_rating",
        "length_of_service",
        "avg_training_score",
    ]

    data_transform = Transformer(
        hr_promotion_10,
        target_index_promoted,
        Transformer.TransformerType.StandardScaler,
        transform_features,
        seed
    )
    result = data_transform.transform()

    verify_non_categorical = verify_type_non_categorical_columns(transform_features, result)
    assert verify_non_categorical == None, f'Error in result table in column: {verify_non_categorical}'


def test_data_transf_StandardScaler_OneFeature_DiffTargetIndex(
    target_index_previous_year_rating
):

    """Data Transformer test with subset of transform_features. API call: Transformer(hr_data_test, target_index_promoted, Transformer.TransformerType.StandardScaler, 'age', pytest.seed)"""

    transform_features = ["age"]

    data_transform = Transformer(
        hr_promotion_10,
        target_index_previous_year_rating,
        Transformer.TransformerType.StandardScaler,
        transform_features,
        seed
    )
    result = data_transform.transform()

    verify_non_categorical = verify_type_non_categorical_columns(transform_features, result)
    assert verify_non_categorical == None, f'Error in result table in column: {verify_non_categorical}'


def test_data_transf_MinMaxScaler(target_index_promoted):

    """Data Transformer test with MinMaxScaler transformer type. API call: Transformer(hr_data_test, target_index_promoted, Transformer.TransformerType.MinMaxScaler, ['age', 'previous_year_rating', 'length_of_service', 'avg_training_score'], pytest.seed)"""

    transform_features = [
        "age",
        "previous_year_rating",
        "length_of_service",
        "avg_training_score",
    ]

    data_transform = Transformer(
        hr_promotion_10,
        target_index_promoted,
        Transformer.TransformerType.MinMaxScaler,
        transform_features,
        seed
    )
    result = data_transform.transform()

    verify_non_categorical = verify_type_non_categorical_columns(transform_features, result)
    assert verify_non_categorical == None, f'Error in result table in column: {verify_non_categorical}'


def test_data_transf_MinMaxScaler_Feat_None(target_index_promoted):

    """Data Transformer test with MinMaxScaler transformer type with all transform features. API call: Transformer(hr_data_test, target_index_promoted, Transformer.TransformerType.MinMaxScaler, None, pytest.seed)"""

    transform_features = None
    seed = 42

    data_transform = Transformer(
        hr_promotion_10,
        target_index_promoted,
        Transformer.TransformerType.MinMaxScaler,
        transform_features,
        seed
    )
    result = data_transform.transform()

    categorical_columns = [
        'department', 
        'region', 
        'education', 
        'gender', 
        'recruitment_channel'
    ]
    non_categorical_columns = [
        'previous_year_rating', 
        'length_of_service', 
        'KPIs_met >80%', 
        'awards_won?', 
        'avg_training_score'
    ]

    categorical_trans = validate_categorical_trans (hr_promotion_10, result, categorical_columns)
    assert categorical_trans == None, f'Error in result table in column: {categorical_trans}'

    verify_categorical = verify_data_categorical_columns(categorical_columns, result, hr_promotion_10)
    assert verify_categorical  == None, f'Error in result table in column: {verify_categorical[0]} and row: {verify_categorical[1]}'
    
    verify_non_categorical = verify_type_non_categorical_columns(non_categorical_columns, result)
    assert verify_non_categorical == None, f'Error in result table in column: {verify_non_categorical}'


def test_data_transf_Robust(target_index_promoted):

    """Data Transformer test with RobustScaler transformer type with all transform features. API call: Transformer(hr_data_test, target_index_promoted, Transformer.TransformerType.RobustScaler, ['age', 'previous_year_rating', 'length_of_service', 'avg_training_score'], pytest.seed)"""

    transform_features = [
        "age",
        "previous_year_rating",
        "length_of_service",
        "avg_training_score"
    ]

    data_transform = Transformer(
        hr_promotion_10,
        target_index_promoted,
        Transformer.TransformerType.RobustScaler,
        transform_features,
        seed 
    )
    result = data_transform.transform()

    verify_non_categorical = verify_type_non_categorical_columns(transform_features, result)
    assert verify_non_categorical == None, f'Error in result table in column: {verify_non_categorical}'


def test_data_transf_Power(target_index_promoted):

    """Data Transformer test with Power transformer type with non-categorical transform features. API call: Transformer(hr_data_test, target_index_promoted, Transformer.TransformerType.PowerTransformer, ['age', 'previous_year_rating', 'length_of_service', 'avg_training_score'], pytest.seed)"""

    transform_features = [
        "age",
        "previous_year_rating",
        "length_of_service",
        "avg_training_score"
    ]

    data_transform = Transformer(
        hr_promotion_10,
        target_index_promoted,
        Transformer.TransformerType.PowerTransformer,
        transform_features,
        seed
    )
    result = data_transform.transform()

    verify_non_categorical = verify_type_non_categorical_columns(transform_features, result)
    assert verify_non_categorical == None, f'Error in result table in column: {verify_non_categorical}'


def test_data_transf_Quantile(target_index_promoted):

    """Data Transformer test with Quantile transformer type with non-categorical transform features. API call: Transformer(hr_data_test, target_index_promoted, Transformer.TransformerType.QuantileTransformer, ['age', 'previous_year_rating', 'length_of_service', 'avg_training_score'], pytest.seed)"""

    transform_features = [
        "age",
        "previous_year_rating",
        "length_of_service",
    ]

    data_transform = Transformer(
        hr_promotion_10,
        target_index_promoted,
        Transformer.TransformerType.QuantileTransformer,
        transform_features,
        seed
    )
    result = data_transform.transform()

    verify_non_categorical = verify_type_non_categorical_columns(transform_features, result)
    assert verify_non_categorical == None, f'Error in result table in column: {verify_non_categorical}'


def test_data_transf_Normalizer(target_index_promoted):

    """Data Transformer test with Normalizer transformer type with non-categorical transform features. API call: Transformer(hr_data_test, target_index_promoted, Transformer.TransformerType.QuantileTransformer, ['age', 'previous_year_rating', 'length_of_service', 'avg_training_score'], pytest.seed)"""

    transform_features = [
        "department",
        "gender",
        "age",
        "length_of_service"
    ]

    data_transform = Transformer(
        hr_promotion_10,
        target_index_promoted,
        Transformer.TransformerType.Normalizer,
        transform_features,
        seed
    )
    result = data_transform.transform()

    categorical_columns = [
        'department', 
        'gender'
    ]
    non_categorical_columns = [
        'age', 
        'length_of_service'
    ]

    categorical_trans = validate_categorical_trans (hr_promotion_10, result, categorical_columns)
    assert categorical_trans == None, f'Error in result table in column: {categorical_trans}'

    verify_categorical = verify_data_categorical_columns(categorical_columns, result, hr_promotion_10)
    assert verify_categorical  == None, f'Error in result table in column: {verify_categorical[0]} and row: {verify_categorical[1]}'
    
    verify_non_categorical = verify_type_non_categorical_columns(non_categorical_columns, result)
    assert verify_non_categorical == None, f'Error in result table in column: {verify_non_categorical}'


def test_data_transf_method_boxcox(target_index_promoted):

    """Data Transformer test with StandardScaler transformer type with non-categorical transform features and box-cox method. API call: Transformer.TransformerType.StandardScaler, ['age', 'previous_year_rating', 'length_of_service', 'avg_training_score'], pytest.seed, method='box-cox')"""

    transform_features = [
        "region",
        "education",
        "avg_training_score"
    ]

    data_transform = Transformer(
        hr_promotion_10,
        target_index_promoted,
        Transformer.TransformerType.QuantileTransformer,
        transform_features,
        seed,
        method="box-cox"
    )
    result = data_transform.transform()

    categorical_columns = ['region', 'education']
    non_categorical_columns = ['avg_training_score']

    categorical_trans = validate_categorical_trans (hr_promotion_10, result, categorical_columns)
    assert categorical_trans == None, f'Error in result table in column: {categorical_trans}'

    verify_categorical = verify_data_categorical_columns(categorical_columns, result, hr_promotion_10)
    assert verify_categorical  == None, f'Error in result table in column: {verify_categorical[0]} and row: {verify_categorical[1]}'
    
    verify_non_categorical = verify_type_non_categorical_columns(non_categorical_columns, result)
    assert verify_non_categorical == None, f'Error in result table in column: {verify_non_categorical}'


def test_data_transf_output_distribution_Normal(target_index_promoted):

    """Data Transformer test with StandardScaler transformer type with non-categorical transform features and box-cox method. API call: Transformer.TransformerType.StandardScaler, ['length_of_service', 'avg_training_score'], pytest.seed, method='yeo-johnson', output_distribution='normal')"""

    transform_features = [
        "length_of_service", 
        "avg_training_score"
    ]

    data_transform = Transformer(
        hr_promotion_10,
        target_index_promoted,
        Transformer.TransformerType.MinMaxScaler,
        transform_features,
        seed,
        method="yeo-johnson",
        output_distribution="normal",
    )
    result = data_transform.transform()

    verify_non_categorical = verify_type_non_categorical_columns(transform_features, result)
    assert verify_non_categorical == None, f'Error in result table in column: {verify_non_categorical}'


def test_data_transf_output_distribution_Uniform(target_index_promoted):

    """Data Transformer test with StandardScaler transformer type with non-categorical transform features and box-cox method. API call: Transformer.TransformerType.QuantileTransformer, ['length_of_service', 'avg_training_score'], pytest.seed, method='yeo-johnson', output_distribution='uniform')"""

    transform_features = [
        "education", 
        "recruitment_channel", 
        "KPIs_met >80%", 
        "awards_won?"
    ]

    data_transform = Transformer(
        hr_promotion_10,
        target_index_promoted,
        Transformer.TransformerType.QuantileTransformer,
        transform_features,
        seed,
        method="box-cox",
        output_distribution="uniform"
    )
    result = data_transform.transform()

    categorical_columns = ['education', 'recruitment_channel']
    non_categorical_columns = ['KPIs_met >80%','awards_won?']

    categorical_trans = validate_categorical_trans (hr_promotion_10, result, categorical_columns)
    assert categorical_trans == None, f'Error in result table in column: {categorical_trans}'

    verify_categorical = verify_data_categorical_columns(categorical_columns, result, hr_promotion_10)
    assert verify_categorical  == None, f'Error in result table in column: {verify_categorical[0]} and row: {verify_categorical[1]}'
    
    verify_non_categorical = verify_type_non_categorical_columns(non_categorical_columns, result)
    assert verify_non_categorical == None, f'Error in result table in column: {verify_non_categorical}'
