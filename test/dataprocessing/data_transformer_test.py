import pytest
import sys
import numpy as np
import pickle as pk
import copy
import pandas as pd

sys.path.append("../../../responsible-ai-mitigations")
from raimitigations.dataprocessing import Transformer


# data_transformer =  Transformer(dataset, target, transformer_type, transform_features, random_state, method, output_distribution)
# Parameters:
#           dataset,
#           target,
#           transformer_type,
#           transform_features = None,
#           random_state = None,
#           method ='yeo-johnson',
#           output_distribution  = 'uniform'

pytest.hr_promotion_TEST = pd.read_csv(
    "test/datasets/hr_promotion_test" + "/train.csv"
).drop(["employee_id"], axis=1)
pytest.seed = 42


@pytest.fixture
def hr_data_test():
    hr_data = copy.deepcopy(pytest.hr_promotion_TEST)
    return hr_data


@pytest.fixture
def target_index_promoted():
    target_index = pytest.hr_promotion_TEST.columns.get_loc("is_promoted")
    return target_index


@pytest.fixture
def target_index_previous_year_rating():
    target_index = pytest.hr_promotion_TEST.columns.get_loc("previous_year_rating")
    return target_index


# unit test for the Transform API


@pytest.mark.Functional
def test_data_transf_allfeatures(hr_data_test, target_index_promoted):

    """Data Transformer test with default transform_features. API call: Transformer(hr_data_test, target_index_promoted, Transformer.TransformerType.StandardScaler)"""

    data_transform = Transformer(
        hr_data_test, target_index_promoted, Transformer.TransformerType.StandardScaler,
    )
    result = data_transform.transform()

    expected = pd.DataFrame(
        pd.read_pickle("test/datasets/transfer" + "/transfAll.pickle")
    )

    assert result.equals(expected)


@pytest.mark.Functional
def test_data_transf_allfeatures_diffTargetIndex(
    hr_data_test, target_index_previous_year_rating
):

    """Data Transformer test with default transform_features and previous year rating target index. API call: Transformer(hr_data_test, target_index_previous_year_rating, Transformer.TransformerType.StandardScaler)"""

    data_transform = Transformer(
        hr_data_test,
        target_index_previous_year_rating,
        Transformer.TransformerType.StandardScaler,
    )
    result = data_transform.transform()

    expected = pd.DataFrame(
        pd.read_pickle("test/datasets/transfer" + "/transfAllDiffIndex.pickle")
    )

    assert result.equals(expected)


@pytest.mark.Functional
def test_data_transf_StandardScalerFeat_None(hr_data_test, target_index_promoted):

    """Data Transformer test with all transform_features. API call: Transformer(hr_data_test, target_index_promoted, Transformer.TransformerType.StandardScaler, None, pytest.seed)"""

    transform_features = None

    data_transform = Transformer(
        hr_data_test,
        target_index_promoted,
        Transformer.TransformerType.StandardScaler,
        transform_features,
        pytest.seed,
    )
    result = data_transform.transform()

    expected = pd.DataFrame(
        pd.read_pickle("test/datasets/transfer" + "/transfAll.pickle")
    )

    assert result.equals(expected)


@pytest.mark.Functional
def test_data_transf_StandardScaler(hr_data_test, target_index_promoted):

    """Data Transformer test with subset of transform_features. API call: Transformer(hr_data_test, target_index_promoted, Transformer.TransformerType.StandardScaler, ['age', 'previous_year_rating', 'length_of_service', 'avg_training_score'], pytest.seed)"""

    transform_features = [
        "age",
        "previous_year_rating",
        "length_of_service",
        "avg_training_score",
    ]

    data_transform = Transformer(
        hr_data_test,
        target_index_promoted,
        Transformer.TransformerType.StandardScaler,
        transform_features,
        pytest.seed,
    )
    result = data_transform.transform()

    expected = pd.DataFrame(
        pd.read_pickle("test/datasets/transfer" + "/transfStandardScaler.pickle")
    )

    assert result.equals(expected)


@pytest.mark.Functional
def test_data_transf_StandardScaler_OneFeature_DiffTargetIndex(
    hr_data_test, target_index_previous_year_rating
):

    """Data Transformer test with subset of transform_features. API call: Transformer(hr_data_test, target_index_promoted, Transformer.TransformerType.StandardScaler, 'age', pytest.seed)"""

    transform_features = ["age"]

    data_transform = Transformer(
        hr_data_test,
        target_index_previous_year_rating,
        Transformer.TransformerType.StandardScaler,
        transform_features,
        pytest.seed,
    )
    result = data_transform.transform()

    expected = pd.DataFrame(
        pd.read_pickle(
            "test/datasets/transfer" + "/transfStandardScalerDiffIndexAge.pickle"
        )
    )

    assert result.equals(expected)


@pytest.mark.Functional
def test_data_transf_MinMaxScaler(hr_data_test, target_index_promoted):

    """Data Transformer test with MinMaxScaler transformer type. API call: Transformer(hr_data_test, target_index_promoted, Transformer.TransformerType.MinMaxScaler, ['age', 'previous_year_rating', 'length_of_service', 'avg_training_score'], pytest.seed)"""

    transform_features = [
        "age",
        "previous_year_rating",
        "length_of_service",
        "avg_training_score",
    ]

    data_transform = Transformer(
        hr_data_test,
        target_index_promoted,
        Transformer.TransformerType.MinMaxScaler,
        transform_features,
        pytest.seed,
    )
    result = data_transform.transform()

    expected = pd.DataFrame(
        pd.read_pickle("test/datasets/transfer" + "/transfMinMax.pickle")
    )

    assert result.equals(expected)


@pytest.mark.Functional
def test_data_transf_MinMaxScaler_Feat_None(hr_data_test, target_index_promoted):

    """Data Transformer test with MinMaxScaler transformer type with all transform features. API call: Transformer(hr_data_test, target_index_promoted, Transformer.TransformerType.MinMaxScaler, None, pytest.seed)"""

    transform_features = None

    data_transform = Transformer(
        hr_data_test,
        target_index_promoted,
        Transformer.TransformerType.MinMaxScaler,
        transform_features,
        pytest.seed,
    )
    result = data_transform.transform()

    expected = pd.DataFrame(
        pd.read_pickle("test/datasets/transfer" + "/transfMinMaxNone.pickle")
    )

    assert result.equals(expected)


@pytest.mark.Functional
def test_data_transf_Robust(hr_data_test, target_index_promoted):

    """Data Transformer test with RobustScaler transformer type with all transform features. API call: Transformer(hr_data_test, target_index_promoted, Transformer.TransformerType.RobustScaler, ['age', 'previous_year_rating', 'length_of_service', 'avg_training_score'], pytest.seed)"""

    transform_features = [
        "age",
        "previous_year_rating",
        "length_of_service",
        "avg_training_score",
    ]

    data_transform2 = Transformer(
        hr_data_test,
        target_index_promoted,
        Transformer.TransformerType.RobustScaler,
        transform_features,
        pytest.seed,
    )
    result2 = data_transform2.transform()

    expected = pd.DataFrame(
        pd.read_pickle("test/datasets/transfer" + "/transfRobust.pickle")
    )

    assert result2.equals(expected)


@pytest.mark.Functional
def test_data_transf_Power(hr_data_test, target_index_promoted):

    """Data Transformer test with Power transformer type with non-categorical transform features. API call: Transformer(hr_data_test, target_index_promoted, Transformer.TransformerType.PowerTransformer, ['age', 'previous_year_rating', 'length_of_service', 'avg_training_score'], pytest.seed)"""

    transform_features = [
        "age",
        "previous_year_rating",
        "length_of_service",
        "avg_training_score",
    ]

    data_transform2 = Transformer(
        hr_data_test,
        target_index_promoted,
        Transformer.TransformerType.PowerTransformer,
        transform_features,
        pytest.seed,
    )
    result2 = data_transform2.transform()

    expected = pd.DataFrame(
        pd.read_pickle("test/datasets/transfer" + "/transfPower.pickle")
    )

    assert result2.equals(expected)


@pytest.mark.Functional
def test_data_transf_Quantile(hr_data_test, target_index_promoted):

    """Data Transformer test with Quantile transformer type with non-categorical transform features. API call: Transformer(hr_data_test, target_index_promoted, Transformer.TransformerType.QuantileTransformer, ['age', 'previous_year_rating', 'length_of_service', 'avg_training_score'], pytest.seed)"""

    transform_features = [
        "age",
        "previous_year_rating",
        "length_of_service",
        "avg_training_score",
    ]

    data_transform2 = Transformer(
        hr_data_test,
        target_index_promoted,
        Transformer.TransformerType.QuantileTransformer,
        transform_features,
        pytest.seed,
    )
    result2 = data_transform2.transform()

    expected = pd.DataFrame(
        pd.read_pickle("test/datasets/transfer" + "/transfQuantile.pickle")
    )

    assert result2.equals(expected)


@pytest.mark.Functional
def test_data_transf_Normalizer(hr_data_test, target_index_promoted):

    """Data Transformer test with Normalizer transformer type with non-categorical transform features. API call: Transformer(hr_data_test, target_index_promoted, Transformer.TransformerType.QuantileTransformer, ['age', 'previous_year_rating', 'length_of_service', 'avg_training_score'], pytest.seed)"""

    transform_features = [
        "age",
        "previous_year_rating",
        "length_of_service",
        "avg_training_score",
    ]

    data_transform2 = Transformer(
        hr_data_test,
        target_index_promoted,
        Transformer.TransformerType.Normalizer,
        transform_features,
        pytest.seed,
    )
    result2 = data_transform2.transform()

    expected = pd.DataFrame(
        pd.read_pickle("test/datasets/transfer" + "/transfNormalizer.pickle")
    )

    assert result2.equals(expected)


@pytest.mark.Functional
def test_data_transf_method_boxcox(hr_data_test, target_index_promoted):

    """Data Transformer test with StandardScaler transformer type with non-categorical transform features and box-cox method. API call: Transformer.TransformerType.StandardScaler, ['age', 'previous_year_rating', 'length_of_service', 'avg_training_score'], pytest.seed, method='box-cox')"""

    transform_features = [
        "age",
        "previous_year_rating",
        "length_of_service",
        "avg_training_score",
    ]

    data_transform2 = Transformer(
        hr_data_test,
        target_index_promoted,
        Transformer.TransformerType.StandardScaler,
        transform_features,
        pytest.seed,
        method="box-cox",
    )
    result = data_transform2.transform()

    expected = pd.DataFrame(
        pd.read_pickle("test/datasets/transfer" + "/transfboxcox.pickle")
    )

    assert result.equals(expected)


@pytest.mark.Functional
def test_data_transf_method_yeojohnson(hr_data_test, target_index_promoted):

    """Data Transformer test with StandardScaler transformer type with non-categorical transform features and yeo-johnson method. API call: Transformer.TransformerType.StandardScaler, ['age', 'previous_year_rating', 'length_of_service', 'avg_training_score'], pytest.seed, method='yeo-johnson')"""

    transform_features = [
        "age",
        "previous_year_rating",
        "length_of_service",
        "avg_training_score",
    ]

    data_transform = Transformer(
        hr_data_test,
        target_index_promoted,
        Transformer.TransformerType.StandardScaler,
        transform_features,
        pytest.seed,
        method="yeo-johnson",
    )
    result = data_transform.transform()

    expected = pd.DataFrame(
        pd.read_pickle("test/datasets/transfer" + "/transfyeojohnson.pickle")
    )

    assert result.equals(expected)


@pytest.mark.Functional
def test_data_transf_method_subsetFeatures(hr_data_test, target_index_promoted):

    """Data Transformer test with StandardScaler transformer type with non-categorical transform features and box-cox method. API call: Transformer.TransformerType.StandardScaler, ['length_of_service', 'avg_training_score'], pytest.seed, method='box-cox')"""

    transform_features = ["length_of_service", "avg_training_score"]

    data_transform = Transformer(
        hr_data_test,
        target_index_promoted,
        Transformer.TransformerType.StandardScaler,
        transform_features,
        pytest.seed,
        method="box-cox",
    )
    result = data_transform.transform()

    expected = pd.DataFrame(
        pd.read_pickle("test/datasets/transfer" + "/subsetFeatures.pickle")
    )

    assert result.equals(expected)


@pytest.mark.Functional
def test_data_transf_output_distribution_Normal(hr_data_test, target_index_promoted):

    """Data Transformer test with StandardScaler transformer type with non-categorical transform features and box-cox method. API call: Transformer.TransformerType.StandardScaler, ['length_of_service', 'avg_training_score'], pytest.seed, method='yeo-johnson', output_distribution='normal')"""

    transform_features = ["length_of_service", "avg_training_score"]

    data_transform = Transformer(
        hr_data_test,
        target_index_promoted,
        Transformer.TransformerType.StandardScaler,
        transform_features,
        pytest.seed,
        method="yeo-johnson",
        output_distribution="normal",
    )
    result = data_transform.transform()

    expected = pd.DataFrame(
        pd.read_pickle("test/datasets/transfer" + "/transfDistribNormal.pickle")
    )

    assert result.equals(expected)


@pytest.mark.Functional
def test_data_transf_output_distribution_Uniform(hr_data_test, target_index_promoted):

    """Data Transformer test with StandardScaler transformer type with non-categorical transform features and box-cox method. API call: Transformer.TransformerType.QuantileTransformer, ['length_of_service', 'avg_training_score'], pytest.seed, method='yeo-johnson', output_distribution='uniform')"""

    transform_features = ["length_of_service", "avg_training_score"]

    data_transform = Transformer(
        hr_data_test,
        target_index_promoted,
        Transformer.TransformerType.QuantileTransformer,
        transform_features,
        pytest.seed,
        method="yeo-johnson",
        output_distribution="uniform",
    )
    result = data_transform.transform()

    expected = pd.DataFrame(
        pd.read_pickle("test/datasets/transfer" + "/transfDistribUniform.pickle")
    )

    # Verifying that QuantileTransformer provides a non-parametric transformation to map the data to a uniform distribution with values between 0 and 1
    assert result.equals(expected)