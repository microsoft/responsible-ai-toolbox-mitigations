import warnings
import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from raimitigations.utils import (
    split_data,
    get_metrics,
    train_model_fetch_results,
    train_model_plot_results,
)
from raimitigations.dataprocessing import (
    EncoderOHE,
    BasicImputer,
    DataStandardScaler,
    CatBoostSelection,
)


# -----------------------------------
def test_model_utils_bin(df_full, label_col_name):

    plt.switch_backend("Agg")
    warnings.filterwarnings("ignore", "Matplotlib is currently using agg")

    pipe = Pipeline([
        ("imputer", BasicImputer()),
        ("encoder", EncoderOHE()),
        ("std", DataStandardScaler()),
        ("feat_sel", CatBoostSelection(steps=3)),
    ])

    train_x, test_x, train_y, test_y = split_data(df_full, label_col_name, test_size=0.2)
    pipe.fit(train_x, train_y)
    train_x = pipe.transform(train_x)
    test_x = pipe.transform(test_x)

    model = DecisionTreeClassifier()
    model.fit(train_x, train_y)
    pred = model.predict_proba(test_x)

    _ = train_model_fetch_results(train_x, train_y, test_x, test_y, best_th_auc=True)
    _ = train_model_fetch_results(train_x, train_y, test_x, test_y, model="xgb", best_th_auc=False)
    _ = get_metrics(test_y, pred, fixed_th=0.6)
    pred = list(pred[:,1])
    _ = get_metrics(test_y, pred)
    pred = pd.Series(pred)
    _ = get_metrics(test_y, pred)
    with pytest.raises(Exception):
        _ = get_metrics(test_y, 10)


    _ = split_data(df_full, label_col_name, full_df=True, regression=True)

    _ = train_model_plot_results(
        train_x, train_y, test_x, test_y, model="log", train_result=True, plot_pr=True, best_th_auc=False
    )
    _ = train_model_plot_results(
        train_x, train_y, test_x, test_y, model="knn", train_result=False, plot_pr=False, best_th_auc=False
    )


# -----------------------------------
def test_model_utils_multi(df_multiclass, label_col_name):

    pipe = Pipeline([
        ("encoder", EncoderOHE()),
        ("std", DataStandardScaler()),
    ])

    train_x, test_x, train_y, test_y = split_data(df_multiclass, label_col_name, test_size=0.2)
    pipe.fit(train_x, train_y)
    train_x = pipe.transform(train_x)
    test_x = pipe.transform(test_x)

    model = DecisionTreeClassifier()
    _ = train_model_fetch_results(train_x, train_y, test_x, test_y, model=model)


# -----------------------------------
def test_model_utils_reg(df_regression, label_col_name):

    pipe = Pipeline([
        ("encoder", EncoderOHE()),
        ("std", DataStandardScaler()),
    ])

    train_x, test_x, train_y, test_y = split_data(df_regression, label_col_name, test_size=0.2, regression=True)
    pipe.fit(train_x, train_y)
    train_x = pipe.transform(train_x)
    test_x = pipe.transform(test_x)

    model = DecisionTreeRegressor()
    _ = train_model_fetch_results(train_x, train_y, test_x, test_y, model=model, regression=True)