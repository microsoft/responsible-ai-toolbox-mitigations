from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import warnings

from raimitigations.dataprocessing import (
    split_data,
    EncoderOHE,
    BasicImputer,
    DataStandardScaler,
    CatBoostSelection,
    train_model_fetch_results,
    train_model_plot_results,
)


def test_model_utils(df_full, label_col_name):

    plt.switch_backend("Agg")
    warnings.filterwarnings("ignore", "Matplotlib is currently using agg")

    pipe = Pipeline(
        [
            ("imputer", BasicImputer()),
            ("encoder", EncoderOHE()),
            ("std", DataStandardScaler()),
            ("feat_sel", CatBoostSelection(steps=3)),
        ]
    )

    train_x, test_x, train_y, test_y = split_data(df_full, label_col_name, test_size=0.2)
    pipe.fit(train_x, train_y)
    X = pipe.transform(train_x)
    X_test = pipe.transform(test_x)

    _ = train_model_fetch_results(X, train_y, X_test, test_y, best_th_auc=True)
    _ = train_model_fetch_results(X, train_y, X_test, test_y, model_name="xgb", best_th_auc=False)

    _ = split_data(df_full, label_col_name, full_df=True, regression=True)

    _ = train_model_plot_results(
        X, train_y, X_test, test_y, model_name="log", train_result=True, plot_pr=True, best_th_auc=False
    )
    _ = train_model_plot_results(
        X, train_y, X_test, test_y, model_name="knn", train_result=False, plot_pr=False, best_th_auc=False
    )
