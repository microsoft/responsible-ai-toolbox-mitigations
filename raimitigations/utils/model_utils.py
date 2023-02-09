from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
import xgboost as xgb
import matplotlib.pyplot as plt

from .metric_utils import MetricNames, get_metrics

DECISION_TREE = "tree"
KNN = "knn"
XGBOOST = "xgb"
LOGISTIC = "log"


# -----------------------------------
def split_data(
    df: pd.DataFrame,
    label: str,
    test_size: float = 0.2,
    full_df: bool = False,
    regression: bool = False,
    random_state: int = None,
):
    """
    Splits the dataset given by df into train and test sets.

    :param df: the dataset that will be splitted;
    :param label: the name of the label column;
    :param test_size: a value between [0.0, 1.0] that indicates the size of the test dataset. For example,
        if test_size = 0.2, then 20% of the original dataset will be used as a test set;
    :param full_df: If ``full_df`` is set to True, this function
        returns 2 dataframes: a train and a test dataframe, where both datasets include the label column
        given by the parameter ``label``. Otherwise, 4 values are returned:

            * **train_x:** the train dataset containing all features (all columns except the label column);
            * **test_x:**  the test dataset containing all features (all columns except the label column);
            * **train_y:** the train dataset containing only the label column;
            * **test_y:** the test dataset containing only the label column;

    :param regression: if True, the problem is treated as a regression problem. This way, the split between
        train and test is random, without any stratification. If False, then the problem is treated as a
        classification problem, where the label column is treated as a list of labels. This way, the split
        tries to maintain the same proportion of classes in the train and test sets;
    :param random_state: controls the randomness of how the data is divided.
    :return: if ``full_df`` is set to True, this function returns 2 dataframes: a train and a test dataframe,
        where both datasets include the label column given by the parameter ``label``. Otherwise, 4 values are
        returned:

            * **train_x:** the train dataset containing all features (all columns except the label column);
            * **test_x:**  the test dataset containing all features (all columns except the label column);
            * **train_y:** the train dataset containing only the label column;
            * **test_y:** the test dataset containing only the label column;
    :rtype: tuple
    """
    X = df.drop(columns=[label])
    y = df[label]
    if regression:
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=random_state)
    else:
        train_x, test_x, train_y, test_y = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
    if full_df:
        train_df = train_x
        train_df[label] = train_y
        test_df = test_x
        test_df[label] = test_y
        return train_df, test_df

    return train_x, test_x, train_y, test_y


# -----------------------------------
def _get_model(model_name: Union[BaseEstimator, str]):
    if type(model_name) != str:
        return model_name
    if model_name == DECISION_TREE:
        model = DecisionTreeClassifier(max_features="sqrt")
    elif model_name == XGBOOST:
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            learning_rate=0.1,
            n_estimators=30,
            max_depth=10,
            colsample_bytree=0.7,
            alpha=0.0,
            reg_lambda=10.0,
            nthreads=4,
            verbosity=0,
            use_label_encoder=False,
        )
    elif model_name == LOGISTIC:
        model = LogisticRegression()
    else:
        model = KNeighborsClassifier()
    return model


# -----------------------------------
def _plot_precision_recall(y, y_pred, plot_pr):
    if y_pred.shape[1] > 2:
        return
    y_pred = y_pred[:, 1]
    precision, recall, thresholds = metrics.precision_recall_curve(y, y_pred)
    if plot_pr:
        fig, ax = plt.subplots()
        ax.plot(recall, precision, "k--", lw=2)
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        # plt.savefig('./pr_plot.png', dpi=100)
        plt.show()


# -----------------------------------
def _plot_confusion_matrix(classes, y, y_pred):
    cm = metrics.confusion_matrix(y, y_pred, labels=classes)
    print(cm)
    cm = metrics.confusion_matrix(y, y_pred, labels=classes, normalize="true")
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    disp.ax_.set_title("Confusion Matrix")
    plt.show()


# -----------------------------------
def _print_stats(result_metrics: dict):
    metrics_print = {
        MetricNames.AUC_KEY: "ROC AUC",
        MetricNames.PREC_KEY: "Precision",
        MetricNames.RECALL_KEY: "Recall",
        MetricNames.F1_KEY: "F1",
        MetricNames.ACC_KEY: "Accuracy",
        MetricNames.TH_ROC: "Optimal Threshold (ROC curve)",
        MetricNames.TH_PR: "Optimal Threshold (Precision x Recall curve)",
        MetricNames.TH: "Threshold used",
        MetricNames.MSE_KEY: "MSE",
        MetricNames.RMSE_KEY: "RMSE",
        MetricNames.MAE_KEY: "MAE",
        MetricNames.R2_KEY: "R2",
    }
    for key in metrics_print.keys():
        if key in result_metrics.keys():
            print(f"{metrics_print[key]}: {result_metrics[key]}")


# -----------------------------------
def evaluate_set(
    y: np.ndarray,
    y_pred: np.ndarray,
    regression: bool = False,
    plot_pr: bool = True,
    best_th_auc: bool = True,
    classes: Union[list, np.ndarray] = None,
):
    """
    Evaluates the performance of a prediction array based on its true values. This function computes
    a set of metrics, prints the computed metrics, and if the problem is a classfication problem, then
    it also plots the confusion matrix and the precision x recall graph. Finally, return a dictionary
    with the following metrics (depends if it is a regression or classification problem):

        * **Classification:** ROC AUC, Precision, Recall, F1, acuracy, log loss, threshold used, and
          final classes predicted (using the probabilities with the threshold);
        * **Regression:** MSE, RMSE, MAE, and R2

    :param y: an array with the true labels or true values;
    :param y_pred: an arry with the predicted values. For classification problems, this array must contain
        the probabilities for each class, with shape = (N, C), where N is the number of rows and C is the
        number of classes;
    :param regression: if True, regression metrics are computed. If False, only classification
        metrics are computed;
    :param plot_pr: if True, plots a graph showing the precision and recall values for
        different threshold values;
    :param best_th_auc: if True, the best threshold is computed using ROC graph. If False,
        the threshold is computed using the precision x recall graph;
    :param classes: an array or list with the unique classes of the label column. These classes
        are used when plotting the confusion matrix.
    :return: a dictionary with the following metrics:

        * **Classification:** ROC AUC, Precision, Recall, F1, acuracy, log loss, threshold used, and
          final classes predicted (using the probabilities with the threshold);
        * **Regression:** MSE, RMSE, MAE, and R2

    :rtype: dict
    """
    results_dict = get_metrics(y, y_pred, regression=regression, best_th_auc=best_th_auc)
    if not regression:
        _plot_precision_recall(y, y_pred, plot_pr)
        if MetricNames.FINAL_PRED in results_dict.keys():
            _plot_confusion_matrix(classes, y, results_dict[MetricNames.FINAL_PRED])

    _print_stats(results_dict)

    return results_dict


# -----------------------------------
def train_model_plot_results(
    x: pd.DataFrame,
    y: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model: Union[BaseEstimator, str] = DECISION_TREE,
    train_result: bool = True,
    plot_pr: bool = True,
    best_th_auc: bool = True,
):
    """
    Given a train and test sets, and a model name, this function instantiates the model,
    fits the model to the train dataset, predicts the output for the train and test sets,
    and then compute a set of metrics that evaluates the performance of the model in
    both sets. If the "model" parameter is an estimator object, then this object will be
    used instead. These metrics are printed in the stdout. Returns the trained model.
    This function assumes a classification problem.

    :param x: the feature columns of the train dataset;
    :param y: the label column of the train dataset;
    :param x_test: the feature columns of the test dataset;
    :param y_test: the label column of the test dataset;
    :param model: the object of the model to be used, or a string specifying the model
        to be used. The string values allowed are:

            *  **"tree":** Decision Tree Classifier
            *  **"knn":** KNN Classifier
            *  **"xgb":** XGBoost
            *  **"log":** Logistic Regression

    :param train_result: if True, shows the results for the train dataset. If False, show the
        results only for the test dataset;
    :param plot_pr: if True, plots a graph showing the precision and recall values for
        different threshold values;
    :param best_th_auc: if True, the best threshold is computed using ROC graph. If False,
        the threshold is computed using the precision x recall graph.
    :return: returns the model object used to fit the dataset provided.
    :rtype: reference to the model object used
    """
    model = _get_model(model)
    model.fit(x, y)

    pred_train = model.predict_proba(x)
    pred_test = model.predict_proba(x_test)

    if train_result:
        print("\nTRAIN SET:\n")
        evaluate_set(y, pred_train, plot_pr=plot_pr, best_th_auc=best_th_auc, classes=model.classes_)
    print("\nTEST SET:\n")
    evaluate_set(y_test, pred_test, plot_pr=plot_pr, best_th_auc=best_th_auc, classes=model.classes_)

    return model


# -----------------------------------
def train_model_fetch_results(
    x: pd.DataFrame,
    y: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model: Union[BaseEstimator, str] = DECISION_TREE,
    best_th_auc: bool = True,
    regression: bool = False,
):
    """
    Given a train and test sets, and a model name, this function instantiates the model,
    fits the model to the train dataset, predicts the output for the test set,
    and then compute a set of metrics that evaluates the performance of the model in the
    test set. Returns a dictionary with the computed metrics (the same dictionary returned
    by the ``get_metrics()`` function).

    :param x: the feature columns of the train dataset;
    :param y: the label column of the train dataset;
    :param x_test: the feature columns of the test dataset;
    :param y_test: the label column of the test dataset;
    :param model: the object of the model to be used, or a string specifying the model
        to be used. The string values allowed are:

            *  **"tree":** Decision Tree Classifier
            *  **"knn":** KNN Classifier
            *  **"xgb":** XGBoost
            *  **"log":** Logistic Regression

    :param best_th_auc: if True, the best threshold is computed using ROC graph. If False,
        the threshold is computed using the precision x recall graph;
    :param regression: if True, regression metrics are computed. If False, only classification
        metrics are computed.
    :return: a dictionary with the computed metrics(the same dictionary returned
        by the ``get_metrics()`` function).
    :rtype: dict
    """
    model = _get_model(model)
    model.fit(x, y)
    if regression:
        pred_test = model.predict(x_test)
    else:
        pred_test = model.predict_proba(x_test)
    results_dict = get_metrics(y_test, pred_test, regression=regression, best_th_auc=best_th_auc)

    return results_dict
