import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator
import xgboost as xgb
import matplotlib.pyplot as plt
from copy import deepcopy

DECISION_TREE = "tree"
KNN = "knn"
XGBOOST = "xgb"
LOGISTIC = "log"


# -----------------------------------
def split_data(df: pd.DataFrame, label: str, test_size: float = 0.2, full_df: bool = False, regression: bool = False):
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
        tries to maintain the same proportion of classes in the train and test sets.
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
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size)
    else:
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size, stratify=y)
    if full_df:
        train_df = train_x
        train_df[label] = train_y
        test_df = test_x
        test_df[label] = test_y
        return train_df, test_df

    return train_x, test_x, train_y, test_y


# -----------------------------------
def _get_model(model_name: str):
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
def _roc_evaluation(Y, y_pred):
    roc_auc = metrics.roc_auc_score(Y, y_pred, average="weighted")
    fpr, tpr, th = metrics.roc_curve(Y, y_pred, drop_intermediate=True)
    target = tpr - fpr
    index = np.argmax(target)
    best_th = th[index]

    return roc_auc, best_th


# -----------------------------------
def _get_precision_recall_th(Y, y_pred):
    precision, recall, thresholds = metrics.precision_recall_curve(Y, y_pred)
    fscore = (2 * precision * recall) / (precision + recall)
    index = np.argmax(fscore)
    best_th = thresholds[index]
    return best_th


# -----------------------------------
def _plot_precision_recall(Y, y_pred, plot_pr):
    precision, recall, thresholds = metrics.precision_recall_curve(Y, y_pred)
    if plot_pr:
        fig, ax = plt.subplots()
        ax.plot(recall, precision, "k--", lw=2)
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        # plt.savefig('./pr_plot.png', dpi=100)
        plt.show()
    return precision, recall, thresholds


# -----------------------------------
def _probability_to_class(prediction, th):
    classes = []
    for p in prediction:
        c = 0
        if p >= th:
            c = 1
        classes.append(c)
    return classes


# -----------------------------------
def _plot_confusion_matrix(estimator, y, y_pred):
    cm = metrics.confusion_matrix(y, y_pred, labels=estimator.classes_)
    print(cm)
    cm = metrics.confusion_matrix(y, y_pred, labels=estimator.classes_, normalize="true")
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=estimator.classes_)
    disp.plot(cmap=plt.cm.Blues)
    disp.ax_.set_title("Confusion Matrix")
    plt.show()


# -----------------------------------
def _print_stats(roc, acc, precision, recall, f1):
    print("Acuracy: %.2f%%" % (acc * 100.0))
    print("\nPrecision: ", np.mean(precision))
    print("\nRecall: ", np.mean(recall))
    print("\nF1 = ", np.mean(f1))
    print("\nROC AUC = ", roc)


# -----------------------------------
def fetch_results(Y: np.ndarray, y_pred: np.ndarray, best_th_auc: bool):
    """
    Given a set of true labels (Y) and predicted labels (y_pred), compute a series
    of metrics and values to measure the performance of the predictions provided.
    The metrics computed are:

        * ROC AUC
        * Best threshold found to binarize the results. The predictions are binarized
          using this threshold before computing the precision, recall, F,1 and accuracy
        * Precision
        * Recall
        * F1 score
        * Accuracy
        * Binarized predictions.

    :param Y: an array with the true labels;
    :param y_pred: an arry with the prediction probabilities for each label;
    :param best_th_auc: if True, the best threshold is computed using ROC graph. If False,
        the threshold is computed using the precision x recall graph.
    :return: a tuple with the computed metrics and the binarized predictions. The tuple
        returned contains (in this order):

        * ROC AUC
        * Best threshold found to binarize the results. The predictions are binarized
          using this threshold before computing the precision, recall, F,1 and accuracy
        * Precision
        * Recall
        * F1 score
        * Accuracy
        * Binarized predictions.

    :rtype: tuple
    """
    roc, auc_th = _roc_evaluation(Y, y_pred)
    pr_th = _get_precision_recall_th(Y, y_pred)
    best_th = pr_th
    if best_th_auc:
        best_th = auc_th
    best_y_pred = _probability_to_class(y_pred, best_th)
    best_acc = metrics.accuracy_score(Y, best_y_pred)
    best_precision, best_recall, best_f1, s = metrics.precision_recall_fscore_support(Y, best_y_pred)

    return roc, best_th, best_precision, best_recall, best_f1, best_acc, best_y_pred


# -----------------------------------
def evaluate_set(
    model: BaseEstimator,
    Y: np.ndarray,
    y_pred: np.ndarray,
    is_train: bool = True,
    plot_pr: bool = True,
    best_th_auc: bool = True,
):
    """
    Evaluates the performance of a model based on its predictions. This function computes
    a set of metrics, plots the confusion matrix, plots the precision x recall graph, prints
    the metrics computed, and then returns the following metrics:

        * ROC AUC
        * Precision
        * Recall
        * Best threshold found to binarize the results. The predictions are binarized
          using this threshold before computing the precision, recall, F,1 and accuracy

    :param model: the model used to make the predictions given by the **y_pred** parameter;
    :param Y: an array with the true labels;
    :param y_pred: an arry with the prediction probabilities for each label;
    :param is_train: a boolean value that indicates if the predictions are for the train
        set or not;
    :param plot_pr: if True, plots a graph showing the precision and recall values for
        different threshold values;
    :param best_th_auc: if True, the best threshold is computed using ROC graph. If False,
        the threshold is computed using the precision x recall graph.
    :return: the following computed metrics:

        * ROC AUC
        * Precision
        * Recall
        * Best threshold found to binarize the results. The predictions are binarized
          using this threshold before computing the precision, recall, F,1 and accuracy

    :rtype: tuple
    """
    if is_train:
        print("------------\nTRAIN\n------------")
    else:
        print("------------\nTEST\n------------")

    pr, rc, th = _plot_precision_recall(Y, y_pred, plot_pr)

    roc, auc_th, best_pr, best_rc, best_f1, best_acc, best_y_pred = fetch_results(Y, y_pred, best_th_auc)

    _plot_confusion_matrix(model, Y, best_y_pred)

    _print_stats(roc, best_acc, best_pr, best_rc, best_f1)

    return roc, pr, rc, th


# -----------------------------------
def train_model_plot_results(
    x: pd.DataFrame,
    y: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = DECISION_TREE,
    train_result: bool = True,
    plot_pr: bool = True,
    best_th_auc: bool = True,
):
    """
    Given a train and test sets, and a model name, this function instantiates the model,
    fits the model to the train dataset, predicts the output for the train and test sets,
    and then compute a set of metrics that evaluates the performance of the model in
    both sets. These metrics are printed in the stdout. Returns the trained model.

    :param x: the feature columns of the train dataset;
    :param y: the label column of the train dataset;
    :param x_test: the feature columns of the test dataset;
    :param y_test: the label column of the test dataset;
    :param model_name: a string specifying the model to be used. There are four models allowed:

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
    model = _get_model(model_name)
    model.fit(x, y)

    pred_train = model.predict_proba(x)[:, 1]
    pred_test = model.predict_proba(x_test)[:, 1]

    if train_result:
        evaluate_set(model, y, pred_train, is_train=True, plot_pr=plot_pr, best_th_auc=best_th_auc)
    evaluate_set(model, y_test, pred_test, is_train=False, plot_pr=plot_pr, best_th_auc=best_th_auc)

    return model


# -----------------------------------
def train_model_fetch_results(x, y, x_test, y_test, model_name=DECISION_TREE, best_th_auc=True):
    """
    Given a train and test sets, and a model name, this function instantiates the model,
    fits the model to the train dataset, predicts the output for the test set,
    and then compute a set of metrics that evaluates the performance of the model in the
    test set. Returns a dictionary with the computed metrics, with the following keys:

        * **"roc":** the ROC AUC obtained for the test set;
        * **"th":** the best threshold found for binarizing the predicted probabilities for
          label. This threshold is found by using the ROC graph, or the precision x recall
          graph. The approach used is determined by the 'best_th_auc' parameter;
        * **"pr":** the precision obtained after binarizing the outputs using the previously
          mentioned threshold;
        * **"rc":** the recall obtained after binarizing the outputs using the previously
          mentioned threshold;
        * **"f1":** the F1 score obtained after binarizing the outputs using the previously
          mentioned threshold;
        * **"accuracy":** the accuracy obtained after binarizing the outputs using the previously
          mentioned threshold;
        * **"y_pred":** the binarized predictions.

    :param x: the feature columns of the train dataset;
    :param y: the label column of the train dataset;
    :param x_test: the feature columns of the test dataset;
    :param y_test: the label column of the test dataset;
    :param model_name: a string specifying the model to be used. There are four models allowed:

            *  **"tree":** Decision Tree Classifier
            *  **"knn":** KNN Classifier
            *  **"xgb":** XGBoost
            *  **"log":** Logistic Regression

    :param best_th_auc: if True, the best threshold is computed using ROC graph. If False,
        the threshold is computed using the precision x recall graph.
    :return: a dictionary with the computed metrics, with the following keys:

        * **"roc":** the ROC AUC obtained for the test set;
        * **"th":** the best threshold found for binarizing the predicted probabilities for
          label. This threshold is found by using the ROC graph, or the precision x recall
          graph. The approach used is determined by the 'best_th_auc' parameter;
        * **"pr":** the precision obtained after binarizing the outputs using the previously
          mentioned threshold;
        * **"rc":** the recall obtained after binarizing the outputs using the previously
          mentioned threshold;
        * **"f1":** the F1 score obtained after binarizing the outputs using the previously
          mentioned threshold;
        * **"accuracy":** the accuracy obtained after binarizing the outputs using the previously
          mentioned threshold;
        * **"y_pred":** the binarized predictions.

    :rtype: dict
    """
    model = _get_model(model_name)
    model.fit(x, y)
    pred_test = model.predict_proba(x_test)[:, 1]
    roc, best_th, best_pr, best_rc, best_f1, best_acc, best_y_pred = fetch_results(y_test, pred_test, best_th_auc)

    result = {
        "roc": roc,
        "th": best_th,
        "pr": best_pr.mean(),
        "rc": best_rc.mean(),
        "f1": best_f1.mean(),
        "accuracy": best_acc,
        "y_pred": best_y_pred,
    }

    return result


# -----------------------------------
def _get_fold(X, y, train_idx, test_idx, transform_pipe: list):
    train_x = X.filter(items=train_idx, axis=0)
    train_y = y.filter(items=train_idx, axis=0)
    test_x = X.filter(items=test_idx, axis=0)
    test_y = y.filter(items=test_idx, axis=0)

    transform_copy = [deepcopy(tf) for tf in transform_pipe]
    for tf in transform_copy:
        if tf._get_fit_input_type() == tf.FIT_INPUT_DF:
            tf.fit(train_x)
        else:
            tf.fit(train_x, train_y)
    for tf in transform_copy:
        train_x = tf.transform(train_x)
        test_x = tf.transform(test_x)

    return train_x, train_y, test_x, test_y


# -----------------------------------
def evaluate_model_kfold(X: pd.DataFrame, y: pd.Series, transform_pipe: list, model: BaseEstimator):
    cv = StratifiedKFold(n_splits=10)
    roc_list = []
    for train_idx, test_idx in cv.split(X, y):
        train_x, train_y, test_x, test_y = _get_fold(X, y, train_idx, test_idx, transform_pipe)
        model.fit(X=train_x, y=train_y)
        y_pred = model.predict_proba(test_x)
        if len(y_pred.shape) > 1 and y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
        roc_auc = metrics.roc_auc_score(test_y, y_pred, average="weighted", multi_class="ovr")
        roc_list.append(roc_auc)
    return np.mean(roc_list)
