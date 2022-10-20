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

from .data_utils import err_float_01

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


# ----------------------------------------
def _pred_to_numpy(pred: Union[np.ndarray, list, pd.DataFrame]):
    if type(pred) == pd.DataFrame:
        pred = pred.to_numpy()
    elif type(pred) == list:
        pred = np.array(pred)
    elif type(pred) != np.ndarray:
        raise ValueError(
            (
                "ERROR: The y_pred parameter passed to the get_metrics_and_log_mlflow() "
                "function must be a numpy array, a list, or a pandas dataframe. Instead, "
                f"got a value from type {type(pred)}."
            )
        )

    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, 1)

    return pred


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
def _roc_evaluation(Y, y_pred):
    # Binary classification
    if y_pred.shape[1] <= 2:
        y_pred = y_pred[:, 1]
        roc_auc = metrics.roc_auc_score(Y, y_pred, average="weighted")
        fpr, tpr, th = metrics.roc_curve(Y, y_pred, drop_intermediate=True)
        target = tpr - fpr
        index = np.argmax(target)
        best_th = th[index]
    # Multi-class
    else:
        roc_auc = metrics.roc_auc_score(Y, y_pred, average="weighted", multi_class="ovr")
        best_th = None

    return roc_auc, best_th


# -----------------------------------
def _get_precision_recall_th(Y, y_pred):
    if y_pred.shape[1] > 2:
        return None
    y_pred = y_pred[:, 1]
    precision, recall, thresholds = metrics.precision_recall_curve(Y, y_pred)
    fscore = (2 * precision * recall) / (precision + recall)
    index = np.argmax(fscore)
    best_th = thresholds[index]
    return best_th


# -----------------------------------
def _plot_precision_recall(Y, y_pred, plot_pr):
    if y_pred.shape[1] > 2:
        return None, None, None
    y_pred = y_pred[:, 1]
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
def probability_to_class_binary(prediction, th):
    classes = []
    prediction = prediction[:, 1]
    for p in prediction:
        c = 0
        if p >= th:
            c = 1
        classes.append(c)
    return classes


# -----------------------------------
def probability_to_class_multi(prediction):
    new_pred = prediction.argmax(axis=1)
    return new_pred


# -----------------------------------
def probability_to_class(prediction, th):
    if prediction.shape[1] > 2:
        return probability_to_class_multi(prediction)
    return probability_to_class_binary(prediction, th)


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
def _print_stats(roc, acc, precision, recall, f1):
    print("Acuracy: %.2f%%" % (acc * 100.0))
    print("\nPrecision: ", np.mean(precision))
    print("\nRecall: ", np.mean(recall))
    print("\nF1 = ", np.mean(f1))
    print("\nROC AUC = ", roc)


# -----------------------------------
def fetch_results(
    Y: np.ndarray, y_pred: Union[np.ndarray, list, pd.DataFrame], best_th_auc: bool, fixed_th: float = None
):
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
    :param y_pred: an arry with the prediction probabilities for each class, with shape = (N, C),
        where N is the number of rows and C is the number of classes;
    :param best_th_auc: if True, the best threshold is computed using ROC graph. If False,
        the threshold is computed using the precision x recall graph. This parameter is ignored
        if 'fixed_th' is a value different from None;
    :param fixed_th: a value between [0, 1] that should be used as the threshold for classification
        tasks.
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
    y_pred = _pred_to_numpy(y_pred)
    roc, auc_th = _roc_evaluation(Y, y_pred)
    pr_th = _get_precision_recall_th(Y, y_pred)
    best_th = pr_th
    if best_th_auc:
        best_th = auc_th
    if fixed_th is not None:
        err_float_01(fixed_th, "fixed_th")
        best_th = fixed_th
    best_y_pred = probability_to_class(y_pred, best_th)
    best_acc = metrics.accuracy_score(Y, best_y_pred)
    best_precision, best_recall, best_f1, s = metrics.precision_recall_fscore_support(Y, best_y_pred)

    return roc, best_th, best_precision, best_recall, best_f1, best_acc, best_y_pred


# -----------------------------------
def evaluate_set(
    Y: np.ndarray,
    y_pred: np.ndarray,
    is_train: bool = True,
    plot_pr: bool = True,
    best_th_auc: bool = True,
    model: BaseEstimator = None,
    classes: Union[list, np.ndarray] = None,
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

    :param Y: an array with the true labels;
    :param y_pred: an arry with the prediction probabilities for each class, with shape = (N, C),
        where N is the number of rows and C is the number of classes;
    :param is_train: a boolean value that indicates if the predictions are for the train
        set or not;
    :param plot_pr: if True, plots a graph showing the precision and recall values for
        different threshold values;
    :param best_th_auc: if True, the best threshold is computed using ROC graph. If False,
        the threshold is computed using the precision x recall graph;
    :param model: the model used to make the predictions given by the **y_pred** parameter.
        This model is only used to get the different classes of the label column. These classes
        can be provided directly to the ``classes`` parameter instead;
    :param classes: an array or list with the unique classes of the label column. These classes
        are used when plotting the confusion matrix. These classes can also be obtained through
        the fitted model, which can be provided directly to the ``model`` parameter instead. This
        parameter is ignored if ``model`` is also provided.
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
    if model is not None:
        classes = model.classes_
    _plot_confusion_matrix(classes, Y, best_y_pred)

    _print_stats(roc, best_acc, best_pr, best_rc, best_f1)

    return roc, pr, rc, th


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
    both sets. These metrics are printed in the stdout. Returns the trained model.

    :param x: the feature columns of the train dataset;
    :param y: the label column of the train dataset;
    :param x_test: the feature columns of the test dataset;
    :param y_test: the label column of the test dataset;
    :param model: the object of the model to be used, or a string specifying the model
        to be used. There are four models allowed:

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
        evaluate_set(y, pred_train, is_train=True, plot_pr=plot_pr, best_th_auc=best_th_auc, model=model)
    evaluate_set(y_test, pred_test, is_train=False, plot_pr=plot_pr, best_th_auc=best_th_auc, model=model)

    return model


# -----------------------------------
def train_model_fetch_results(
    x: pd.DataFrame,
    y: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model: Union[BaseEstimator, str] = DECISION_TREE,
    best_th_auc: bool = True,
):
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
    :param model: the object of the model to be used, or a string specifying the model
        to be used. There are four models allowed:

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
    model = _get_model(model)
    model.fit(x, y)
    pred_test = model.predict_proba(x_test)
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
def fetch_cohort_results(
    X: Union[pd.DataFrame, np.ndarray],
    y_true: Union[pd.DataFrame, np.ndarray],
    y_pred: Union[pd.DataFrame, np.ndarray],
    cohort_def: Union[dict, list, str] = None,
    cohort_col: list = None,
    shared_th: bool = False,
):
    """
    Computes several classification metrics for a given array of predictions
    for the entire dataset and for a set of cohorts. The cohorts used to compute
    these metrics are defined by the ``cohort_def`` or ``cohort_col`` parameters
    (but not both). These parameters are the ones used in the constructor method
    of the :class:`~raimitigations.cohort.CohortManager` class.

    :param X: the dataset containing the feature columns. This dataset is used to
        filter the instances that belong to each cohort;
    :param y_true: the true labels of the instances in the ``X`` dataset;
    :param y_pred: the predicted labels;
    :param cohort_def: a list of cohort definitions, a dictionary of cohort definitions,
        or the path to a JSON file containing the definition of all cohorts. For more
        details on this parameter, please check the :class:`~raimitigations.cohort.CohortManager`
        class;
    :param cohort_col: a list of column names or indices, from which one cohort is created for each
        unique combination of values for these columns. This parameter is ignored if ``cohort_def``
        is provided;
    :param shared_th: if True, the binarization of the predictions is made using the same threshold
        for all cohorts. The threshold used is the one computed for the entire dataset. If False,
        a different threshold is computed for each cohort;
    :return: a dataframe containing the metrics for the entire dataset and for all the defined
        cohorts.
    :rtype: pd.DataFrame
    """
    from ..cohort import CohortManager

    def _metric_tuple_to_dict(metric_tuple):
        metric_dict = {
            "roc": metric_tuple[0],
            "pr": metric_tuple[2],
            "recall": metric_tuple[3],
            "f1": metric_tuple[4],
            "acc": metric_tuple[5],
            "th": metric_tuple[1],
        }
        return metric_dict

    metrics_dict = {}
    metrics_dict["all"] = _metric_tuple_to_dict(fetch_results(y_true, y_pred, best_th_auc=True))
    metrics_dict["all"]["cht_size"] = y_true.shape[0]

    th = None
    if shared_th:
        th = metrics_dict["all"]["th"]

    if cohort_def is None and cohort_col is None:
        raise ValueError("ERROR: one of the two parameters must be provided: 'cohort_col' or 'cohort_def'.")
    if cohort_def is None:
        cht_manager = CohortManager(cohort_col=cohort_col)
    else:
        cht_manager = CohortManager(cohort_def=cohort_def)

    cht_manager.fit(X, y_true)
    subsets = cht_manager.get_subsets(X, y_pred)
    y_pred_dict = {}
    for cht_name in subsets.keys():
        y_pred_dict[cht_name] = subsets[cht_name]["y"]

    subsets = cht_manager.get_subsets(X, y_true)
    for cht_name in subsets.keys():
        y_subset = subsets[cht_name]["y"]
        y_pred_subset = y_pred_dict[cht_name]
        results = fetch_results(y_subset, y_pred_subset, best_th_auc=True, fixed_th=th)
        metrics_dict[cht_name] = _metric_tuple_to_dict(results)
        metrics_dict[cht_name]["cht_size"] = y_subset.shape[0]

    queries = cht_manager.get_queries()

    df_dict = {
        "cohort": [],
        "cht_query": [],
        "cht_size": [],
        "roc": [],
        "threshold": [],
        "pr": [],
        "recall": [],
        "f1": [],
        "acc": [],
    }
    for key in metrics_dict.keys():
        df_dict["cohort"].append(key)
        if key == "all":
            df_dict["cht_query"].append("all")
        else:
            df_dict["cht_query"].append(queries[key])
        df_dict["cht_size"].append(metrics_dict[key]["cht_size"])
        df_dict["roc"].append(metrics_dict[key]["roc"])
        df_dict["threshold"].append(metrics_dict[key]["th"])
        df_dict["pr"].append(metrics_dict[key]["pr"].mean())
        df_dict["recall"].append(metrics_dict[key]["recall"].mean())
        df_dict["f1"].append(metrics_dict[key]["f1"].mean())
        df_dict["acc"].append(metrics_dict[key]["acc"])

    df = pd.DataFrame(df_dict)
    return df
