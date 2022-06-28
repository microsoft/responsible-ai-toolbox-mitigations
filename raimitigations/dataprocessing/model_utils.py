import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import matplotlib.pyplot as plt
from copy import deepcopy

DECISION_TREE = "tree"
KNN = "knn"
XGBOOST = "xgb"
LOGISTIC = "log"


# -----------------------------------
def split_data(df, label, test_size=0.2, full_df=False, regression=False):
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
def get_model(model_name):
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
def roc_evaluation(Y, y_pred):
    roc_auc = metrics.roc_auc_score(Y, y_pred, average="weighted")
    fpr, tpr, th = metrics.roc_curve(Y, y_pred, drop_intermediate=True)
    target = tpr - fpr
    index = np.argmax(target)
    best_th = th[index]

    return roc_auc, best_th


# -----------------------------------
def get_precision_recall_th(Y, y_pred):
    precision, recall, thresholds = metrics.precision_recall_curve(Y, y_pred)
    fscore = (2 * precision * recall) / (precision + recall)
    index = np.argmax(fscore)
    best_th = thresholds[index]
    return best_th


# -----------------------------------
def plot_precision_recall(Y, y_pred, plot_pr):
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
def probability_to_class(prediction, th):
    classes = []
    for p in prediction:
        c = 0
        if p >= th:
            c = 1
        classes.append(c)
    return classes


# -----------------------------------
def plot_confusion_matrix(estimator, y, y_pred):
    cm = metrics.confusion_matrix(y, y_pred, labels=estimator.classes_)
    print(cm)
    cm = metrics.confusion_matrix(y, y_pred, labels=estimator.classes_, normalize="true")
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=estimator.classes_)
    disp.plot(cmap=plt.cm.Blues)
    disp.ax_.set_title("Confusion Matrix")
    plt.show()


# -----------------------------------
def print_stats(roc, acc, precision, recall, f1):
    print("Acuracy: %.2f%%" % (acc * 100.0))
    print("\nPrecision: ", np.mean(precision))
    print("\nRecall: ", np.mean(recall))
    print("\nF1 = ", np.mean(f1))
    print("\nROC AUC = ", roc)


# -----------------------------------
def fetch_results(Y, y_pred, best_th_auc):
    roc, auc_th = roc_evaluation(Y, y_pred)
    pr_th = get_precision_recall_th(Y, y_pred)
    best_th = pr_th
    if best_th_auc:
        best_th = auc_th
    best_y_pred = probability_to_class(y_pred, best_th)
    best_acc = metrics.accuracy_score(Y, best_y_pred)
    best_precision, best_recall, best_f1, s = metrics.precision_recall_fscore_support(Y, best_y_pred)

    return roc, best_th, best_precision, best_recall, best_f1, best_acc, best_y_pred


# -----------------------------------
def evaluate_set(model, Y, y_pred, is_train=True, plot_pr=True, best_th_auc=True):

    if is_train:
        print("------------\nTRAIN\n------------")
    else:
        print("------------\nTEST\n------------")

    pr, rc, th = plot_precision_recall(Y, y_pred, plot_pr)

    roc, auc_th, best_pr, best_rc, best_f1, best_acc, best_y_pred = fetch_results(Y, y_pred, best_th_auc)

    plot_confusion_matrix(model, Y, best_y_pred)

    print_stats(roc, best_acc, best_pr, best_rc, best_f1)

    return roc, pr, rc, th


# -----------------------------------
def train_model_plot_results(
    x, y, x_test, y_test, model_name=DECISION_TREE, train_result=True, plot_pr=True, best_th_auc=True
):
    model = get_model(model_name)
    model.fit(x, y)

    pred_train = model.predict_proba(x)[:, 1]
    pred_test = model.predict_proba(x_test)[:, 1]

    if train_result:
        evaluate_set(model, y, pred_train, is_train=True, plot_pr=plot_pr, best_th_auc=best_th_auc)
    evaluate_set(model, y_test, pred_test, is_train=False, plot_pr=plot_pr, best_th_auc=best_th_auc)

    return model


# -----------------------------------
def train_model_fetch_results(x, y, x_test, y_test, model_name=DECISION_TREE, best_th_auc=True):
    model = get_model(model_name)
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
def _get_fold(X, y, train_idx, test_idx, transform_pipe):
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
def evaluate_model_kfold(X, y, transform_pipe, model):
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
