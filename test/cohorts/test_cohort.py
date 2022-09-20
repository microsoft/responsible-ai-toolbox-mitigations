from typing import Union
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
    log_loss
)

import raimitigations.dataprocessing as dp
from raimitigations.cohort.cohort_definition import CohortDefinition
from raimitigations.cohort.base_cohort import CohortManager

SEED = 42


# -----------------------------------
def create_df():
    np.random.seed(SEED)
    random.seed(SEED)
    def add_nan(vec, pct):
        vec = list(vec)
        nan_index = random.sample(range(len(vec)), int(pct * len(vec)))
        for index in nan_index:
            vec[index] = np.nan
        return vec

    df = dp.create_dummy_dataset(
        samples=1000,
        n_features=3,
        n_num_num=0,
        n_cat_num=0,
        n_cat_cat=0,
        num_num_noise=[0.01, 0.05],
        pct_change=[0.05, 0.1],
    )
    #col_with_nan = ["num_0", "num_1", "CN_0_num_0"]
    #for col in col_with_nan:
    #    if col != "label":
    #        df[col] = add_nan(df[col], 0.05)

    df = df.sample(frac=1)

    X = df.drop(columns=["label"])
    y = df[["label"]]

    return X, y

# -----------------------------------
def get_model():
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
    return model

# -----------------------------------
class MetricNames():
    RMSE_KEY = "rmse"
    MSE_KEY = "mse"
    MAE_KEY = "mae"
    R2_KEY = "r2"
    ACC_KEY = "acc"
    AUC_KEY = "auc"
    PREC_KEY = "precision"
    RECALL_KEY = "recall"
    F1_KEY = "f1"
    LOG_LOSS_KEY = "log_loss"

# -----------------------------------

# ----------------------------------------
def _get_auc_metric(y: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]):
    # Binary classification
    if y_pred.shape[1] <= 2:
        if y_pred.shape[1] == 1:
            y_pred = y_pred[:, 0]
        else:
            y_pred = y_pred[:, 1]
        roc_auc = roc_auc_score(y, y_pred, average="weighted")
    # Multi-class
    else:
        roc_auc = roc_auc_score(y, y_pred, average="weighted", multi_class="ovo")
    return roc_auc

# -----------------------------------
def _probability_to_class_binary(prediction: Union[np.ndarray, list], th: float):
    classes = []
    if prediction.shape[1] == 1:
        prediction = prediction[:, 0]
    else:
        prediction = prediction[:, 1]

    for p in prediction:
        c = 0
        if p >= th:
            c = 1
        classes.append(c)
    return classes


# -----------------------------------
def _probability_to_class_multi(prediction: Union[np.ndarray, list]):
    new_pred = prediction.argmax(axis=1)
    return new_pred


# -----------------------------------
def _probability_to_class(prediction: Union[np.ndarray, list]):
    if prediction.shape[1] > 2:
        return _probability_to_class_multi(prediction)
    return _probability_to_class_binary(prediction, 0.5)


# -----------------------------------
def _get_precision_recall_fscore(y: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]):
    precision, recall, f1, sup = precision_recall_fscore_support(y, y_pred)
    precision_sup = np.sum(precision * sup) / np.sum(sup)
    recall_sup = np.sum(recall * sup) / np.sum(sup)
    f1_sup = np.sum(f1 * sup) / np.sum(sup)
    return precision_sup, recall_sup, f1_sup


# ----------------------------------------
def get_classification_metrics(y: Union[np.ndarray, list], y_pred_prob: Union[np.ndarray, list]):
    roc_auc = _get_auc_metric(y, y_pred_prob)
    y_pred = _probability_to_class(y_pred_prob)
    precision_sup, recall_sup, f1_sup = _get_precision_recall_fscore(y, y_pred)
    acc = accuracy_score(y, y_pred)
    loss = log_loss(y, y_pred_prob)
    results = {
        MetricNames.ACC_KEY: acc,
        MetricNames.AUC_KEY: roc_auc,
        MetricNames.PREC_KEY: precision_sup,
        MetricNames.RECALL_KEY: recall_sup,
        MetricNames.F1_KEY: f1_sup,
        MetricNames.LOG_LOSS_KEY: loss
    }

    return results

# ----------------------------------------

X, y = create_df()

cohort_pipeline = [
    dp.BasicImputer(verbose=False),
    dp.DataMinMaxScaler(verbose=False),
    #dp.EncoderOrdinal(verbose=False),
    get_model()
]


c1 = [ ['num_0', '>', -1] ]
c2 = [ ['num_0', '<', -2] ]
c3 = None
#c3 = [ ['CN_1_num_1', '==', 'val1_1'] ]

cohort_set = CohortManager(
    transform_pipe=cohort_pipeline,
    cohort_def=[c1, c2, c3]
)
cohort_set.fit(X=X, y=y)
#new_X = cohort_set.transform(X)
pred = cohort_set.predict_proba(X)
print(len(pred))

results = get_classification_metrics(y, pred)
print(results)





