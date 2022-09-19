import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline

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
        samples=500,
        n_features=2,
        n_num_num=0,
        n_cat_num=2,
        n_cat_cat=0,
        num_num_noise=[0.01, 0.05],
        pct_change=[0.05, 0.1],
    )
    col_with_nan = ["num_0", "num_1"]
    for col in col_with_nan:
        if col != "label":
            df[col] = add_nan(df[col], 0.1)

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

X, y = create_df()

cohort_pipeline = [
    dp.BasicImputer(verbose=False),
    dp.DataMinMaxScaler(verbose=False)
]

'''
c1 = [ ['CN_0_num_0', '==', 'val0_1'], 'and', ['num_0', '>', 0.0] ]
c2 = [ ['CN_0_num_0', '==', 'val0_0'], 'and', ['num_0', '>', 0.0] ]
c3 = None
#c3 = [ ['CN_1_num_1', '==', 'val1_1'] ]

cohort_set = CohortManager(
    transform_pipe=cohort_pipeline,
    cohort_def=[c1, c2, c3]
)
'''

cohort_set = CohortManager(
    transform_pipe=cohort_pipeline,
    cohort_col=["CN_0_num_0", "CN_1_num_1"]
)
cohort_set.fit(X=X, y=y)
cohort_set.save("cohort.json")
new_X = cohort_set.transform(X)


cohort_set = CohortManager(
    transform_pipe=cohort_pipeline,
    cohort_def="cohort.json"
)
cohort_set.fit(X=X, y=y)
new_X = cohort_set.transform(X)


# ------------------------
# using sklearn's pipeline
# ------------------------
'''
skpipe = Pipeline([
    ("cohort_preprocess", cohort_set),
    ("encoder", dp.EncoderOrdinal(verbose=False)),
    ("model", get_model())
])
skpipe.fit(X, y)
pred = skpipe.predict_proba(X)
pred = skpipe.predict(X)
print(pred)
'''

# -----------------------------------

X, y = create_df()

try:
    cohort_pipeline = [
        dp.BasicImputer(verbose=False),
        dp.DataMinMaxScaler(verbose=False)
    ]
    cohort_set = CohortManager(
        transform_pipe=None,
        cohort_col=["CN_0_num_0", "CN_1_num_1"]
    )
except Exception as e:
    print(e)


try:
    cohort_pipeline = [
        dp.BasicImputer(verbose=False),
        dp.DataMinMaxScaler(verbose=False),
        dp.Rebalance(verbose=False)
    ]
    cohort_set = CohortManager(
        transform_pipe=cohort_pipeline,
        cohort_col=["CN_0_num_0", "CN_1_num_1"]
    )
except Exception as e:
    print(e)


try:
    cohort_pipeline = [
        dp.BasicImputer(verbose=False),
        dp.DataMinMaxScaler(verbose=False),
        dp.Synthesizer(verbose=False)
    ]
    cohort_set = CohortManager(
        transform_pipe=cohort_pipeline,
        cohort_col=["CN_0_num_0", "CN_1_num_1"]
    )
except Exception as e:
    print(e)


try:
    cohort_pipeline = [
        dp.BasicImputer(verbose=False),
        get_model(),
        dp.DataMinMaxScaler(verbose=False),
    ]
    cohort_set = CohortManager(
        transform_pipe=cohort_pipeline,
        cohort_col=["CN_0_num_0", "CN_1_num_1"]
    )
except Exception as e:
    print(e)

