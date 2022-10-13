import pandas as pd
from raimitigations.utils import fetch_results
from raimitigations.cohort import CohortManager


def fetch_cohort_results(X, y_true, y_pred, cohort_col):
    def _metric_tuple_to_dict(metric_tuple):
        metric_dict = {
            "roc":metric_tuple[0],
            "pr":metric_tuple[2],
            "recall":metric_tuple[3],
            "f1":metric_tuple[4],
            "acc":metric_tuple[5],
        }
        return metric_dict

    metrics = {}
    metrics['all'] = _metric_tuple_to_dict( fetch_results(y_true, y_pred, best_th_auc=True) )
    metrics['all']['cht_size'] = y_true.shape[0]

    cht_manager = CohortManager(cohort_col=cohort_col)
    cht_manager.fit(X, y_true)
    subsets = cht_manager.get_subsets(X, y_pred)
    y_pred_dict = {}
    for cht_name in subsets.keys():
        y_pred_dict[cht_name] = subsets[cht_name]['y']

    subsets = cht_manager.get_subsets(X, y_true)
    for cht_name in subsets.keys():
        y_subset = subsets[cht_name]['y']
        y_pred_subset = y_pred_dict[cht_name]
        metrics[cht_name] = _metric_tuple_to_dict( fetch_results(y_subset, y_pred_subset, best_th_auc=True) )
        metrics[cht_name]['cht_size'] = y_subset.shape[0]

    queries = cht_manager.get_queries()

    df_dict = {"cohort":[], "cht_query":[], "cht_size":[], "roc":[], "pr":[], "recall":[], "f1":[], "acc":[]}
    for key in metrics.keys():
        df_dict["cohort"].append(key)
        if key == "all":
            df_dict["cht_query"].append("all")
        else:
            df_dict["cht_query"].append(queries[key])
        df_dict["cht_size"].append(metrics[key]["cht_size"])
        df_dict["roc"].append(metrics[key]["roc"])
        df_dict["pr"].append(metrics[key]["pr"].mean())
        df_dict["recall"].append(metrics[key]["recall"].mean())
        df_dict["f1"].append(metrics[key]["f1"].mean())
        df_dict["acc"].append(metrics[key]["acc"])

    df = pd.DataFrame(df_dict)
    return df

