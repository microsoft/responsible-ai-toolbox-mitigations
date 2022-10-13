import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import raimitigations.dataprocessing as dp


RESULT_KEYS = ["roc", "pr", "rc", "f1"]
COL_METRIC = "Metric"
COL_VALUE = "Value"
COL_TEST = "Test Case"

# -----------------------------------
def remove_corr_feat(df, label_col):
    """
    Creates a CorrelatedFeatures object, fit this object
    using the dataset df, and then remove a set of the
    correlated features from the dataset. Returns the
    dataset without a set of the correlated features.

    :param df: the full data frame to be analyzed;
    :param label_col: the column name or column index of
        the label column.
    """
    cor_feat = dp.CorrelatedFeatures(
                    method_num_num=["spearman", "pearson", "kendall"],
                    num_corr_th=0.9,
                    num_pvalue_th=0.05,
                    method_num_cat="model",
                    model_metrics=["f1", "auc"],
                    metric_th=0.9,
                    cat_corr_th=0.9,
                    cat_pvalue_th=0.01,
                    save_json=False,
                    verbose=False
                )
    cor_feat.fit(df=df, label_col=label_col)
    proc_df = cor_feat.transform(df)
    return proc_df

# -----------------------------------
def feature_selection(train_x, train_y, test_x, feat_sel_type='fwd'):
    """
    Creates a feature selection object, fit this object
    using the dataset train_x, and then remove a set of the
    correlated features from the datasets train_x and test_x.
    Returns the datasets train_x and test_x containing only
    the selected features.

    :param train_x: the data frame containing only the feature columns of the
        training set;
    :param train_y: the data frame containing only the label column of the
        training set;
    :param test_x: the data frame containing only the feature columns of the
        test set;
    :param feat_sel_type: specifies which feature selection approach is used:

        * 'forward': SeqFeatSelection object using the forward strategy;
        * 'backward': SeqFeatSelection object using the backward strategy;
        * 'catboost': CatBoostSelection object.
    """
    if feat_sel_type == 'forward':
        feat_sel = dp.SeqFeatSelection(forward=True, n_jobs=4, verbose=False)
    elif feat_sel_type == 'backward':
        feat_sel = dp.SeqFeatSelection(forward=False, n_jobs=4, verbose=False)
    elif feat_sel_type == 'catboost':
        feat_sel = dp.CatBoostSelection(verbose=False)
    else:
        raise ValueError("ERROR: 'feat_sel_type' must be one of the following: ['forward, 'backward', 'catboost']")
    feat_sel.fit(X=train_x, y=train_y)
    train_x_sel = feat_sel.transform(train_x)
    test_x_sel = feat_sel.transform(test_x)
    return train_x_sel, test_x_sel

# -----------------------------------
def transform_num_data(train_x, test_x, scaler_ref, num_col):
    """
    Creates a scaler object (specified by the scaler_ref parameter), fit
    it to the train_x dataset, and then apply the scaler to the train_x and
    test_x datasets. Return the train_x and test_x datasets with all numerical
    columns specified by num_col scaled.

    :param train_x: the data frame containing only the feature columns of the
        training set;
    :param test_x: the data frame containing only the feature columns of the
        test set;
    :param scaler_ref: the class reference for the scaler to be used. Must be
        one of the scalers implemented in the scaler dataprocessing.scaler
        submodule;
    :param num_col: a list with the name of the numerical columns.
    """
    ignore = None
    if num_col is not None:
        ignore = [col for col in train_x.columns if col not in num_col]
    transformer = scaler_ref(exclude_cols=ignore, verbose=False)
    transformer.fit(train_x)
    train_x_scl = transformer.transform(train_x)
    test_x_scl = transformer.transform(test_x)
    return train_x_scl, test_x_scl

# -----------------------------------
def artificial_smote(train_x, train_y, strategy, under_sample):
    """
    Creates a Rebalance object using the oversampling strategy specified
    by the strategy parameter and the under sampling method specified by the
    under_sample parameter. Fit this object using the train_x and train_y
    datasets, and then resample the training set provided. Returns the resampled
    train_x and train_y sets.

    :param train_x: the data frame containing only the feature columns of the
        training set;
    :param train_y: the data frame containing only the label column of the
        training set;
    :param strategy: the strategy used for the oversampling method. This is the
        same as the strategy_over parameter from the Rebalance class;
    :param under_sample: the under sampling method used. This is the same as
        the under_sampler parameter from the Rebalance class.
    """
    rebalance = dp.Rebalance(
                X=train_x,
                y=train_y,
                strategy_over=strategy,
                over_sampler=True,
                under_sampler=under_sample,
                verbose=False
            )
    train_x_res, train_y_res = rebalance.fit_resample()
    return train_x_res, train_y_res

# -----------------------------------
def artificial_ctgan(train_x, train_y, strategy, savefile, epochs=400):
    """
    Creates a Synthesizer object with the CTGAN model and using the strategy
    specified by the strategy parameter. Fit this object using the train_x and
    train_y datasets, and then create new artificial instances. Returns the
    original train_x and train_y sets with the new artificial instances.

    :param train_x: the data frame containing only the feature columns of the
        training set;
    :param train_y: the data frame containing only the label column of the
        training set;
    :param strategy: the strategy used for the transform method of the
        Synthesizer class;
    :param epochs: the number of epochs that the Synthesizer class should train for.
    """
    synth = dp.Synthesizer(
                X=train_x,
                y=train_y,
                epochs=epochs,
                model="ctgan",
                load_existing=True,
                save_file=savefile,
                verbose=False
            )
    synth.fit()
    syn_train_x, syn_train_y = synth.transform(X=train_x, y=train_y, strategy=strategy)
    return syn_train_x, syn_train_y

# -----------------------------------
def result_statistics(result_list):
    """
    Build a dictionary with the statistics of the results
    obtained.

    :param result_list: a list of result metrics. Each index
        in this list must be a list of result metrics returned
        by the utils.train_model_fetch_results function.
    """
    result_stat = {}
    for result in result_list:
        for key in RESULT_KEYS:
            if key in result_stat.keys():
                result_stat[key].append(result[key])
            else:
                result_stat[key] = [result[key]]

    return result_stat


# -----------------------------------
def add_results_df(result_df, result_stat, test_name):
    """
    Adds a new experiment in a data frame containing multiple
    experiment results. The new experiment is identified by the
    test_name parameter, and it is defined by the result dictionary
    given by the result_stat parameter. Returns the result_df data
    frame with the data of the new experiment.

    :param result_df: a data frame containing the data of experiments
        executed previously (or None if this is the first experiment);
    :param result_stat: the result dictionary returned by the
        result_statistics function containing the metrics for the new
        experiment to be added to the result_df data frame;
    :param test_name: the name of the new experiment.
    """
    col_test = []
    col_metric = []
    col_value = []
    for metric in RESULT_KEYS:
        col_value += result_stat[metric]
        col_test += [test_name for _ in range(len(result_stat[metric]))]
        col_metric += [metric for _ in range(len(result_stat[metric]))]

    new_df = pd.DataFrame()
    new_df[COL_VALUE] = col_value
    new_df[COL_TEST] = col_test
    new_df[COL_METRIC] = col_metric
    new_df[COL_VALUE] = new_df[COL_VALUE].apply(float)

    if result_df is None:
        return new_df

    result_df = pd.concat([result_df, new_df], axis=0)

    return result_df

# -----------------------------------
def plot_results(res_df, y_lim=[0.5, 0.75]):
    """
    Creates a bar plot with the metrics of multiple experiments.

    :param res_df: the data frame containing the metrics of one or
        more experiments. Must be a data frame returned by the
        add_results_df function;
    :param y_lim: a list with two float values specifying the range
        of the Y axis of the bar plot.
    """
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    #fig.set_dpi(100)

    sns.set_theme(style="whitegrid")
    plt.ylim(y_lim[0], y_lim[1])
    ax = sns.barplot(x=COL_METRIC, y=COL_VALUE, hue=COL_TEST, data=res_df)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=20)
    #ax.axes.set_title("Title",fontsize=50)
    ax.set_xlabel(COL_METRIC, fontsize=30)
    ax.set_ylabel(COL_VALUE, fontsize=30)
    ax.tick_params(labelsize=15)
    plt.show()