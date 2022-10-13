import random
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


def _create_num_var(samples: int, regression: bool, n_features: int, n_correlated: int):
    """
    Creates an artificial dataset with the column 'label' representing
    the label column (binary or float values) and the remaining columns
    represent the features that should be used to predict the label.

    :param samples: the number of samples to be created
    :param n_features: the number of features to be created
    :param n_correlated: the number of correlated features. If
        n_correlated > 0, some features will be correlated to each other
    :param regression: if True, the label column consists of float values. If False,
        the label column is created to resamble a classification task.
    :return: a dataframe containing only numerical features.
    :rtype: pd.DataFrame
    """
    n_informative = n_features - n_correlated
    if regression:
        X, y = make_regression(
            n_samples=samples,
            n_features=n_features,
            n_informative=n_informative,
        )
    else:
        X, y = make_classification(
            n_samples=samples,
            n_features=n_features,
            n_informative=n_informative,
            n_repeated=0,
            n_redundant=n_correlated,
            n_clusters_per_class=1,
            weights=[0.2],
            class_sep=2,
        )

    colnames = ["num_" + str(i) for i in range(n_features)]
    df = pd.DataFrame(X, columns=colnames)
    df["label"] = y
    return df


# -----------------------------------
def _add_cor_num_num_var_det(df: pd.DataFrame, n_correlated: int, num_num_noise: list):
    """
    Creates n_correlated new numerical features. Each new column created is based on
    one of the existing columns in the dataset. The ith new column created uses the
    ith numerical column of the dataset df as the base column, and then adds a noise
    to it. This way, the new column is created by using a baseline existing column and
    adding a noise to it, where the standard deviation used for the noise is a random
    value between num_num_noise[0] and num_num_noise[1].

    :param df: a dataframe containing only numerical features that will be modified to
        include a set of categorical features correlated to its existing numerical
        features. This dataframe must be created using the 'create_num_var' function;
    :param n_correlated: the number of correlated numerical features that should be
        created;
    :param num_num_noise:  a list with two values, where num_num_noise[0] < num_num_noise[1]
        and both values must be between [0, 1]. The ith new numerical feature is created
        by copying the ith existing numerical feature in the dataset df and adding a noise
        to it. The standard deviation used for generating the noise is a random value
        between num_num_noise[0] and num_num_noise[1].
    :return: a dataframe containing correlated features.
    :rtype: pd.DataFrame
    """
    num_col = [col for col in df.columns if "num_" in col]

    if n_correlated > len(num_col):
        raise ValueError("ERROR: trying to create to many correlated numerical features.")

    for i in range(n_correlated):
        org_col = df[num_col[i]].values
        std_noise = num_num_noise[0] + random.random() * (num_num_noise[1] - num_num_noise[0])
        noise = np.random.normal(0, std_noise, org_col.shape[0])
        new_col = org_col + noise
        col_name = f"num_c{i}_{num_col[i]}"
        df[col_name] = new_col

    return df


# -----------------------------------
def _add_cor_num_cat_var_det(df: pd.DataFrame, n_categorical: int, pct_change: list = [0.1, 0.3], name: str = "cat"):
    """
    Creates a set of categorical features that are correlated to the
    existing numerical features in the dataframe 'df'. The ith new
    categorical feature created will be correlated to the ith existing
    numerical feature of the dataset df. To force this correlation, the
    numerical feature will be categorized by creating bins, where the number
    of bins varies between 2 to 10. Each bin is associated with a categorical
    value. After that, we change a fraction of p bins by swapping the
    categorical value of some bins. Here, p is a value in the range [0,1]
    that is chosen to be between pct_change[0] and pct_change[1].

    :param df: a dataframe containing only numerical
        features that will be modified to include a set of categorical
        features correlated to its existing numerical features. This
        dataframe must be created using the 'create_num_var' function;
    :param n_categorical: the number of correlated categorical features
        that should be created;
    :param pct_change: a list with two values, where
        pct_change[0] < pct_change[1] and both values must be between
        [0, 1]. For each categorical feature created, after creating
        the categorical values based on the numerical bins (based on
        one of the numerical features), a fraction of p values will
        be swapped randomly. Here, p is a value selected randomly
        in the range [pct_change[0], pct_change[1]];
    :param name: the prefix used to create the column name of the new
        categorical features.
    :return: a dataframe containing correlated features.
    :rtype: pd.DataFrame
    """
    num_col = [col for col in df.columns if "num_" in col]
    new_col_list = []
    for i in range(n_categorical):
        # the numerical feature to be correlated to
        col_repl = num_col[i]
        # choose the number of bins used when categorizing
        # the chosen numerical feature
        n_dif_val = random.randint(2, 5)
        # categorize the numerical feature
        labels = [f"val{i}_{j}" for j in range(n_dif_val)]
        new_col = pd.cut(df[col_repl], bins=n_dif_val, labels=labels)

        # choose the fraction of values that
        # will have their categories swapped
        pct_changed = random.uniform(pct_change[0], pct_change[1])
        n_changed = int(pct_changed * df.shape[0])
        rand_len = random.randrange(start=1, stop=n_changed)
        indices = random.sample(range(n_changed), rand_len)
        # swap some of the categorical values to force a
        # reduction in the correlation between the numerical
        # and the categorical feature
        for index in indices:
            new_col[index] = random.choice(labels)

        col_name = f"{name}_{i}_{col_repl}"
        df[col_name] = new_col
        df[col_name] = df[col_name].astype("object")
        new_col_list.append(col_name)

    return df, new_col_list


# -----------------------------------
def create_dummy_dataset(
    samples: int,
    n_features: int,
    n_num_num: int,
    n_cat_num: int,
    n_cat_cat: int,
    num_num_noise: list = [0.1, 0.2],
    pct_change: list = [0.1, 0.3],
    regression: bool = False,
):
    """
    Creates an artificial dataset containing numerical and categorical features, where
    several pairs of correlated features are observed. These pairs of correlated features
    can be a pair of both numerical, both categorical, or numerical and categorical
    features.

    :param samples: the number of samples to be created;
    :param n_features: the number of numerical features to be created;
    :param n_correlated: the number of pairs of correlated features, wherein each pair
        both features are numerical;
    :param n_cat_num: the number of pairs of correlated features, where each pair is
        constituted by a numerical and a categorical feature;
    :param n_cat_cat: the number of pairs of correlated features, wherein each pair both
        features are categorical;
    :param pct_change: a list with two values, where pct_change[0] < pct_change[1]
        and both values must be between [0, 1]. For each categorical feature created, a
        fraction of p values will be swapped randomly. Here, p is a value selected
        randomly in the range [pct_change[0], pct_change[1]];
    :param regression: if True, the label column consists of float values. If False,
        the label column is created to resemble a classification task.
    :return: a dataframe containing correlated features.
    :rtype: pd.DataFrame
    """
    error = False
    if n_num_num > n_features or n_cat_num > n_features or n_cat_cat > n_features:
        error = True
    if n_cat_num < n_cat_cat:
        error = True
    if error:
        raise ValueError(
            "ERROR: invalid parameter for the create_dummy_dataset function. The n_features "
            + "parameter must be the largest value between the following parameters: n_features, "
            + "n_num_num, n_cat_num, and n_cat_cat. Also, the following must also be true: "
            + "n_cat_num >= n_cat_cat."
        )

    df = _create_num_var(samples, regression, n_features, 0)
    df = _add_cor_num_num_var_det(df, n_num_num, num_num_noise)
    df, _ = _add_cor_num_cat_var_det(df, n_cat_num, pct_change, name="CN")
    df, _ = _add_cor_num_cat_var_det(df, n_cat_cat, pct_change, name="CC")
    return df
