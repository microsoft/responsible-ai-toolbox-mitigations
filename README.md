# Project

This repo consists of a python library that aims to help users including data scientists debug and mitigate errors in their data so that they can build more fair and unbiased models starting from the data cleaning stage.

There are two main functions of this library: Data Balance Analysis and Error Mitigation

The goal of Data Balance Analysis to provide metrics that help to determine how balanced the data that is being trained on is.

# Notebook Examples

- [Data Balance Analysis Walk Through](notebooks/databalanceanalysis/data_balance_overall.ipynb)
- [Data Balance Analysis Adult Census Example](notebooks/databalanceanalysis/data_balance_census.ipynb)
- [Random Sample Mitigation Example](notebooks/dataprocessing/error_random_sample.ipynb)
- [Data Rebalance Mitigation Example](notebooks/dataprocessing/error_rebalance.ipynb)
- [Data Split Example](notebooks/dataprocessing/errors_analysis_split.ipynb)
- [Data Transformer Example](notebooks/dataprocessing/errors_mitigation_transform.ipynb)
- [End to End Notebook](notebooks/data_balance_e2e.ipynb)

## Maintainers

- [Akshara Ramakrishnan](https://github.com/akshara-msft)

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

Data Balance Analysis:

**FeatureBalanceMeasure**

> **label_col** : name of the column that contains the label for the data
> **sensitive_cols** : a list of the columns of interest to analyze for data imbalances
> measures(df)
> Parameters:
> **df** : Pandas Data Frame to calculate the Feature Balance Measures on

**DistributionBalanceMeasure**

> **sensitive_cols** : a list of the columns of interest to analyze for data imbalances
> measures(df)
> Parameters:
> **df** : Pandas Data Frame to calculate the Distribution Balance Measures on

**AggregateBalanceMeasure**

> **sensitive_cols** : a list of the columns of interest to analyze for data imbalances
> measures(df)
> Parameters:
> **df** : Pandas Data Frame to calculate the Aggregate Balance Measures on

Data Processing: Preprocessing data component to help with splitting and
transforming a dataset.

**RandomSample**

sample (dataset, target, sample_size, stratify = False)\*

Return a data random sample or random stratify sample. We use Sklearn to enable
this functionality.

Parameters:

> **dataset** : Pandas Data Frame.

> **target** : str, int

- When str, it is representing the name of the label column

- Wnen int, it corresponds the label column integer index (zero base)

> **sample**\_**size** :

- The training data split size. The default is 0.9, which split the dataset to
  90% training and 10% testing. Training and Test split values add up to 1.

> **stratify** : bool, default is False.

- If not None, data is split in a stratified fashion, using this as the class
  labels.

Return: A Pandas Frame dataset.

**Split**

split (dataset, target, train_size = 0.9, random_state = None,
categorical_features = True, drop_null = True, drop_duplicates = False,
stratify = False)

[sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html?highlight=train_test_split#sklearn.model_selection.train_test_split)

Split the dataset into training and testing sets. In the process, we handle null
values, duplicate records, and transform all categorical features.

Parameters:

> **dataset** : Pandas Data Frame

> **target** : str, int

- When str, it is representing the name of the label column

- Wnen int, it corresponds to the label column integer index (zero base) of
  the target feature
  "

  > **train_size** :

- The training data split size. The default is 0.9, which split the dataset to
  90% training and 10% testing. Training and Test split values add up to 1.

> **random**\_**state** :

- Control the randomization of the algorithm.

- ‘None’: the random number generator is the RandomState instance used by
  np.random.

> **categorical**\_**features** : bool, default=True

- A flag to indicates the presence of categorical features.

> **drop**\_**null** : bool, default=True

- If flag is set to True, records with null values are dropped, otherwise they
  are replaced by the mean.

> **drop_duplicates** : bool, default=False

- If flag is set to True, duplicate records are dropped.

> **stratify** : bool, default=False

- If not None, data is split in a stratified fashion, using this as the class
  labels.

Return: A NumPy array

**Rebalance**

rebalance (dataset, target, sampling_strategy = ‘auto’, random_state =
None, smote_tomek = None, smote = None, tomek = None)\*

Combine over- and under-sampling using SMOTE Tomek. Over-sampling using SMOTE
and under-sampling using Tomek links.

Parameters:

> **dataset** :

- A Pandas Data Frame representing the data to rebalance.

> **target** : str, int

- For using as the classes for rebalancing the data.

- When str, it is representing the name of the label column

- Wnen int, it corresponds to the label column integer index (zero base) of
  the target feature

> **sampling_strategy** : str

- 'minority': resample only the minority class.

- 'not minority': resample all classes but the minority class.

- 'not majority': resample all classes but the majority class.

- 'all': resample all classes.

- 'auto': equivalent to 'not majority'.

> **random_state** :

    Control the randomization of the algorithm.

- ‘None’: the random number generator is the RandomState instance used by
  np.random.

- ‘If Int’: random_state is the seed used by the random number generator.

> **smote_tomek** : The SMOTETomek object to use.

- If not given by Caller, a SMOTE object with default parameters will be
  given.

- [imblearn.combine.SMOTETomek](https://imbalanced-learn.org/dev/references/generated/imblearn.combine.SMOTETomek.html)

> **smote** : The SMOTE object to use.

- If not given by Caller, a SMOTE object with default parameters will be
  given.

- [imblearn_over_sampling.SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html)

> **tomek** : The TomekLinks object to use.

- If not given by Caller, a TomekLinks object with sampling strategy=’all’
  will be given.

- [imblearn.under_sampling.TomekLinks](https://imbalanced-learn.org/dev/references/generated/imblearn.under_sampling.TomekLinks.html)

Return: A rebalanced NumPy array.

Note:
The DataRebalance call with SMOTETomek object to use could be failing with following message:
Expected n_neighbors <= n_samples, but n_samples = 3, n_neighbors = 6
when the data are not perfectly balanced and there are not enough samples (3 in the shown above error) and the number of neighbors is 6.
The workaround solution could be rebalance with SMOTE and Tomek objects instead of SMOTETomek

**Transform**

transform (dataset, target, random_state = None, transformer_type,
transform_features= None, method=None, output_distribution=None)\*

Transform the data into a standardized or a normalized form.

[sklearn.preprocessing](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing)

Parameters:

> **dataset** :

- A Pandas Data Frame representing the data to transform.

> **target** : str, int

- When str, it is representing the name of the label column

- Wnen int, it corresponds to the label column integer index (zero base) of
  the target feature

> **transformer_type** : enum Enum object for available transformations.

- StandardScaler:
  [sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

  Standardize features by removing the mean and scaling to unit variance.  
  z = (x - u) / s (where u is the mean of the training samples or zero if
  with_mean=False, and s is the standard deviation of the training samples or
  one if with_std=False).

- MinMaxScaler:
  [sklearn.preprocessing.MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)

  Transform features by scaling each feature to a given range. This estimator
  scales and translates each feature individually such that it is in the given
  range on the training set, e.g. between zero and one.

- RobustScaler:
  [sklearn.preprocessing.RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)

  Scale features using statistics that are robust to outliers. This Scaler
  removes the median and scales the data according to the quantile range
  (defaults to IQR: Interquartile Range). The IQR is the range between the 1st
  quartile (25th quantile) and the 3rd quartile (75th quantile).

- PowerTransformer:
  [sklearn.preprocessing.PowerTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)

  Apply a power transform feature-wise to make data more Gaussian-like. This
  is useful for modeling issues related to heteroscedasticity (non-constant
  variance), or other situations where normality is desired.  
  Box-Cox transform requires input data to be strictly positive, while
  Yeo-Johnson supports both positive and negative data.

- QuantileTransformer:
  [sklearn.preprocessing.QuantileTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html)

  Transform features using quantiles information. This method transforms the
  features to follow a uniform or a normal distribution. Therefore, for a
  given feature, this transformation tends to spread out the most frequent
  values. It also reduces the impact of (marginal) outliers: this is therefore
  a robust preprocessing scheme.

- Normaliser:
  [sklearn.preprocessing.Normalizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html)

  Normalize samples individually to unit norm. Each sample (i.e. each row of
  the data matrix) with at least one nonzero component is rescaled
  independently of other samples so that its norm (l1, l2 or inf) equals one.

> **transform_features** : List of the features to transform. The list could

    be the indexes or the names of the features.

> **random_state** : Control the randomization of the algorithm.

    ‘None’: the random number generator is the RandomState instance used by
    np.random.

> **method** : str, default=’yeo-johnson’ Possible choices are: ‘yeo-johnson’

    ‘box-cox’

> **output_distribution** : str, default=‘uniform’ Possible choices are:

    ‘uniform’ ‘normal’ Marginal distribution for the transformed data.

Return: A NumPy array.
