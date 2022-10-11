.. _getting-started:

Getting Started
===============

Here we provide an overview of the library, while also providing useful links within the documentation.

:ref:`Encoder API<DataEncoding>`
--------------------------------

The :ref:`Encoder API<DataEncoding>` allows for ordinal or one-hot encoding of categorical features.

When is feature encoding a useful mitigation technique?
#######################################################

The :ref:`Encoder API<DataEncoding>` can be useful in cases where a feature does not contain sufficient information about the task because (1) the semantic
information of the feature content has been hidden by the original encoding format or (2) because the model may not have the capacity to interpret the semantic
information.

.. admonition:: Example

    If a feature contains values such as {“agree”, “mostly agree”, “neutral”, “mostly disagree”, “disagree”}, the string interpretation (or
    ordering) of these values cannot express the fact that the “agree” value is better than “mostly agree”. In other cases, if the data contains a
    categorical feature with high cardinality but there is no inherent ordering between the feature values, the training algorithm may still assign an
    inappropriate ordering to the values.

Responsible AI tip about feature encoding
##########################################

Although feature encoding is a generally useful technique in machine learning, it's important to be aware that encoding can sometimes affect different data
cohorts differently, which could result in fairness-related harms or reliability and safety failures. To illustrate, imagine you have two cohorts of interest:
“non-immigrants” and “immigrants”. If the data contains the “country of birth” as a feature, and the value of that feature is mostly uniform within the
“non-immigrant” cohort but highly variable across the “immigrant” cohort, then the wrong ordering interpretation will negatively impact the “immigrant” cohort
more because there are more possible values of the “country of birth” feature.

:ref:`Feature Selection<FeatureSelection>`
------------------------------------------

The :ref:`Feature Selection API<FeatureSelection>` enables selecting a subset of features that are the most informative for the prediction task.

When is feature selection a useful mitigation technique?
########################################################

Sometimes training datasets may contain features that either have very little information about the task or are redundant in the context of other existing
features. Selecting the right feature subset may improve the predictive power of models, their generalization properties, and their inference time. Focusing
only on a subset of features also helps practitioners in the process of model understanding and interpretation.

Responsible AI tip about feature selection
##########################################

It's important to be aware that although feature selection is a generally useful machine-learning technique, it can sometimes affect various data cohorts
differently, with the potential to result in fairness-related harms or reliability and safety failures. For example, it may be the case that within a particular
cohort there exists full correlation between two features, but not with the rest of the data. In addition, if this cohort is also a minority group, the meaning
and weight of a feature value can be drastically different.

.. admonition:: Example

    In the United States there are both private and public undergraduate schools, while in some countries all degree-granting schools are public. A university
    in the United States deciding which applicants to interview for graduate school uses the feature ``previous program type`` (meaning either private or public
    university). The university is interested in several location-based cohorts indicating where applicants did recent undergrad studies. However, a small group
    of applicants are from a country where all schools are public, thus their “previous program type” is always set to “public”. The feature ``previous program
    type`` is redundant for this cohort and not helpful to the prediction task of recommending who to interview. Furthermore, this feature selection could be even
    more harmful if the model, due to existing correlations in the larger data, has learned a negative correlation between “public” undergrad studies to acceptance
    rates in grad school. For the grad school program, this may even lead to harms of underrepresentation or even erasure of individuals from the countries with
    only “public” education.


:ref:`Imputers<DataImputer>`
----------------------------

The :ref:`Imputer API<DataImputer>` enables a simple approach for replacing missing values across several columns with different parameters, simultaneously replacing with the mean,
median, most constant, or most frequent value in a dataset.

When is the Imputer API a useful mitigation technique?
######################################################

Sometimes because of data collection practices, a given cohort may be missing data on a feature that is particularly helpful for prediction. This happens frequently
when the training data comes from different sources of data collection (e.g., different hospitals collect different health indicators) or when the training data
spans long periods of time, during which the data collection protocol may have changed.

Responsible AI tip about imputing value
#######################################

It's important to be aware that although imputing values is a generally useful machine-learning technique, it has the potential to result in fairness-related harms
of over- or underrepresentation, which can impact quality of service or allocation of opportunities or resources, as well as reliability and safety.

It is recommended, for documentation and provenance purposes, to **rename features** after applying this mitigation so that the name conveys the information of which
values have been imputed and how.

To **avoid overfitting**, it is important that feature imputation for testing datasets is performed based on statistics (e.g., minimum, maximum, mean, frequency)
that are retrieved from the training set only. This approach ensures no information from the other samples in the test set is used to improve the prediction on an
individual test sample.


:ref:`Sampling<Sampler>`
------------------------

The :ref:`Sampling API<Sampler>` enables data augmentation by rebalancing existing data or synthesizing new data.

When is the Sampling API a useful mitigation technique?
#######################################################

Sampling helps address data imbalance in a given class or feature, a common problem in machine learning.

Responsible AI tip about sampling
#################################

The problem of data imbalance is most commonly studied in the context of class imbalance. However, from the responsible AI perspective the problem is much broader:
Feature-value imbalance may lead to not enough data for cohorts of interest, which in turn may lead to lower quality predictions.

.. admonition:: Example

    Consider the task of predicting whether a house will sell for higher or lower than the asking price. Even when the class is balanced, there still may be feature
    imbalance for the geographic location because population densities vary in different areas. As such, if the goal is to improve model performance for areas with
    a lower population density, oversampling for this group may help the model to better represent these cohorts.


:ref:`Scalers<DataScaler>`
--------------------------

The :ref:`Scaler API<DataScaler>` enables applying numerical scaling transformations to several features at the same time.

When is scaling feature values a useful mitigation technique?
#############################################################

In general, scaling feature values is important for training algorithms that compute distances between different data samples based on several numerical features
(e.g., KNNs, PCA). But because the semantic meaning of different features can vary significantly, computing distances across scaled versions of such features is
more meaningful.

.. admonition:: Example

    Consider training data has the two numerical features, ``age`` and ``yearly wage``. When computing distances across samples, the ``yearly wage`` feature will
    impact the distance significantly more than the ``age`` - not because it is more important but because it has a higher range of values.

Scaling is also critical for the convergence of popular gradient-based optimization algorithms for neural networks. Scaling also prevents the phenomenon of fast
saturation of activation functions (e.g., sigmoids) in neural networks.

Responsible AI tip about scalers
################################

Note that scalers transform the feature values globally, meaning that they scale the feature based on all samples of the dataset. This may not always be the most
fair or inclusive approach, depending on the use case.

For example, if a training dataset for predicting credit reliability combines data from several countries, individuals with a relatively high salary for their
particular country may still fall in the lower-than-average range when minimum and maximum values for scaling are computed based on data from countries where
salaries are a lot higher. This misinterpretation of their salary may then lead to a wrong prediction, potentially resulting in the withholding of opportunities
and resources.

Similarly in the medical domain, people with different ancestry may have varied minimum and maximum values for specific disease indicators. Scaling globally could
lead the algorithm to underdiagnose the disease of interest for some ancestry cohorts. Of course, depending on the capacity and non-linearity of the training
algorithm, the algorithm itself may find other ways of circumventing such issues. Nevertheless, it may still be a good idea for AI practitioners to apply a more
cohort-aware approach by scaling one cohort at a time.

:ref:`Data Balance Metrics<databalance-api>`
--------------------------------------------

:ref:`Aggregate measures<aggregate_measures>`
#############################################

These measures look at the distribution of records across all value combinations of sensitive feature columns. For example, if ``sex`` and ``race`` are specified as
sensitive features, the API tries to quantify imbalance across all combinations of the specified features (e.g., ``[Male, Black]``, ``[Female, White]``, ``[Male, Asian
Pacific Islander]``)


.. list-table::
   :widths: 5 5 5
   :header-rows: 1
   :class: longtable

   * - Measure
     - Description
     - Interpretation
   * - `Atkinson index`_
     - The Atkinson index presents the |br|
       percentage of total income that |br|
       a given society would have to |br|
       forego in order to have more equal |br|
       shares of income among its |br|
       citizens. This measure depends on |br|
       the degree of societal aversion to |br|
       inequality (a theoretical parameter |br|
       decided by the researcher), where a |br|
       higher value entails greater social |br|
       utility or willingness by individuals |br|
       to accept smaller incomes in exchange |br|
       for a more equal distribution. |br| |br|
       An important feature of the Atkinson |br|
       index is that it can be decomposed |br|
       into within-group and between-group |br|
       inequality.
     - Range ``[0,1]`` |br|
       ``0`` = perfect equality |br|
       ``1`` = maximum inequality |br| |br|
       In this case, it is the |br|
       proportion of records for a |br|
       sensitive column's combination.
   * - `Theil T index`_
     - ``GE(1) = Theil T``, which is more |br|
       sensitive to differences at the |br|
       top of the distribution. The Theil |br|
       index is a statistic used to measure |br|
       economic inequality. The Theil index |br|
       measures an entropic "distance" the |br|
       population is away from the "ideal" |br|
       egalitarian state of everyone having |br|
       the same income.
     - If everyone has the same income, |br|
       then ``T_T`` equals 0. |br| |br|
       If one person has all the income, |br|
       then ``T_T`` gives the result ln(N). |br| |br|
       ``0`` means equal income and larger |br|
       values mean higher level of |br|
       disproportion.
   * - `Theil L index`_
     - GE(0) = Theil L, which is more |br|
       sensitive to differences at the |br|
       lower end of the distribution. |br|
       Thiel L is the logarithm of |br|
       (mean income)/(income i), over |br|
       all the incomes included in the |br|
       summation. It is also referred |br|
       to as the mean log deviation |br|
       measure. Because a transfer from |br|
       a larger income to a smaller one |br|
       will change the smaller income's |br|
       ratio more than it changes the |br|
       larger income's ratio, the |br|
       transfer-principle is satisfied |br|
       by this index.
     - Same interpretation as |br|
       Theil T index.


.. _Atkinson index: https://en.wikipedia.org/wiki/Atkinson_index
.. _Theil T index: https://en.wikipedia.org/wiki/Theil_index
.. _Theil L index: https://en.wikipedia.org/wiki/Theil_index

.. |br| raw:: html

   <br />


:ref:`Distribution measures<distribution_measures>`
###################################################

These metrics compare the data with a reference distribution (currently only uniform distribution is supported). They are calculated per sensitive feature
column and do not depend on the class label column.

.. list-table::
   :widths: 5 5 5
   :header-rows: 1
   :class: longtable

   * - Measure
     - Description
     - Interpretation
   * - `KL divergence`_
     - Kullbeck–Leibler (KL) divergence |br|
       measures how one probability |br|
       distribution is different from |br|
       a second reference probability |br|
       distribution. It is the measure |br|
       of the information gained when |br|
       one revises one's beliefs from |br|
       the prior probability distribution |br|
       Q to the posterior probability |br|
       distribution P. In other words, |br|
       it is the amount of information |br|
       lost when Q is used to approximate P.
     - Non-negative. |br| |br|
       ``0`` means ``P = Q``.
   * - `JS distance`_
     - The Jensen-Shannon (JS) distance |br|
       measures the similarity between two |br|
       probability distributions. It is the |br|
       symmetrized and smoothed version of |br|
       the Kullback–Leibler (KL) divergence |br|
       and is the square root of JS divergence.
     - Range ``[0, 1]``. |br| |br|
       ``0`` means perfectly same to |br|
       balanced distribution.
   * - `Wasserstein distance`_
     - This distance is also known as the |br|
       Earth mover's distance (EMD), since |br|
       it can be seen as the minimum amount |br|
       of “work” required to transform ``u`` |br|
       into ``v``, where “work” is measured |br|
       as the amount of distribution weight |br|
       that must be moved, multiplied by the |br|
       distance it has to be moved.
     - Non-negative. |br| |br|
       ``0`` means ``P = Q``.
   * - `Infinite norm distance`_
     - Also known as the Chebyshev distance |br|
       or chessboard distance, this is the |br|
       distance between two vectors that is |br|
       the greatest of their differences along |br|
       any coordinate dimension.
     - Non-negative. |br| |br|
       ``0`` means ``P = Q``.
   * - `Total variation distance`_
     - The total variation distance is equal |br|
       to half the L1 (Manhattan) distance |br|
       between the two distributions. Take the |br|
       difference between the two proportions |br|
       in each category, add up the absolute |br|
       values of all the differences, and then |br|
       divide the sum by 2.
     - Non-negative. |br| |br|
       ``0`` means ``P = Q``.
   * - `Chi-square test`_
     - The chi-square test is used to test the |br|
       null hypothesis that the categorical |br|
       data has the given expected frequencies |br|
       in each category.
     - The p-value gives evidence |br|
       against null-hypothesis that |br|
       the difference in observed and |br|
       expected frequencies is by |br|
       random chance.



.. _KL divergence: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
.. _JS distance: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
.. _Wasserstein distance: https://en.wikipedia.org/wiki/Wasserstein_metric
.. _Infinite norm distance: https://en.wikipedia.org/wiki/Chebyshev_distance
.. _Total variation distance: https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures
.. _Chi-square test: https://en.wikipedia.org/wiki/Chi-squared_test

:ref:`Feature measures<feature_measures>`
#########################################

These measure whether each combination of sensitive features is receiving the positive outcome (true prediction) at balanced probabilities. Many of these
metrics were influenced by the paper, Measuring Model Biases in the Absence of Ground Truth (Osman Aka, Ken Burke, Alex Bäuerle, Christina Greer, Margaret
Mitchell).

.. _Measuring Model Biases in the Absence of Ground Truth: https://arxiv.org/abs/2103.03417#:~:text=Measuring%20Model%20Biases%20in%20the%20Absence%20of%20Ground,man%20and%20woman%29%20with%20respect%20to%20groundtruth%20labels.

.. list-table::
   :widths: 5 5 5 5
   :header-rows: 1
   :class: longtable

   * - Association |br|
       Metric
     - Family
     - Description
     - Interpretation / |br| Formula
   * - `Statistical parity`_
     - Fairness
     - The proportion of each segment |br|
       of a protected class (e.g., |br|
       gender) should receive the |br|
       positive outcome at equal |br|
       rates.
     - Parity increases with |br|
       proximity to 0. |br| |br|
       DP = P(Y|A=“Male”)- |br|
       P(Y|A=“Female”)
   * - Pointwise |br| mutual |br|
       information |br| (`PMI`_), |br|
       normalized PMI
     - Entropy
     - The PMI of a pair of feature |br|
       values (e.g.,  Gender=Male |br|
       and Gender=Female) quantifies |br|
       the discrepancy between the |br|
       probability of their |br|
       coincidence, given their |br|
       joint distribution and their |br|
       individual distributions |br|
       (assuming independence).
     - Range (normalized) ``[−1,1]`` |br| |br|
       ``-1`` for no co-occurences |br| |br|
       ``0`` for co-occurences at |br|
       random |br| |br|
       ``1`` for complete |br|
       co-occurences
   * - Sorensen-Dice |br|
       coefficient |br| (`SDC`_)
     - Intersection |br|
       over union
     - The SDC is used to gauge the |br|
       similarity of two samples |br|
       and is related to F1 score.
     - Equals twice the number of |br|
       elements common to both |br|
       sets divided by the sum |br|
       of the number of elements |br|
       in each set.
   * - `Jaccard index`_
     - Intersection |br|
       over union
     - Similar to SDC, the Jaccard |br|
       index guages the similarity |br|
       and diversity of sample sets.
     - Equals the size of the |br|
       intersection divided by |br|
       the size of the union of |br|
       the sample sets.
   * - `Kendall rank`_ |br| `correlation`_
     - Correlation |br|
       and |br| statistical |br|
       tests
     - This is used to measure the |br|
       ordinal association between |br|
       two measured quantities.
     - High when observations |br|
       have a similar rank |br|
       between the two variables |br|
       and low when observations |br|
       have a dissimilar rank.
   * - `Log-`_ |br|
       `likelihood`_ |br|
       `ratio`_
     - Correlation |br|
       and |br|
       statistical |br|
       tests
     - This metric calculates the |br|
       degree to which data |br|
       supports one variable versus |br|
       another. The log-likelihood |br|
       ratio gives the probability |br|
       of correctly predicting the |br|
       label in ratio to |br|
       probability of incorrectly |br|
       predicting label.
     - If likelihoods are similar, |br|
       it should be close to 0.
   * - `T-test`_
     - Correlation |br|
       and |br|
       statistical |br|
       tests
     - The t-test is used to |br|
       compare the means of two |br|
       groups (pairwise).
     - The value that is being |br|
       assessed for statistical |br|
       significance in the |br|
       t-distribution.



.. _Statistical parity: https://en.wikipedia.org/wiki/Fairness_%28machine_learning%29
.. _PMI: https://en.wikipedia.org/wiki/Pointwise_mutual_information
.. _SDC: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
.. _Jaccard index: https://en.wikipedia.org/wiki/Jaccard_index
.. _Kendall rank: https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
.. _correlation: https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
.. _Log-: https://en.wikipedia.org/wiki/Likelihood_function#Likelihood_ratio
.. _likelihood: https://en.wikipedia.org/wiki/Likelihood_function#Likelihood_ratio
.. _ratio: https://en.wikipedia.org/wiki/Likelihood_function#Likelihood_ratio
.. _T-test: https://en.wikipedia.org/wiki/Student's_t-test


:ref:`Cohort Management<cohort>`
--------------------------------

The :ref:`Cohort Management<cohort>` feature allows managing multiple cohorts using a simple interface.
This is an important tool for guaranteeing fairness across different cohorts, as shown in the scenarios
described here. The :ref:`cohort.CohortManager<cohort_manager>` allows the application of different data
processing pipelines over each cohort, and therefore represents a powerful tool when dealing with sensitive
cohorts.

.. admonition:: Example: Imputing missing values for each cohort separately

    Consider the following situation: a dataset that shows several details of similar cars from a specific brand.
    The column ``price`` stores the price of a car model in US Dollars, while the column ``country`` indicates
    the country where that price was observed. Due to the differences in economy and local currency, it is expected
    that the price of these models will vary greatly based on the ``country`` column. Suppose now that we want to
    impute the missing values in the ``price`` columns using the mean value of that column. Given that the prices
    differ greatly based on the different country cohorts, then it is expected that this imputation approach
    will end up inserting a lot of noise into the ``price`` column. Instead, we could use the mean value of the
    ``price`` column based on each cohort, that is: compute the mean ``price`` value for each cohort and impute
    the missing values based on the mean value of the cohort to which the instance belongs. This will
    greatly reduce the noise inserted by the imputation method. This can be easily achieved by using the
    :ref:`cohort.CohortManager<cohort_manager>` class.


Get involved
------------

In the future, we plan to integrate more functionalities around data and model-oriented mitigations. Some top-of-mind improvements for the team include bagging and
boosting, better data synthesis, constrained optimizers, and handling data noise. If you would like to collaborate or contribute to any of these ideas, contact us
at responsible-ai-toolbox@microsoft.com.

