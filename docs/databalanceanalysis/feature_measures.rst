.. _feature_measures:

FeatureBalanceMeasure
=====================


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

.. |br| raw:: html

   <br />

.. automodule:: databalanceanalysis.feature_measures
   :members:
   :undoc-members:
   :show-inheritance: