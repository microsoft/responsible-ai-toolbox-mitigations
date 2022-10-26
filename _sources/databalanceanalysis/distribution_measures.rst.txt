.. _distribution_measures:

DistributionBalanceMeasure
==========================


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


.. |br| raw:: html

   <br />

.. automodule:: raimitigations.databalanceanalysis.distribution_measures
   :members:
   :undoc-members:
   :show-inheritance: