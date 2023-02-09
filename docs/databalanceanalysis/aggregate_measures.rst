.. _aggregate_measures:

AggregateBalanceMeasure
=======================

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

.. automodule:: raimitigations.databalanceanalysis.aggregate_measures
   :members:
   :undoc-members:
   :show-inheritance:

