.. _cohort:

Cohort
======

This module provides the necessary tools for users to create and manage different cohorts throughout
their data processing and model training pipeline. In scenarios where a dataset has cohorts with very
different distributions for certain features, applying a mitigation over the entire dataset might only
make these differences worse. In these cases, applying a mitigation over each cohort separately is
more advantageous than applying it over the whole dataset. Check out :ref:`this example<target_mitigation>`
for a more in-depth analysis of this problem. The classes available in this module are:

    * :ref:`cohort.CohortDefinition<cohort_def>`: allows the creation and filtering of a cohort. Represents
      a lightweight class capable of saving the filters used to extract a single cohort from a dataset;
    * :ref:`cohort.CohortManager<cohort_manager>`: allows the application of different data processing
      pipelines over each cohort. Also allows the creation and filtering of multiple cohorts using a simple
      interface. Finally, allows the creation of different estimators for each cohort using the ``.predict()``
      and ``predict_proba()`` interfaces. This class uses the :ref:`cohort.CohortDefinition<cohort_def>`
      internally in order to create, filter, and manipulate multiple cohorts.
    * :ref:`cohort.DecoupledClass<decoupled_class>`: allows training different estimators (models) for different 
      cohorts and combining them in a way that optimizes different definitions of group fairness. It also allows 
      leveraging transfer learning for minority cohorts when the training data for such cohorts is not sufficient. 
      The technique was originally presented in `"Decoupled classifiers for group-fair and efficient machine 
      learning." <https://www.microsoft.com/en-us/research/publication/decoupled-classifiers-for-group-fair-and-efficient-machine-learning/>`_   
      Cynthia Dwork, Nicole Immorlica, Adam Tauman Kalai, and Max Leiserson. Conference on fairness, 
      accountability and transparency. PMLR, 2018.

**Highlights include:**

    * Interface for breaking a dataset into multiple cohorts.
    * Two approaches for defining cohorts: based on the different values of a column, or based on custom filters.
    * Creation of custom pipelines for each cohort, allowing the creation of different estimators for each cohort. 
      Possibility to search and combine the custom estimators for each cohort such that jointly they optimize a group fairness definition of choice.
    * Mitigating cases with low representation data for minority cohorts through transfer learning.
    * Simple interface, which also implements the ``.fit()``, ``.transform()``, ``.predict()``, ``.predict_proba()``, and ``.fit_resample()`` methods
      (some of these methods will only be available depending on the transformations/estimators in the custom pipelines defined for each cohort).


API
---

.. toctree::
   :maxdepth: 2

   cohort


Examples
--------

All notebooks listed here can be found in the folder **notebooks/cohort/**.

.. toctree::
   :maxdepth: 1

   module_tests
   case_studies
