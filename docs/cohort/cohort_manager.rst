.. _cohort_manager:

CohortManager
=============

The **CohortManager** allows the application of different data processing pipelines over each cohort. Also allows the creation and
filtering of multiple cohorts using a simple interface. Finally, allows the creation of different estimators for each cohort using
the ``.predict()`` and ``predict_proba()`` interfaces. This class uses the :ref:`cohort.CohortDefinition<cohort_def>`
internally in order to create, filter, and manipulate multiple cohorts. There are multiple ways of using the
:ref:`cohort.CohortManager<cohort_manager>` class when building a pipeline, and these different scenarios are summarized in following
figure.

.. figure:: ../imgs/scenarios.jpg
  :scale: 25
  :alt: Balancing over cohorts

  Figure 1 - The CohortManager class can be used in different ways to target mitigations to different cohorts. The main differences
  between these scenarios consist on whether the same or different type of data mitigation is applied to the cohort data, and whether
  a single or separate models will be trained for different cohorts. Depending on these choices, CohortManager will take care of
  slicing the data accordingly, applying the specified data mitigation strategy, merging the data back, and retraining the model(s).

The **Cohort Manager - Scenarios and Examples** notebook, located in ``notebooks/cohort/cohort_manager_scenarios.ipynb`` and listed in
the **Examples** section below, shows how each of these scenarios can be implemented through simple code snippets.

.. autoclass:: raimitigations.cohort.CohortManager
   :members:

.. rubric:: Class Diagram

.. inheritance-diagram:: raimitigations.cohort.CohortManager
     :parts: 1

.. _cohort_manager_ex:

Examples
--------

.. nbgallery::
   ../notebooks/cohort/cohort_manager
   ../notebooks/cohort/cohort_manager_scenarios
   ../notebooks/cohort/case_study/case_1
   ../notebooks/cohort/case_study/case_1_rebalance
   ../notebooks/cohort/case_study/case_1_dashboard
   ../notebooks/cohort/case_study/case_2
   ../notebooks/cohort/case_study/case_3
   ../notebooks/cohort/case_study/integration_raiwidgets