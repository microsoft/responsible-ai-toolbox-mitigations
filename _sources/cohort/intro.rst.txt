.. _cohort:

Cohort
======

This module provides the necessary tools for users to create and manage different cohorts throughout
their data processing and model training pipeline. In scenarios where a dataset has cohorts with very
different distributions for certain features, applying a mitigation over the entire dataset might only
make these differences even worse. In these cases, applying a mitigation over each cohort separately is
more advantageous than applying it over the whole dataset. Check out :ref:`this example<target_mitigation>`
for a more in-depth analysis of this problem. The classes available in this module are:

    * :ref:`cohort.CohortDefinition<cohort_def>`: allows the creation and filtering of a cohort. Represents
      a lightweight class capable of saving the filters used to extract a single cohort from a dataset;
    * :ref:`cohort.CohortManager<cohort_manager>`: allows the application of different data processing
      pipelines over each cohort. Also allows the creation and filtering of multiple cohorts using a simple
      interface. Finally, allows the creation of different estimators for each cohort using the ``.predict()``
      and ``predict_proba()`` interfaces. This class uses the :ref:`cohort.CohortDefinition<cohort_def>`
      internally in order to create, filter, and manipulate multiple cohorts.


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
