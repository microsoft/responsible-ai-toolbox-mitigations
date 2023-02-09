.. _DataImputer:

Imputers
========

This sub-module of the **dataprocessing** package collects all imputation transformers implemented here.
Imputers are responsible for removing missing values from a dataset by replacing these missing values with
some valid value inferred from known data. The difference between each imputer lies in how these valid values are computed before
being used to replace a missing value.

We support 3 types of imputation: Basic (Simple), Iterative and K-Nearest Neighbor (KNN). **BasicImputer** is a univariate imputation algorithim; to impute missing values of a feature, it only uses the non-missing values of that feature, while  **IterativeDataImputer** and **KNNDataImputer** are multivariate algorithims that use the entire set of features to estimate all missing values in the set. Hence, you should consider the relationship between the features in your data when choosing an imputation method.

   * :ref:`BasicImputer<basic>`: provides basic strategies to fill missing values, using a stant value or statistics of non-missing data. It takes a single feature into account, one at a time, independently.
   * :ref:`IterativeDataImputer<iterative>`: allows for a more sophisticated but flexible approach, by predicting each feature with missing values as a function of other features in the set. It provides flexibility in the user's choice of regressor used to train, predict and impute data.
   * :ref:`KNNDataImputer<knn>`: scans for the nearest rows to the row with missing data and imputes each missing value using the uniform or distance-weighted average of n-nearest neighbors in the set.

All the imputer methods from the **dataprocessing** package are based on the abstract class presented below, called
**DataImputer**.

.. autoclass:: raimitigations.dataprocessing.DataImputer
   :members:
   :show-inheritance:

The following is a list of all imputers implemented in this module. All of the classes below inherit from
the **DataImputer** class, and thus, have access to all of the methods previously shown.

.. toctree::
   :maxdepth: 1
   :caption: Child Classes

   basic
   iterative
   knn

.. rubric:: Class Diagram

.. inheritance-diagram:: raimitigations.dataprocessing.BasicImputer raimitigations.dataprocessing.IterativeDataImputer raimitigations.dataprocessing.KNNDataImputer
     :parts: 1

Examples
********

.. nbgallery::
   ../../notebooks/dataprocessing/module_tests/basic_imputation
   ../../notebooks/dataprocessing/module_tests/iterative_imputation
   ../../notebooks/dataprocessing/module_tests/knn_imputation