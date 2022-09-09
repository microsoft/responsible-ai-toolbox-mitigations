.. _DataScaler:

Scalers
=======

This sub-module of the **dataprocessing** package collects all scaler transformers implemented here.
Scalers are responsible scaling numeric features in order to make these features comparable and have similar
value ranges. There are different scaling approaches that can be applied, each with their own advantages.
All the scaling methods from the **dataprocessing** package are based on the abstract class presented below, called
**DataScaler**.

.. autoclass:: dataprocessing.DataScaler
   :members:
   :show-inheritance:

The following is a list of all scalers implemented in this module. All of the classes below inherit from
the **DataScaler** class, and thus, have access to all of the methods previously shown.

.. toctree::
   :maxdepth: 1
   :caption: Child Classes

   standard
   minmax
   quantile
   power
   robust
   normalize

.. rubric:: Class Diagram

.. inheritance-diagram:: dataprocessing.DataRobustScaler dataprocessing.DataPowerTransformer dataprocessing.DataQuantileTransformer dataprocessing.DataMinMaxScaler dataprocessing.DataStandardScaler dataprocessing.DataNormalizer
     :parts: 1


Examples
********

.. nbgallery::
   ../../notebooks/dataprocessing/module_tests/scaler



