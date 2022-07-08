Imputers
========

This sub-module of the **dataprocessing** package collects all imputation transformers implemented here.
Imputers are responsible for removing missing values from a dataset by replacing these missing values with
some valid value. The difference between each imputer lies in how these valid values are computed before
being used to replace a missing value.
All the imputer methods from the **dataprocessing** package are based on the abstract class presented below, called
**DataImputer**.

.. autoclass:: dataprocessing.DataImputer
   :members:
   :show-inheritance:

The following is a list of all imputers implemented in this module. All of the classes below inherit from
the **DataImputer** class, and thus, have access to all of the methods previously shown.

.. toctree::
   :maxdepth: 3

   basic

Example
*******

.. nbgallery::
   ../notebooks/module_tests/imputation