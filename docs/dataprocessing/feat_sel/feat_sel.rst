Feature Selection
=================

This sub-module of the **dataprocessing** package collects a set of transformers that remove a set of unimportant
features from a dataset. The difference between each feature selection approach lies in how this importance metric
is computed. All the feature selection methods from the **dataprocessing** package are based on the abstract
class presented below, called **FeatureSelection**.

.. autoclass:: dataprocessing.FeatureSelection
   :members:
   :show-inheritance:

The following is a list of all feature selection methods implemented in this module. All
of the classes below inherit from the **FeatureSelection** class, and thus, have access to
all of the methods previously shown.

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Child Classes

   seq
   catboost
   correlation


.. rubric:: Class Diagram

.. inheritance-diagram:: dataprocessing.CatBoostSelection dataprocessing.SeqFeatSelection dataprocessing.CorrelatedFeatures
     :parts: 1


Examples
********

.. nbgallery::
   ../notebooks/module_tests/feat_sel_sequential
   ../notebooks/module_tests/feat_sel_catboost
   ../notebooks/module_tests/feat_sel_corr_tutorial
