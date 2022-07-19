Sampling
========

This sub-module of the **dataprocessing** package collects all sampling transformers implemented here.
Sampling transformers are responsible for creating artificial data for a dataset. This is useful when
a dataset is imbalanced when considering a given class or feature. The difference between sampling methods
lies in how the artificial data is created.

Since the sampling methods implemented here are very different from each other, we didn't create any base class
for the sampling methods. Therefore, each sampling class here inherits directly from the **DataProcessing**
abstract class. Below is a list of the sampling methods present in the **dataprocessing** module:


.. toctree::
   :maxdepth: 1
   :caption: Child Classes

   rebalance
   synthesizer


.. rubric:: Class Diagram

.. inheritance-diagram:: dataprocessing.Rebalance dataprocessing.Synthesizer
     :parts: 1


Examples
********

.. nbgallery::
   ../notebooks/module_tests/rebalance_sdv
   ../notebooks/module_tests/rebalance_imbl