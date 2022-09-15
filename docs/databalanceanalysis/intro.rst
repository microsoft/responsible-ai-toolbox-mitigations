.. _databalance:

DataBalanceAnalysis
===================

DataBalanceAnalysis contains a set of metrics for diagnosing and measuring data imbalance. This module is intended to be used as part of the error
diagnosis process for failure modes that are due to class or feature imbalance. After measuring with DataBalanceAnalysis, AI practitioners can then
work to mitigate the failure through techniques available in the library's DataProcessing module.

.. admonition:: Example

    A model trained for house-price prediction is discovered to be underperforming for houses that do not have an attached garage. The AI practitioner
    determines that this failure is due to the underrepresentation of houses with no garage in the training data. The practitioner can use metrics in the
    DataBalanceAnalysis module to measure the feature imbalance (“garage” vs. “no garage”), then work to mitigate the issue by using one of the sampling
    techniques available in the library's DataProcessing module for augmenting data.


API
---

.. toctree::
   :maxdepth: 2

   databalanceanalysis


Examples
--------

All notebooks listed here can be found in the folder **notebooks/**.

.. nbgallery::
   ../notebooks/databalanceanalysis/data_balance_census
   ../notebooks/databalanceanalysis/data_balance_overall
   ../notebooks/data_balance_e2e