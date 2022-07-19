.. _DataEncoding:

Encoders
========

This sub-module of the **dataprocessing** package collects all encoding transformers implemented here.
Encoders are responsible for encoding categorical features into numerical features. All the encoder
methods from the **dataprocessing** package are based on the abstract class presented below, called
**DataEncoding**.

.. autoclass:: dataprocessing.DataEncoding
   :members:
   :show-inheritance:

The following is a list of all encoders implemented in this module. All of the classes below inherit from
the **DataEncoding** class, and thus, have access to all of the methods previously shown.

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Child Classes

   ordinal
   ohe

.. rubric:: Class Diagram

.. inheritance-diagram:: dataprocessing.EncoderOHE dataprocessing.EncoderOrdinal
     :parts: 1

Examples
********

.. nbgallery::
   ../notebooks/module_tests/encoding