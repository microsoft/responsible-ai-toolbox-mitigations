.. _install_guide:

Installation guide
==================

Responsible AI Mitigations requires Python 3.7, 3.8, or 3.9. Make sure you have one of these Python versions before continuing.

To install, open your console and type:

.. code-block:: console

    > pip install raimitigations

``raimitigations`` has three installation options:

    1. ``pip install raimitigations``: the default installation showed above installs only the essential packages. It doesn't install packages
       used in some tutorial Python notebooks included in the ``notebooks/`` folder
    2. ``pip install raimitigations[all]``: this option installs all essential packages installed in the default option plus
       all packages used in the tutorial notebooks in the ``notebooks/`` folder
    3. ``pip install raimitigations[dev]``: installs all packages installed in the ``[all]`` option plus all packages used for
       development, such as: ``pytest``, ``sphinx``, etc.

Alternatively, you can also clone the main branch of the `Responsible AI Mitigations official repository`_. Then, move to the root folder of the repo, and type:

.. _Responsible AI Mitigations official repository: https://github.com/microsoft/responsible-ai-toolbox-mitigations

.. code-block:: console

    > pip install .