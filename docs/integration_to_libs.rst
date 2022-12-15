.. _integration_other_libs:

How this library works with the Responsible AI Toolbox
======================================================

.. figure:: imgs/responsible-ai-toolbox-mitigations.png
  :scale: 35
  :alt: Responsible AI Toolbox

The Responsible AI Mitigations Library is part of the `Responsible AI Toolbox`_, a larger effort for integrating and building development tools for responsible AI.
One of the central contributions of the Toolbox is the `dashboard`_, which bringing together several mature Responsible AI tools in the areas of `machine learning
interpretability`_, `unfairness assessment and mitigation`_, `error analysis`_, `causal inference`_, and `counterfactual analysis`_ for a holistic assessment and debugging of
models and making informed business decisions.

**A practitioner using the Responsible AI Mitigations Library may rely on the** `Responsible AI Dashboard`_ **: a dashboard to identify and diagnose failure
modes.** Take a look at this `technical blog`_ on how to leverage the dashboard for pre-mitigation steps.

At a high level, components in the dashboard such as Error Analysis and Model Overview help with the identification stage by discovering cohorts of data for which
the model underperforms. Other components like the Data Explorer, Interpretability, and Counterfactual Analysis assist with understanding underlying reasons for
why the model is underperforming. These components go back to the data (Data Explorer) or to the model (Interpretability) to highlight data statistics and feature
importance. As the practitioner investigates the data or the model, they may create hypotheses about how to map the diagnoses to mitigation steps and then implement
them through the Responsible AI Mitigations library.

From a mitigation perspective, `Fairlearn`_ is a closely relevant library in particular for mitigating fairness-related concerns. The set of mitigations in Fairlearn
approach the problem of mitigating model underperformance for given cohorts by framing it as a cost-sensitive classification problem, where samples that satisfy a
particular constraint (similar to the cohort definition) are weighed differently in the optimization process. These mitigations are complementary to the ones provided
here and can be used in combination together.

In addition, we also encourage practitioners to rigorously validate new post-mitigation models and compare them with previous versions to make sure that the mitigation
step indeed improved the model in the way the practitioner expected and that the mitigation step did not lead to new mistakes. To assist with these steps,
`BackwardCompatibilityML`_ is a package for an extended support on model comparison and backward compatible training.


Integration between raimitigations and raiwidgets
-------------------------------------------------

The ``raimitigations`` and `Responsible AI Toolbox`_ (``raiwidgets``) libraries allow users to apply mitigations or analyze the behavior of certain cohorts. These libraries
can be used together in order to analyze and mitigate certain aspects of a given cohort. We demonstrate this integration between these two libraries in the following
notebook, where we show how to create a set of cohorts in one library and continue the work in the other library by importing these cohorts.

.. nbgallery::
   notebooks/cohort/case_study/integration_raiwidgets




.. _Responsible AI Toolbox: https://responsibleaitoolbox.ai/
.. _unfairness assessment and mitigation: https://fairlearn.org/
.. _error analysis: https://erroranalysis.ai/
.. _causal inference: https://github.com/microsoft/EconML
.. _counterfactual analysis: https://github.com/interpretml/DiCE
.. _machine learning interpretability: https://interpret.ml/
.. _Responsible AI Dashboard: https://responsibleaitoolbox.ai/introducing-responsible-ai-dashboard/
.. _dashboard: https://responsibleaitoolbox.ai/introducing-responsible-ai-dashboard/
.. _technical blog: https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/responsible-ai-dashboard-a-one-stop-shop-for-operationalizing/ba-p/3030944
.. _Fairlearn: https://fairlearn.org/
.. _BackwardCompatibilityML: https://github.com/microsoft/BackwardCompatibilityML