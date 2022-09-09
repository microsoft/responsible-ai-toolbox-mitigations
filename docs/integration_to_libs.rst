.. _integration_other_libs:

How this library works with the Responsible AI Toolbox
======================================================

The Responsible AI Mitigations Library is part of the `Responsible AI Toolbox`_, a larger effort for integrating and building development tools for responsible AI.
One of the central contributions of the Toolbox is the dashboard, which bringing together several mature Responsible AI tools in the areas of `machine learning
interpretability`_, unfairness assessment and mitigation, error analysis, causal inference, and counterfactual analysis for a holistic assessment and debugging of
models and making informed business decisions.

**A practitioner using the Responsible AI Toolbox Mitigations Library may rely on the** `Responsible AI Dashboard`_ :** a dashboard to identify and diagnose failure
modes.** Take a look at this `technical blog`_ on how to leverage the dashboard for pre-mitigation steps.

At a high level, components in the dashboard such as Error Analysis and Model Overview help with the identification stage by discovering cohorts of data for which
the model underperforms. Other components like the Data Explorer, Interpretability, and Counterfactual Analysis assist with understanding underlying reasons for
why the model is underperforming. These components go back to the data (Data Explorer) or to the model (Interpretability) to highlight data statistics and feature
importance. As the practitioner investigates the data or the model, they may create hypotheses about how to map the diagnoses to mitigation steps and then implement
them through the Responsible AI Toolbox Mitigations library.

From a mitigation perspective, `Fairlearn`_ is a closely relevant library in particular for mitigating fairness-related concerns. The set of mitigations in Fairlearn
approach the problem of mitigating model underperformance for given cohorts by framing it as a cost-sensitive classification problem, where samples that satisfy a
particular constraint (similar to the cohort definition) are weighed differently in the optimization process. These mitigations are complementary to the ones provided
here and can be used in combination together.

In addition, we also encourage practitioners to rigorously validate new post-mitigation models and compare them with previous versions to make sure that the mitigation
step indeed improved the model in the way the practitioner expected and that the mitigation step did not lead to new mistakes. To assist with these steps,
`BackwardCompatibilityML`_ is a package for an extended support on model comparison and backward compatible training.

.. _Responsible AI Toolbox: https://responsibleaitoolbox.ai/
.. _machine learning interpretability: https://interpret.ml/
.. _Responsible AI Dashboard: https://responsibleaitoolbox.ai/introducing-responsible-ai-dashboard/
.. _technical blog: https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/responsible-ai-dashboard-a-one-stop-shop-for-operationalizing/ba-p/3030944
.. _Fairlearn: https://fairlearn.org/
.. _BackwardCompatibilityML: https://github.com/microsoft/BackwardCompatibilityML