# Responsible AI Mitigations


This Responsible-AI-Toolbox-Mitigations repo consists of a python library that aims to empower data scientists and ML developers to measure their dataset balance and representation of different dataset cohorts, while having access to mitigation techniques they could incorporate to mitigate errors and fairness issues in their datasets. Together with the measurement and mitigation steps, ML professionals are empowered to build more accurate and fairer models.

This repo is a part of the [Responsible AI Toolbox](https://github.com/microsoft/responsible-ai-toolbox#responsible-ai-toolbox), a suite of tools providing a collection of model and data exploration and assessment user interfaces and libraries that enable a better understanding of AI systems. These interfaces and libraries empower developers and stakeholders of AI systems to develop and monitor AI more responsibly, and take better data-driven actions.


<p align="center">
<img src="./docs/imgs/responsible-ai-toolbox-mitigations.png" alt="ResponsibleAIToolboxMitigationsOverview" width="750"/>

There are two main functions covered in this library:
- **Data Balance Analysis** (Exploratory Data Analysis): covering metrics that help to determine how balanced is your dataset.
- **Data Processing** (Data Enhancements): covering several transformer classes that aim to change or mitigate certain aspects of a dataset.
The goal of this module is to provide a unified interface for different mitigation methods scattered around
multiple machine learning libraries, such as scikit-learn, mlxtend, sdv, among others.


In this library, we take a **targeted approach to mitigating errors** in Machine Learning models. This is complementary and different from the traditional blanket approaches which aim at maximizing a single-score performance number, such as overall accuracy, by merely increasing the size of traning data or model architecture. Since blanket approaches are often costly but also ineffective for improving the model in areas of poorest performance, with targeted approaches to model improvement we focus the improvement efforts in areas previously identified to have more errors and their underlying diagnoses of error. For example, if a practitioner has identified that the model is underperforming for a cohort of interest by using Error Analysis in the Responsible AI Dashboard, they may also continue the debugging process by finding out through Data Balance Analysis and find out that there is class imbalance for this particular cohort. To mitigate the issue, they then focus on improving class imbalance for the cohort of interest by using the Responsible AI Mitigations library. This and several other examples in the documentation of each mitigation function illustrate how targeted approaches may help practitioner best at mitigation giving them more control in the model improvement process.


## Installation

Use the following pip command to install the Responsible AI Toolbox. Make sure you are using Python 3.7, 3.8, or 3.9.

If running in jupyter, please make sure to restart the jupyter kernel after installing.

```
pip install raimitigations
```
## Documentation

To learn more about the supported dataset measurements and mitigation techniques covered in the **raimitigations** package, [please check out this documentation.](https://responsible-ai-toolbox-mitigations.readthedocs.io/en/latest/)



## Data Balance Analysis: Examples

- [Data Balance Analysis Walk Through](notebooks/databalanceanalysis/data_balance_overall.ipynb)
- [Data Balance Analysis Adult Census Example](notebooks/databalanceanalysis/data_balance_census.ipynb)
- [End to End Notebook](notebooks/data_balance_e2e.ipynb)

## Data Processing/Mitigations: Examples

Here is a set of tutorial notebooks that aim to explain how to use each one of the mitigation
methods offered in the **dataprocessing** module.

- [Encoders](notebooks/dataprocessing/module_tests/encoding.ipynb)
- [Scalers](notebooks/dataprocessing/module_tests/scaler.ipynb)
- [Imputers](notebooks/dataprocessing/module_tests/imputation.ipynb)
- [Sequential Feature Selection](notebooks/dataprocessing/module_tests/feat_sel_sequential.ipynb)
- [Feature Selection using Catboost](notebooks/dataprocessing/module_tests/feat_sel_catboost.ipynb)
- [Identifying correlated features: tutorial](notebooks/dataprocessing/module_tests/feat_sel_corr_tutorial.ipynb)
- [Data Rebalance using imblearn](notebooks/dataprocessing/module_tests/rebalance_imbl.ipynb)
- [Data Rebalance using SDV](notebooks/dataprocessing/module_tests/rebalance_sdv.ipynb)

Here is a set of case study scenarios where we use the transformations available in the **dataprocessing**
module in order to train a model for a real-world dataset.

- [Simple Example](notebooks/dataprocessing/module_tests/model_test.ipynb)
- [Case Study 1](notebooks/dataprocessing/case_study/case1.ipynb)
- [Case Study 2](notebooks/dataprocessing/case_study/case2.ipynb)
- [Case Study 3](notebooks/dataprocessing/case_study/case3.ipynb)



## Dependencies

**RAI Toolbox Mitigations** uses several libraries internally. The direct dependencies are the following:

- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [SciPy](https://scipy.org/)
- [Scikit Learn](https://scikit-learn.org/stable/index.html)
- [ResearchPY](https://pypi.org/project/researchpy/)
- [Statsmodels](https://www.statsmodels.org/stable/index.html)
- [Imbalanced Learn](https://imbalanced-learn.org/stable/)
- [SDV](https://pypi.org/project/sdv/)
- [CatBoost](https://catboost.ai/en/docs/)
- [XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_intro.html)
- [MLxtend](https://pypi.org/project/mlxtend/)
- [UCI Dataset](https://pypi.org/project/uci-dataset/)

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### Pre-Commit

This repository uses pre-commit hooks to guarantee that the code format is kept consistent. For development, make sure to
activate pre-commit before creating a pull request. Any code pushed to this repository is checked for code consistency using
Github Actions, so if pre-commit is not used when doing a commit, there is a chance that it fails in the format check workflow.
Using pre-commit will avoid this.

To use pre-commit with this repository, first install pre-commit:

```console
> pip install pre-commit
```

After installed, navigate to the root directory of this repository and activate pre-commit through the following command:

```console
> pre-commit install
```

With pre-commit installed and activated, whenever you do a new commit, pre-commit will check all new code using the pre-commit hooks configured in the *.pre-commit-config.yaml* file, located in the root of the repository. Some of the hooks might make formatting changes to some of the files commited. If any file is changed or if any other hook fails, the commit will fail. If that happens, make the necessary modifications, add the files to the commit and try commiting one more time. Do this until all hooks are successful. Note that these same checks will be done after pushing anything, so if your commit was successful while using pre-commit, it will pass in the format check workflow as well.

### Updating the Docs

The documentation is built using [Sphinx](https://www.sphinx-doc.org/en/master/), [Pandoc](https://pandoc.org/installing.html), and [Graphviz](https://graphviz.org/) (to build the class diagrams). Graphviz and Pandoc must be installed separately ([detailed instructions here for Graphviz](https://graphviz.org/download/) and [here for Pandoc](https://pandoc.org/installing.html)). On Linux, this can be done with `apt` or `yum` (depending on your distribution):

```console
> sudo apt install graphviz pandoc
```

```console
> sudo yum install graphviz pandoc
```

Make sure Graphviz and Pandoc are installed before recompiling the docs. After that, update the documentation files, which are all located inside the ```docs/``` folder. Finally, use:

```console
> cd docs/
> make html
```

To view the documentation, open the file ```docs/_build/html/index.html``` in your browser.

**Note for Windows users:** if you are trying to update the docs in a Windows environment, you might get an error regarding the *_sqlite3* module:

```
ImportError: DLL load failed while importing _sqlite3: The specified module could not be found.
```

To fix this, following the instructions found [in this link](https://www.dev2qa.com/how-to-fix-importerror-dll-load-failed-while-importing-_sqlite3-the-specified-module-could-not-be-found/).


## Support
### How to file issues and get help

This project uses GitHub Issues to track bugs and feature requests. Please search the existing
issues before filing new issues to avoid duplicates.  For new issues, file your bug or
feature request as a new Issue.

For help and questions about using this project, please post your question in Stack Overflow using the ``raimitigations`` tag.

### Microsoft Support Policy

Support for this package is limited to the resources listed above.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Research and Acknowledgements

**Current Maintainers:** [Matheus Mendonça](https://github.com/mrfmendonca), [Dany Rouhana](https://github.com/danyrouh), [Mark Encarnación](https://github.com/markenc)

**Past Maintainers:** [Akshara Ramakrishnan](https://github.com/akshara-msft), [Irina Spiridonova](https://github.com/irinasp)

**Research Contributors:** [Besmira Nushi](https://github.com/nushib), [Rahee Ghosh Peshawaria](https://github.com/raghoshMSFT), [Ece Kamar](https://www.ecekamar.com/)