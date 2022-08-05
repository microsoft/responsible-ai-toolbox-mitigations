# Responsible AI Toolbox Mitigations

This repo consists of a python library that aims to help users including data scientists debug and mitigate errors in their data so that they can build more fair and unbiased models starting from the data cleaning stage.

There are two main functions of this library:
- **Data Balance Analysis:** the goal of Data Balance Analysis to provide metrics that help to determine how balanced the data that is being trained on is.
- **Data Processing:** this module contains several transformer classes that aim to change or mitigate certain aspects of a dataset.
The goal of this module is to provide a unified interface for different mitigation methods scattered around
multiple machine learning libraries, such as scikit-learn, mlxtend, sdv, among others.

## Installation

To install this package, follow these steps:
- clone this repo
- move to the root folder of the repo
- install the package using:
```console
> pip install -e .
```

**NOTE:** The installation process will be updated when the package is released publicly in PyPi.

## Data Balance Analysis: Examples

- [Data Balance Analysis Walk Through](notebooks/databalanceanalysis/data_balance_overall.ipynb)
- [Data Balance Analysis Adult Census Example](notebooks/databalanceanalysis/data_balance_census.ipynb)
- [End to End Notebook](notebooks/data_balance_e2e.ipynb)

## Data Processing: Examples

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

## Documentation

The documentation of the **raimitigation** library [can be found here.](https://sturdy-barnacle-3b9f911d.pages.github.io/index.html)

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

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Research and Acknowledgements

**Current Maintainers:** [Matheus Mendonça](https://github.com/mrfmendonca), [Dany Rouhana](https://github.com/danyrouh), [Mark Encarnación](https://github.com/markenc)

**Past Maintainers:** [Akshara Ramakrishnan](https://github.com/akshara-msft), [Irina Spiridonova](https://github.com/irinasp)

**Research Contributors:** [Besmira Nushi](https://github.com/nushib), [Rahee Ghosh](https://github.com/raghoshMSFT), [Ece Kamar](https://www.ecekamar.com/)