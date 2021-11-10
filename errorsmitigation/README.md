# Introduction 
With an increased scrutiny on the impact of applying AI and ML in making real-world decisions, finding, and mitigating errors in model predictions will become a crucial step in vetting a model for production use. Tools like the Error Analysis have made it simpler to diagnose areas where a model is not performing as expected, but there remains a need for tooling to aid data scientists in fixing those errors, thus come ErrorMitigation.


## Overview
The Error Mitigation API is designed to help data scientists and ML engineers of all levels of expertise simplify their ML training dataset preparation process.  To do so, we combined  principled machine learning methods into several API calls:
``` bash
RandomSample (dataset, target, sample_size = 0.2, stratify = False)
# Returns a data random sample or a data random stratify sample.  We use Sklearn to enable this functionality.

Split (dataset, target, train_size = 0.9, random_state = None, categorical_features = True, drop_null = True, drop_duplicates = False, stratify = False)
# Splits the dataset into training and testing sets. In the process, we handle null values, duplicate records, transform all categorical features, and stratify data. 
# We use Sklearn to enable this functionality.

Rebalance (dataset, target, sampling_strategy = ‘auto’, random_state = None, smote_tomek = None, smote = None, tomek = None)
# Combines over- and under-sampling using SMOTE Tomek, or only Over-sampling using SMOTE or under-sampling using Tomek links. 
# We use imblearn and Sklearn to enable this functionality.

Transform (dataset, target, random_state = None, transformer_type, transform_features= None, method = None, output_distribution = None)
# Transforms the data into a standardized or a normalized form using different transformer types. We use Sklearn to enable this functionality. 
```
Description of Error Mitigation API [here](dataprocessing/README.md)

## Usage
### Windows environment

``` bash
Install Python
git clone --single-branch --branch master https://msresearch@dev.azure.com/msresearch/MLMitigationWorkflows/_git/ErrorsMitigationAPI
cd to ErrorsMitigationAPI
python -m venv <env name>
<env name>/Scripts/activate
pip install -r requirements.txt
```
#### Copy datasets 
Create 'datasets' folder in the project
Copy sample dataset from Azure storage 'saerrormitigation/datasets' to 'datasets' folder
Provide dataset folder, for example 'datasets/hr_promotion', while loading training dataset

#### Use Jupyter notebook
cd to notebooks directory  
Run: Jupyter notebook

#### Use VS Code
Open the ErrorsMitigationAPI folder in VSCode.  
Select the built-in interpreter.  
Create a launch.json file.  Edit file to match: 

``` bash
    {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "cwd": "${workspaceFolder}",
            "program": "test\\test_data_split.py", #change test unit name to run different api
            "env": {"PYTHONPATH": "${workspaceRoot}"},
            "console": "integratedTerminal"
    }
```
Run VSCode and test functionality

### Unix environment on WSL
Follow the [WSL setup instructions](https://docs.microsoft.com/en-us/windows/wsl/install-win10)  
Use the [Ubuntu 18.04](https://www.microsoft.com/en-us/p/ubuntu-1804-lts/9n9tngvndl3q?rtc=1#activetab=pivot:overviewtab)  
or [Ubuntu 20.04](https://www.microsoft.com/en-us/p/ubuntu-2004-lts/9n6svws3rx71?rtc=1&activetab=pivot:overviewtab)

After the initial setup, get into the WSL Bash session by running `wsl` and install required packages:
``` bash
sudo apt update
sudo apt install --upgrade git nano build-essential
sudo apt install --upgrade python3-pip python3-setuptools python3-venv
sudo apt install --upgrade python3.7 python3.7-venv python3.7-doc python3.7-dev
python3.7 -m pip install wheel
git clone --single-branch --branch master https://msresearch@dev.azure.com/msresearch/MLMitigationWorkflows/_git/ErrorsMitigationAPI
cd to ErrorsMitigationAPI
python3.7 -m venv <env name>
source <env name>/bin/activate
pip install -r requirements.txt
```

#### Use VS Code 
Open VSCode.  Do Ctrl, Shift, P and select Remote-WSL: new wsl window using distr...  
Open the ErrorsMitigationAPI folder.  
Select the built-in interpreter.  
Create a launch.json file.  Edit file to match: 
``` bash
    {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "justMyCode": false,
            "cwd": "${workspaceFolder}",
            "program": "test/test_data_split.py", #change test unit name to run different api
            "env": {"PYTHONPATH": "${workspaceRoot}"},
            "console": "integratedTerminal"
    }
```
Run VSCode and test functionality
