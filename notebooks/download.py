import os, shutil
from urllib.request import urlretrieve
import zipfile
import pandas as pd



# -----------------------------------
def download_datasets(data_dir):
    outdirname = data_dir + 'hr_promotion/'
    os.makedirs(outdirname, exist_ok=True)
    missing = (
        not os.path.exists(outdirname+'train.csv') or
        not os.path.exists(outdirname+'test.csv') or
        not os.path.exists(data_dir+'AdultCensusIncome.csv')
    )
    if not missing:
        return

    zipfilename = outdirname + 'dataset.zip'
    urlretrieve(
        'https://publictestdatasets.blob.core.windows.net/data/mitigations-datasets.2.22.2022.zip',
        zipfilename
    )
    with zipfile.ZipFile(zipfilename, 'r') as unzip:
        unzip.extractall(outdirname)
    unzip_fld = outdirname+'mitigations-datasets.2.22.2022/'
    shutil.move(unzip_fld+'hr_promotion/train.csv', outdirname+'train.csv')
    shutil.move(unzip_fld+'hr_promotion/test.csv', outdirname+'test.csv')
    shutil.move(unzip_fld+'AdultCensusIncome.csv', data_dir+'AdultCensusIncome.csv')
    shutil.rmtree(unzip_fld)
    os.remove(zipfilename)
