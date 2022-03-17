# Defines common utilities for dataprocessing tests
import numpy as np
import pandas as pd
import zipfile
import pathlib


from pandas import read_csv
from urllib.request import urlretrieve


def download_hr_promotion_data():
    outdirname = 'mitigations-datasets.2.22.2022'
    zipfilename = outdirname + '.zip'
    if not pathlib.Path(outdirname).exists () :
        urlretrieve('https://publictestdatasets.blob.core.windows.net/data/' + zipfilename, zipfilename)
        with zipfile.ZipFile(zipfilename, 'r') as unzip:
            unzip.extractall('.')

    return outdirname

def create_hr_promotion_data():
    outdirname = download_hr_promotion_data()
    hr_promotion = read_csv('./' + outdirname + '/hr_promotion/train.csv').drop(
        ["employee_id"], axis=1)

    return hr_promotion

def create_hr_promotion_10_data():
    outdirname = download_hr_promotion_data()
    hr_promotion = read_csv('./' + outdirname + '/hr_promotion_10/train.csv').drop(
        ["employee_id"], axis=1)

    return hr_promotion

