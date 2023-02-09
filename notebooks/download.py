import os, shutil
from urllib.request import urlretrieve
import zipfile
import pandas as pd


# -----------------------------------
def download_datasets(data_dir):
    outdir_hr = data_dir + 'hr_promotion/'
    outdir_census = data_dir + 'census/'
    os.makedirs(outdir_hr, exist_ok=True)
    os.makedirs(outdir_census, exist_ok=True)
    missing = (
        not os.path.exists(outdir_hr+'train.csv') or
        not os.path.exists(outdir_hr+'test.csv') or
        not os.path.exists(outdir_census+'test.csv') or
        not os.path.exists(outdir_census+'train.csv') or
        not os.path.exists(data_dir+'AdultCensusIncome.csv') or
        not os.path.exists(data_dir+'apartments-train.csv')
    )
    if not missing:
        return

    zipfilename = outdir_hr + 'dataset.zip'
    urlretrieve(
        'https://publictestdatasets.blob.core.windows.net/data/mitigations-datasets.2.22.2022.zip',
        zipfilename
    )
    with zipfile.ZipFile(zipfilename, 'r') as unzip:
        unzip.extractall(outdir_hr)
    unzip_fld = outdir_hr+'mitigations-datasets.2.22.2022/'
    shutil.move(unzip_fld+'hr_promotion/train.csv', outdir_hr+'train.csv')
    shutil.move(unzip_fld+'hr_promotion/test.csv', outdir_hr+'test.csv')
    shutil.move(unzip_fld+'AdultCensusIncome.csv', data_dir+'AdultCensusIncome.csv')
    shutil.rmtree(unzip_fld)
    os.remove(zipfilename)

    zipfilename = data_dir + 'dataset.zip'
    urlretrieve(
        'https://publictestdatasets.blob.core.windows.net/data/responsibleai.12.28.21.zip',
        zipfilename
    )
    with zipfile.ZipFile(zipfilename, 'r') as unzip:
        unzip.extractall(data_dir)
    shutil.move(data_dir+'adult-test.csv', outdir_census+'test.csv')
    shutil.move(data_dir+'adult-train.csv', outdir_census+'train.csv')
    os.remove(data_dir+'stt_testing_data.csv')
    os.remove(data_dir+'face_verify_sample_rand_data.csv')
    os.remove(zipfilename)

