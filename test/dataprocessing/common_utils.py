# Defines common utilities for dataprocessing tests
from asyncio.windows_events import NULL
import numpy as np
import pandas as pd
import zipfile
import pathlib


from pandas import read_csv
from urllib.request import urlretrieve

PASS = -1
DEFAULT_THRESHOLD = 0

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

def validate_rows (data_set, size, drop_null, drop_duplicate):
    if (drop_null):
        data_set.dropna(axis=0, inplace=True)
    if (drop_duplicate):
        data_set = data_set.drop_duplicates()
    num_rows = int(data_set.shape[0] * size)

    return num_rows

def data_set_drop_null_dup (data_set, drop_null, drop_duplicate):
    if (drop_null):
        data_set.dropna(axis=0, inplace=True)
    if (drop_duplicate):
        data_set = data_set.drop_duplicates()
    num_rows = data_set.shape[0]

    return num_rows

""" 
def validate_categorical_trans (data_set, result_set, column):
    value_count = data_set[column].value_counts()
    colNames = result_set.columns[result_set.columns.str.contains(pat = column)]

    return value_count.shape[0] == colNames.shape[0] """

def validate_categorical_trans (data_set, result_set, c_columns):
    for column in c_columns:
        value_count = data_set[column].value_counts()
        colNames = result_set.columns[result_set.columns.str.contains(pat = column)]
        if value_count.shape[0] != colNames.shape[0]:
            return (column)
    return None

def verify_data(input_hr_column, trans_col, threshold):
    #rc = result.column[trans_col]
    index = trans_col.name.index("_")
    for i in range(trans_col.shape[0]):
        elem = trans_col[i]
        if (elem > threshold):
            if (input_hr_column[i] != trans_col.name[-len(input_hr_column[i]):]):
                return i
            
    return PASS

def verify_data_categorical_columns(categorical_columns, result, hr_promotion_10):

    for current_column in categorical_columns:
        input_hr_column = hr_promotion_10[current_column]
        index_for_result = [result.columns.get_loc(col) for col in result.columns if current_column in col]
        
        for column_id in index_for_result:
            trans_col = result.iloc[:,column_id]
            output = verify_data(input_hr_column, trans_col, DEFAULT_THRESHOLD)

            if output != PASS:
                return (column_id, output)
    
    return None

def verify_type_non_categorical_columns(non_categorical_columns, result):

    for current_column in non_categorical_columns:
        #column_type = result[current_column].dtypes
        #if column_type != 'float64':
        #    return (current_column)
        for current_cell in result[current_column]:
            if type(current_cell) != np.float:
                return (current_cell)
   
    return None