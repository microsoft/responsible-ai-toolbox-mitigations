from abc import abstractmethod
from typing import Union

import numpy as np
import pandas as pd

from ..dataprocessing import DataProcessing
from ..dataprocessing.data_processing import DataFrameInfo
from ..dataprocessing.imputer.basic_imputer import DataImputer
from ..dataprocessing.encoder.ordinal import EncoderOrdinal
from .utils import is_cat, is_num
from ..utils.data_utils import (
    get_cat_cols,
    _transform_ordinal_encoder_with_new_values,
)


class DataDiagnostics(DataProcessing):
    """
    Base class for all data diagnostics subclasses. Implements basic functionalities
    that can be used for different error prediction approaches.

    :param df: pandas dataframe or np.ndarray to predict errors over;

    :param col_predict: a list of the column names or indices that will be subject to error prediction.
        If None, this parameter will be set automatically as being a list of all columns;
    
    :param mode: a string that can take the values:
        - "column", prediction will be applied to each column independently. 
            An error matrix of the same shape as the data will be returned by predict.
        - "row", prediction will be applied over each row as a whole. A list of 
            erroneous row indices will be returned by predict.

    :param verbose: indicates whether internal messages should be printed or not.
    """
    # -----------------------------------
    def __init__(self, df: Union[pd.DataFrame, np.ndarray] = None, col_predict: list = None, mode: str = None, verbose: bool = True):
        super().__init__(verbose)
        self.df_info = DataFrameInfo()
        self._set_df(df)
        self.n_rows = None
        self.n_cols = None
        self.types = []
        self.col_predict = col_predict
        self.mode = self._check_valid_mode(mode)
        self.ordinal_encoder = None
        self.valid_cols = []
        self.fitted = False
        self.predicted = False

    # -----------------------------------
    def _get_fit_input_type(self) -> int:
        """
        Returns the type of data format this class uses. 

        :return: 0 indicating a pandas dataframe data format;
        :rtype: int.
        """
        return self.FIT_INPUT_DF

    # -----------------------------------
    def _set_column_to_predict(self):
        """
        Sets the col_predict attribute representing columns to predict errors over. 
        If these columns are not provided, it defaults to all columns.
        """
        if self.col_predict is not None:
            return

        self.col_predict = self.df_info.columns.to_list()
        self.print_message("No columns specified for error prediction. Error prediction applied to all columns.")

    # -----------------------------------
    def _check_valid_col_predict(self):
        self.col_predict = self._check_error_col_list(self.df_info.columns, self.col_predict, "col_predict")

     # -----------------------------------
    def _check_valid_mode(self, mode: str):
        """
        Verify that the mode parameter passed by the user is a string 
        equal to "row" or "column" only.

        :param mode: string passed by the user for the mode parameter;

        :return: mode parameter in lower case post validation checks;
        :rtype: string.
        """
        if not isinstance(mode, str):
            raise ValueError(
                f"ERROR: the parameter 'mode' must be a string equal to 'row' or 'column'."
            )
        mode = mode.lower()
        if mode not in ["row", "column"]:
            raise ValueError(
                f"ERROR: the parameter 'mode' must be a string equal to 'row' or 'column' only."
            )
        return mode

    # -----------------------------------
    def _set_column_data_types(self, num_thresh: float = 0.25, cat_thresh: float = 0.05) -> list:
        """
        Detects the data type of each column in col_predict. It has 3 data type options:
            - numerical
            - categorical
            - string

        :param num_thresh: a float threshold of the minimum ratio of float-like values of a numerical column;
        :param cat_thresh: a float threshold of the maximum ratio of unique string data of a categorical column;

        :return: a list of data types mapping to each column in col_predict;
        :rtype: a list.
        """
        for col in self.col_predict:
            col_vals = self.df_info.df[col].values
            if is_num(col_vals, num_thresh):
                self.types.append("numerical")
            elif is_cat(col_vals, cat_thresh):
                self.types.append("categorical")
            else:
                self.types.append("string")
    # -----------------------------------
    def _check_predict_data_structure(self, df: Union[pd.DataFrame, np.ndarray]):
        """
        Checks that all columns seen at fit time are present
        at predict time and vice versa.

        :param df: pandas dataframe used at predict time.
        """
        for col in self.valid_cols:
            if col not in list(df):
                raise KeyError(f"ERROR: Column: {col} seen at fit time, but not present in dataframe.")
        for col in self.col_predict:
            if col not in self.valid_cols:
                raise KeyError(f"ERROR: Column: {col} not seen at fit time.")
            
    # -----------------------------------
    def _apply_encoding_fit(self) -> pd.DataFrame:
        """
        Creates and fits an ordinal encoder to all categorical data before
        fitting the child class if enable_encoder=True, otherwise it excludes
        all categorical data from the process.

        :return: resulting pandas dataframe;
        :rtype: pd.dataframe.
        """
        all_cat_cols = [col for i,col in enumerate(self.col_predict) if self.types[i] in ["categorical", "string"]]
        all_num_cols = [col for col in self.col_predict if col not in all_cat_cols]
        
        df_valid = pd.DataFrame()

        if self.enable_encoder is False:
            self.print_message(
                "\nWARNING: Categorical columns will be excluded from this process.\n"
                + "If you'd like to include these columns, you need to use 'enable_encoder'=True.\n"
            )
            df_valid = self._get_df_subset(self.df_info.df, all_num_cols)

        else:
            self.print_message(
                "\nWARNING: 'enable_encoder'=True and categorical columns will be encoded using ordinal encoding.\n "
            )
            self.ordinal_encoder = EncoderOrdinal(df=self.df_info.df, col_encode=all_cat_cols, unknown_value=np.nan)
            self.ordinal_encoder.fit()
            df_valid = self.ordinal_encoder.transform(self.df_info.df)
            df_valid = self._get_df_subset(df_valid, self.col_predict)
            
        self.valid_cols = list(df_valid)

        return df_valid
    # -----------------------------------
    def _apply_encoding_predict(self, df_valid: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes categorical data before applying predict. It uses the 
        mapping of the encoder created at fit time while also encoding 
        new values on top of the original mapping.

        :param df_valid: pandas dataframe to encode;

        :return: resulting pandas dataframe;
        :rtype: pd.dataframe.
        """
        all_cat_cols = get_cat_cols(df_valid)
        if self.enable_encoder is False:
            if len(all_cat_cols) > 0:
                raise ValueError(
                    "ERROR: Categorical data unseen at fit time and can't be included in the prediction process without encoding.\n"
                    + "If you'd like to ordinal encode and include these columns, use 'enable_encoder'=True.\n"
                )
            df_to_predict = df_valid

        else:
            df_to_predict, _ = _transform_ordinal_encoder_with_new_values(self.ordinal_encoder, df_valid)

        return df_to_predict
    
    # -----------------------------------
    @abstractmethod
    def _fit(self):
        """
        Abstract method. For a given concrete class, this method must run the
        error prediction steps implemented and save any important information 
        in a set of class-specific attributes. These attributes are then used 
        in the predict and transform methods for error prediction.
        """
        pass

    # -----------------------------------
    def fit(self, df: Union[pd.DataFrame, np.ndarray] = None):
        """
        Default fit method for all error predictors that inherit from the DataDiagnostics class. The
        following steps are executed: (i) set the dataset, (ii) set the list of columns that
        will be subject to error prediction, (iii) check for any invalid input, (iv) call the fit method of the
        child class.

        :param df: the dataset to fit the error prediction class on.
        """
        self._set_df(df, require_set=True)
        self._set_column_to_predict()
        self._check_valid_col_predict()
        self._set_column_data_types()
        self._fit()
        self.fitted = True
        self.df_info.clear_df_mem()
        return self

    # -----------------------------------
    @abstractmethod
    def _predict(self, df: pd.DataFrame):
        """
        Abstract method. For a given concrete class, this method must 
        predict errors present in the dataset using the error prediction 
        class implemented and return an error matrix or a list of erroneous 
        row indices based on the mode parameter.

        :param df: a pandas dataframe to perform error prediction on.
        """
        pass

    # -----------------------------------
    def predict(self, df: Union[pd.DataFrame, np.ndarray]) -> Union[np.array, list]:
        """
        Default predict method for all error predictors that inherit from 
        the DataDiagnostics class. Predicts errors in a given dataset.

        :param df: the dataset to perform error prediction over.

        :return: if mode = "row", an error matrix of the same shape as the input 
            data, mapping an error indicator to each value as follows:
            - -1 indicates an error;
            - +1 indicates no error was predicted or that this error module is 
                not applicable for the column's data type;
            - np.nan for columns not in col_predict.
            otherwise if mode = "column", a list of erroneous row indices.
        :rtype: a 2-dimensional np.array or a list.
        """
        self._check_if_fitted()
        predict_df = self._fix_col_transform(df)
        error_result = self._predict(predict_df)
        self.predicted = True
        return error_result

    # -----------------------------------
    def transform(self, df: Union[pd.DataFrame, np.ndarray], imputer: DataImputer = None) -> pd.DataFrame:
        """
        The default transform function of all error predictors that inherit 
        from the DataDiagnostics class. Calls the predict function and transforms 
        an input dataset using the error matrix or list of erroneous row indices 
        predicted. If mode = "row", it removes erroneous rows from the dataset, otherwise 
        if mode = "column", it removes erroneous values in the data with the option 
        to apply a fit/transform imputer object afterwards.

        :param df: the dataset to transform;
        :param imputer: an imputer object to fit to and impute this dataset if mode = "column". 
            You can use imputers present in this library of type dataprocessing.DataImputer,
            your options are: BasicImputer(), IterativeDataImputer() or KNNDataImputer().
            If you'd like to leave erroneous values as np.nan, use None;

        :return: the transformed dataframe;
        :rtype: pandas dataframe.
        """

        #self._check_if_predicted()
        transf_df = self._fix_col_transform(df)
        if self.mode == "row":
            erroneous_indices = self.predict(transf_df)
            return transf_df.drop(erroneous_indices, axis=0).reset_index(drop=True)
        else:
            error_matrix = self.predict(transf_df)
            error_mask = np.where(error_matrix == -1, False, True)
            mask_df = pd.DataFrame(data=error_mask, columns=transf_df.columns)
            masked_df = transf_df[mask_df]
            if imputer is None:
                return masked_df
            else:
                imputer.fit(masked_df)
                new_df = imputer.transform(masked_df)
                return new_df

    '''
    # -----------------------------------
    @abstractmethod
    def _transform(self, df: pd.DataFrame, imputer: DataImputer = None) -> pd.DataFrame:
        """
        Abstract method. For a given concrete class, this method transforms 
        an input dataset using the error matrix predicted by the predict function.
        """
        pass
        
    '''
    # -----------------------------------
    def get_col_predict(self) -> list:
        """
        Returns a list with the column names or column indices of
            columns subject to error prediction.

        :return: a list with the column names or column indices of
            columns subject to error prediction.
        :rtype: list
        """
        return self.col_predict.copy()
