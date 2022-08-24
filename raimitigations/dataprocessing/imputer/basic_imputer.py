from typing import Union
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

from .imputer import DataImputer
from ..data_utils import get_cat_cols


class BasicImputer(DataImputer):
    """
    Concrete class that imputes missing data in a dataset using a set of simple
    strategies. Implements a simple imputation approach, where the missing values
    are filled with the mean, median, constant value, or the most frequent value,
    where mean and median are only valid for numerical values. This subclass uses
    the :class:`~sklearn.impute.SimpleImputer` class from :mod:`sklearn` in the background.
    The main advantage is that this subclass allows using the simple imputation approach over
    several different columns at once, each with its own set of parameters. For more details see:
    https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html

    :param df: pandas data frame that contains the columns to be encoded;

    :param col_impute: a list of the column names or indexes that will be imputed.
        If None, this parameter will be set automatically as being a list of all
        columns with any NaN value;

    :param categorical: a dict indicating the parameters used by
        :class:`~sklearn.impute.SimpleImputer`. Represents the parameters of  :class:`~sklearn.impute.SimpleImputer`
        used on all categorical columns not represented in the
        ``specific_col`` param. The dict has the following structure:

            | {
            |   **'missing_values'**:np.nan,
            |   **'strategy'**:'constant',
            |   **'fill_value'**:'NULL'
            | }

        where 'missing_values', 'strategy', and 'fill_value' are
        the parameters used by sklearn's SimpleImputer. If None,
        this dict will be auto-filled as the one above;

    :param numerical: similar to ``categorical``, but instead, represents
        the parameters of the SimpleImputer to be used on all
        numerical columns not present in the ``specific_col`` param.
        If None, this dict will be auto-filled as follows:

            | {
            |   **'missing_values'**:np.nan,
            |   **'strategy'**:'mean',
            |   **'fill_value'**:None
            | }

    :param specific_col: a dict of dicts. Each key of the main dict must be a
        column name present in the ``col_impute`` param. This key must
        be associated with a dict similar to the one in ``categorical`` param,
        which indicates the parameters to be used by the SimpleImputer for
        the specified column (key). If one of the columns in ``col_impute``
        are not present in the main dict, then the type of this column is
        automatically identified as being either numeric or categorical.
        And then, the ``categorical`` or ``numerical`` parameters are used for those
        columns. The dict structure is given by:

            | {
            |   **COL_NAME1:**  {
            |           **'missing_values':** np.nan,
            |           **'strategy':** 'constant',
            |           **'fill_value':** 'NULL'
            |       }
            |   **COL_NAME2:**  {
            |           **'missing_values':** np.nan,
            |           **'strategy':** 'constant',
            |           **'fill_value':** 'NULL'
            |       }
            |   etc.
            | }

    :param verbose: indicates whether internal messages should be printed or not.
    """

    # -----------------------------------
    def __init__(
        self,
        df: Union[pd.DataFrame, np.ndarray] = None,
        col_impute: list = None,
        categorical: dict = None,
        numerical: dict = None,
        specific_col: dict = None,
        verbose: bool = True,
    ):
        super().__init__(df, col_impute, verbose)
        self.categorical = categorical
        self.numerical = numerical
        self.specific_col = specific_col
        self._set_dicts()

    # -----------------------------------
    def _check_single_key(self, org_key: str, param_dict: dict):
        """
        Checks if a dictionary passed to the numerical, categorical or specific_col
        parameters are valid dictionary, that is, contains all required keys.

        :param org_key: the key or variable name associated with the dictionary param_dict.
            This variable is used to inform the user which dictionary (param_dict) is
            invalid, that is, the key or variable that is associated with an invalid
            dictionary;
        :param param_dict: the dictionary associated with the 'numerical' or 'categorical'
            parameters, or with one of the keys of the specific_col parameter.
        """
        keys = param_dict.keys()
        if "missing_values" not in keys:
            raise ValueError(
                "ERROR: expected the key 'missing_values' in the dictionary:\n" + f"{org_key}: {param_dict}"
            )
        if "strategy" not in keys:
            raise ValueError("ERROR: expected the key 'strategy' in the dictionary:\n" + f"{org_key}: {param_dict}")
        if param_dict["strategy"] == "constant":
            if "fill_value" not in keys:
                raise ValueError(
                    "ERROR: expected the key 'fill_value' in the dictionary:\n" + f"{org_key}: {param_dict}"
                )
            else:
                is_none = param_dict["fill_value"] is None
                is_nan = False
                if not is_none and type(param_dict["fill_value"]) != str and np.isnan(param_dict["fill_value"]):
                    is_nan = True
                if is_nan or is_none:
                    raise ValueError(
                        "ERROR: when 'strategy' == 'constant' for the SimpleImputer, the "
                        + f"user must provide a valid 'fill_value', not None or Nan.\nError "
                        + f"occured for dictionary:\n{org_key}: {param_dict}"
                    )

    # -----------------------------------
    def _set_dicts(self):
        """
        If one of the dictionaries that specify how to impute a given data
        (numerical, categorical, and specific_col) is set to None, then create
        a default dictionary for all numerical and another dictionary for all
        categorical data.
        """
        if self.categorical is None:
            self.categorical = {"missing_values": np.nan, "strategy": "constant", "fill_value": "NULL"}
        if self.numerical is None:
            self.numerical = {"missing_values": np.nan, "strategy": "mean", "fill_value": None}
        if self.specific_col is None:
            self.specific_col = {}

    # -----------------------------------
    def _check_valid_dicts(self):
        """
        Checks if the dictionaries that specify how to impute a given data
        (numerical, categorical, and specific_col) are all appropriately
        set. Uses the _check_single_key over each individual dictionary.
        """
        param1_err = type(self.categorical) != dict
        param2_err = type(self.numerical) != dict
        param3_err = type(self.specific_col) != dict
        if param1_err or param2_err or param3_err:
            raise ValueError(
                "ERROR: one of the parameters 'categorical', 'numerical', and 'specific_col' "
                + "is not a dict. Check the documentation for more information."
            )

        if self.specific_col != {}:
            if type(list(self.specific_col.keys())[0]) == int:
                new_dict = {}
                if self.column_type == self.COL_NAME:
                    for col in self.specific_col.keys():
                        key = self._get_column_from_index(col)
                        new_dict[key] = self.specific_col[col]
                else:
                    for col in self.specific_col.keys():
                        new_dict[str(col)] = self.specific_col[col]
                self.specific_col = new_dict

            for col in self.specific_col.keys():
                if self.col_impute is not None and col not in self.col_impute:
                    raise ValueError(
                        f"ERROR: unexpected column name in the 'specific_col' dict param. "
                        + f"Expected keys must be in {self.col_impute}"
                    )
                self._check_single_key(col, self.specific_col[col])
        self._check_single_key("categorical", self.categorical)
        self._check_single_key("numerical", self.numerical)

    # -----------------------------------
    def _fit(self):
        """
        Fit method complement used specifically for the current class.
        The following steps are executed: (i) check if all the dictionaries
        are properly set, (ii) separate which columns should be imputed using
        the self.numerical dictionary, which should use the self.categorical
        dictionary, and which columns should use specific dictionaries present
        in the self.specific_col attribute, (iii) run over each column that
        should be imputed (self.col_impute) and create and fit the appropriate
        SimpleImputer object over this column.
        """
        self._check_valid_dicts()
        non_spec_cols = [value for value in self.col_impute if value not in self.specific_col.keys()]
        self.cat_cols = get_cat_cols(self.df, subset=non_spec_cols)
        self.num_cols = [col for col in non_spec_cols if col not in self.cat_cols]

        self.imputers = {}
        for col in self.col_impute:
            if col in self.cat_cols:
                self.imputers[col] = SimpleImputer(
                    missing_values=self.categorical["missing_values"],
                    strategy=self.categorical["strategy"],
                    fill_value=self.categorical["fill_value"],
                )
            elif col in self.num_cols:
                self.imputers[col] = SimpleImputer(
                    missing_values=self.numerical["missing_values"],
                    strategy=self.numerical["strategy"],
                    fill_value=self.numerical["fill_value"],
                )
            else:
                self.imputers[col] = SimpleImputer(
                    missing_values=self.specific_col[col]["missing_values"],
                    strategy=self.specific_col[col]["strategy"],
                    fill_value=self.specific_col[col]["fill_value"],
                )
            df_valid = self._get_df_subset(self.df, [col])
            self.imputers[col].fit(df_valid)

    # -----------------------------------
    def _transform(self, df: pd.DataFrame):
        """
        Transform method complement used specifically for the current class.

        :param df: the full dataset being transformed.
        """
        transf_df = df.copy()
        for col in self.col_impute:
            df_valid = self._get_df_subset(df, [col])
            is_int1 = "int" in df_valid[col].dtype.name
            transf_df[[col]] = self.imputers[col].transform(df_valid)
            is_int2 = "int" in transf_df[col].dtype.name
            if is_int1 and not is_int2:
                transf_df[col] = transf_df[col].apply(int)
        return transf_df
