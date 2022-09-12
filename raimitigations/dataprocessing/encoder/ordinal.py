from typing import Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from .encoder import DataEncoding


class EncoderOrdinal(DataEncoding):
    """
    Concrete class that applies ordinal encoding over a dataset. The categorical features
    are encoded using the ordinal encoding class from``sklearn``. The main difference between
    using the``sklearn`` implementation directly is that we allow the user to pass a list of
    columns to be encoded when creating the class, instead of having to explicitly use
    the ``sklearn`` class over each column individually.

    :param df: pandas data frame that contains the columns to be encoded;

    :param col_encode: a list of the column names or indexes that will be encoded.
        If None, this parameter will be set automatically as being a list of all
        categorical variables in the dataset;

    :param categories: can be a dict or a string:

        - **dict:** a dict that indicates the order of the values for each column in ``col_encode``.
          That is, a dict of lists, where the keys must be valid column names and the
          value associated to key k is a list of size n, where n is the number of
          different values that exist in column k, and this list represents the order of
          the encoding for that column;
        - **string:** the only string value allowed is "auto". When categories = "auto", the
          categories dict used by``sklearn`` is generated automatically;

    :param unknown_err: if True, an error will occur when the transform method is called upon a
        dataset with a new category in one of the encoded columns that were not present in
        the training dataset (provided to the :meth:`fit` method). If False, no error will occur in
        the previous situation. Instead, every unknown category in a given encoded column
        will be replaced by the label unknown_value;

    :param unknown_value: the value used when an unknown category is found in one of the encoded
        columns. This parameter must be different than the other labels already used by the
        column(s) with unknown values. We recommend using negative values to avoid conflicts;

    :param verbose: indicates whether internal messages should be printed or not.
    """

    UNK_VALUES = ["UNKNOWN", "_unknown_", "UNK", "?"]

    # -----------------------------------
    def __init__(
        self,
        df: Union[pd.DataFrame, np.ndarray] = None,
        col_encode: list = None,
        categories: Union[dict, str] = "auto",
        unknown_err: bool = False,
        unknown_value: Union[int, float] = -1,
        verbose: bool = True,
    ):
        super().__init__(df, col_encode, verbose)
        self.categories = categories
        self.unknown_err = unknown_err
        self.handle_err = "use_encoded_value"
        if unknown_err:
            self.handle_err = "error"
        self.unknown_value = unknown_value

    # -----------------------------------
    def _check_categories(self):
        """
        Checks if the value provided to the categories parameter is valid.
        If it is not valid, an appropriate ValueError is raised informing
        the user of a possible explanation for the error.
        """
        if self.categories == "auto":
            return

        error = False
        if type(self.categories) != dict:
            error = True

        if not error:
            for key in self.categories.keys():
                # check if the value associated to the current key is a list or a numpy array
                is_list = type(self.categories[key]) == list
                is_np = type(self.categories[key]) == np.ndarray and len(self.categories[key].shape) == 1
                if not is_list and not is_np:
                    error = True

                # check if the column name provided is valid and exists in the dataset
                if type(key) == int:
                    if self.column_type == self.COL_NAME:
                        new_key = self._get_column_from_index(key)
                    else:
                        new_key = str(key)
                    if new_key not in self.df.columns:
                        raise ValueError(
                            f"ERROR: the key '{key}' from the categories parameter is not a valid "
                            + f"column index in the dataset provided."
                        )
                elif key not in self.df.columns:
                    raise ValueError(
                        f"ERROR: the key '{key}' from the categories parameter is not a valid "
                        + f"column name in the dataset provided."
                    )

                # check if the values provided to the current key are
                # actually in the unique values of the desired column
                subset = self._get_df_subset(self.df, [key])
                unique = subset.iloc[:, 0].unique()
                for value in self.categories[key]:
                    if value not in unique:
                        raise ValueError(
                            f"ERROR: the value '{value}' provided to the the list of values for the key '{key}' in the 'categories' "
                            + f"parameter does not match any of the unique values found in the column '{key}' of the dataset provided."
                        )

        if error:
            raise ValueError(
                "ERROR: the 'categories' parameter must be a dictionary, with the same length "
                + "as the 'col_encode' parameter. Each index i of this list must be another list specifying "
                + "the order of the existing values of the column col_encode[i]. If a value is not given, "
                + "it will be assigned a None value."
            )

    # -----------------------------------
    def _auto_complete_categories(self, df: pd.DataFrame):
        """
        Generates the 'categories' parameter used by OrdinalEncoder (from``sklearn``)
        based on the value provided to the categories parameter. If a categorical
        column in col_encode is missing from the categories parameter, it is auto-
        matically generated.

        :param df: the full dataset with the columns to be encoded.
        """
        columns = df.columns
        self.categories_list = []
        for col in columns:
            col_in_categories = type(self.categories) == str or col not in self.categories.keys()
            if type(self.categories) == str or col_in_categories:
                unique = self._get_df_subset(df, [col]).iloc[:, 0].unique()
                strList = [x for x in unique if isinstance(x, str)]
                strList = sorted(strList)
                numList = [x for x in unique if x not in strList and not np.isnan(x)]
                numList = list(np.sort(numList))
                unique = strList + numList
                self.categories_list.append(unique)
            else:
                self.categories_list.append(self.categories[col])

    # -----------------------------------
    def _create_ordinal_encoder(self):
        if self.handle_err == "use_encoded_value":
            self.encoder = OrdinalEncoder(
                categories=self.categories_list, handle_unknown=self.handle_err, unknown_value=self.unknown_value
            )
        else:
            self.encoder = OrdinalEncoder(categories=self.categories_list, handle_unknown=self.handle_err)

    # -----------------------------------
    def _get_unknown_value(self, category_index: int):
        values = self.categories_list[category_index]
        unk_value = None
        for val in self.UNK_VALUES:
            if val not in values:
                unk_value = val
                break

        if unk_value is None:
            count = 0.0
            while unk_value is None:
                new_value = f"{self.UNK_VALUES[0]}_{count}"
                count += 1
                if new_value not in values:
                    unk_value = new_value

        return unk_value

    # -----------------------------------
    def _create_mapping_dict(self):
        mapping = {}
        for i, col in enumerate(self.col_encode):
            mapping[col] = {}
            mapping[col]["values"] = list(self.categories_list[i].copy())
            mapping[col]["labels"] = [j for j in range(len(list(self.categories_list[i])))]
            if not self.unknown_err:
                unk_value = self._get_unknown_value(i)
                mapping[col]["values"].append(unk_value)
                mapping[col]["labels"].append(self.unknown_value)
            mapping[col]["n_labels"] = len(mapping[col]["labels"])
        self.mapping = mapping

    # -----------------------------------
    def _fit(self):
        """
        Steps for running the fit method for the current class. Starts by
        (i) setting the dataset (self.df), (ii) checking if the categories
        parameter is valid, (iii) generate the categories_list, required
        by the OrdinalEncoding class, followed by (iv) the creation of the
        OrdinalEncoding object, and (v) the call to the fit method of the
        OrdinalEncoding object.
        """
        df_valid = self._get_df_subset(self.df, self.col_encode)
        self._check_categories()
        self._auto_complete_categories(df_valid)
        self._create_ordinal_encoder()
        self.encoder.fit(df_valid)
        self._create_mapping_dict()

    # -----------------------------------
    def _transform(self, df: pd.DataFrame):
        """
        Steps for running the transform method for the current class.

        :param df: the full dataset with the columns to be encoded.
        """

        def _to_int(value):
            if np.isnan(value):
                return value
            return int(value)

        transf_df = df.copy()
        df_valid = self._get_df_subset(df, self.col_encode)
        transf_df[self.col_encode] = self.encoder.transform(df_valid)
        for col in self.col_encode:
            transf_df[col] = transf_df[col].apply(_to_int)
        return transf_df

    # -----------------------------------
    def _inverse_transform(self, df: Union[pd.DataFrame, np.ndarray]):
        self._check_if_fitted()
        transf_df = df.copy()
        df_valid = self._get_df_subset(df, self.col_encode)
        transf_df[self.col_encode] = self.encoder.inverse_transform(df_valid)
        return transf_df

    # -----------------------------------
    def get_mapping(self):
        """
        Returns a dictionary with all the information regarding the mapping
        performed by the ordinal encoder. The dictionary contains the following
        structure:

        * One key for each column. Each key is associated with a secondary
          dictionary with the following keys:

          - **"values":** the unique values encountered in the column;
          - **"labels":** the labels assigned to each of the unique values.
            the list from the "values" key is aligned with this
            list, that is, mapping[column]["labels][i] is the
            label assigned to the value mapping[column]["values][i].
          - **"n_labels":** the number of labels. If unknown_err is set to False,
            this will account for the label for unknown values.

        :return: a dictionary with all the information regarding the mapping
            performed by the ordinal encoder.
        :rtype: dict
        """
        return self.mapping.copy()
