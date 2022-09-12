from abc import abstractmethod
from typing import Union

import numpy as np
import pandas as pd

from ..data_processing import DataProcessing
from ..data_utils import get_cat_cols


class DataScaler(DataProcessing):
    """
    Base class used by all data scaling or data transformation classes.
    Implements basic functionalities that can be used by different
    scaling or transformation class.

    :param df: pandas data frame that contains the columns to be scaled
        and/or transformed;

    :param exclude_cols: a list of the column names or indexes that shouldn't be
        transformed, that is, a list of columns to be ignored. This way, it is
        possible to transform only certain columns and leave other columns
        unmodified. This is useful if the dataset contains a set of binary columns
        that should be left as is, or a set of categorical columns (which can't
        be scaled or transformed). If the categorical columns are not added in this
        list (``exclude_cols``), the categorical columns will be automatically identified
        and added into the ``exclude_cols`` list. If None, this parameter will be set
        automatically as being a list of all categorical variables in the dataset;

    :param include_cols:  list of the column names or indexes that should be
        transformed, that is, a list of columns to be included in the dataset being
        transformed. This parameter uses an inverse logic from the ``exclude_cols``, and
        thus these two parameters shouldn't be used at the same time. The user must
        used either the ``include_cols``, or the ``exclude_cols``, or neither of them;

    :param transform_pipe: a list of transformations to be used as a pre-processing
        pipeline. Each transformation in this list must be a valid subclass of the
        current library (:class:`~raimitigations.dataprocessing.EncoderOrdinal`, :class:`~raimitigations.dataprocessing.BasicImputer`, etc.). Some feature selection
        methods require a dataset with no categorical features or with no missing values
        (depending on the approach). If no transformations are provided, a set of default
        transformations will be used, which depends on the feature selection approach
        (subclass dependent). This parameter also accepts other scalers in the list. When
        this happens and the :meth:`inverse_transform` method of self is called, the
        :meth:`inverse_transform` method of all scaler objects that appear in the ``transform_pipe``
        list after the last non-scaler object are called in a reversed order. For example,
        if :class:`~raimitigations.datapreprocessing.DataMinMaxScaler. is instantiated with transform_pipe=[BasicImputer(), DataQuantileTransformer(),
        EncoderOHE(), DataPowerTransformer()], then, when calling :meth:`fit` on the
        ``DataMinMaxScaler`` object, first the dataset will be fitted and transformed using
        BasicImputer, followed by DataQuantileTransformer, EncoderOHE, and DataPowerTransformer,
        and only then it will be fitted and transformed using the current DataMinMaxScaler.
        The :meth:`transform` method works in a similar way, the difference being that it doesn't call
        :meth:`fit` for the data scaler in the ``transform_pipe``. For the :meth:`inverse_transform`
        method, the inverse transforms are applied in reverse order, but only the scaler objects
        that appear after the last non-scaler object in the ``transform_pipe``: first, we inverse the
        ``DataMinMaxScaler``, followed by the inversion of the ``DataPowerTransformer``. The ``DataQuantileTransformer``
        isn't reversed because it appears between two non-scaler objects: ``BasicImputer`` and ``EncoderOHE``;

    :param verbose: indicates whether internal messages should be printed or not.
    """

    # -----------------------------------
    def __init__(
        self,
        df: Union[pd.DataFrame, np.ndarray] = None,
        exclude_cols: list = None,
        include_cols: list = None,
        transform_pipe: list = None,
        verbose: bool = True,
    ):
        super().__init__(verbose)
        self.df = None
        self.fitted = False
        self._set_df(df)
        self.exclude_cols = exclude_cols
        self.include_cols = include_cols
        self._set_transforms(transform_pipe)

    # -----------------------------------
    def _set_col_filters(self):
        """
        Sets or complement the list of columns to be ignored. First, get the list of all
        categorical columns and add all categorical columns not in exclude_cols to the
        exclude_cols list. Then, define the scale_col attribute, which specifies all columns
        that will be scaled or transformed as being, created based on the list of all
        columns in the dataset and the list of columns that should be ignored (exclude_cols).
        """
        # checks if include_cols was provided
        filters_err_msg = None
        if self.include_cols is not None:
            # don't allow the user to use include_cols and exclude_cols at the same time
            if self.exclude_cols is not None:
                raise ValueError(
                    "ERROR: include_cols and exclude_cols can't be used at the same time. Use only "
                    + "either one of these two parameters."
                )
            if self.include_cols != []:
                # fill the exclude_cols parameter with all column names or indices
                # not present in the parameter include_cols
                self.exclude_cols = [col for col in self.df.columns if col not in self.include_cols]
                filters_err_msg = f"include_cols = {self.include_cols}"

        cat_col = get_cat_cols(self.df)
        if self.exclude_cols is None:
            self.exclude_cols = cat_col
        else:
            for col in self.df.columns:
                if col in cat_col and col not in self.exclude_cols:
                    self.exclude_cols.append(col)

        if filters_err_msg is None:
            filters_err_msg = f"exclude_cols = {self.exclude_cols}"

        self.scale_col = [col for col in self.df.columns if col not in self.exclude_cols]
        if self.scale_col == []:
            raise ValueError(
                f"ERROR: the scaler {type(self).__name__} isn't being applied to any numeric column. There must be "
                + f"at least one numeric column in the dataset after filtering all the columns in 'include_cols' and "
                + f"removing the columns from 'ignore_cols'. Check if the resulting dataset (after filtering) or the "
                + f"original dataset has at least one numeric column. The filters used for this scaler are:"
                + f"\n\t{filters_err_msg}"
                + f"\n\tcategorical columns identified = {cat_col}"
            )

    # -----------------------------------
    def _check_error_col_filters(self):
        """
        Checks for any errors or inconsistencies in the exclude_cols parameter, which is
        provided in the constructor method. Also checks if all columns in exclude_cols
        are present in the dataset.
        """
        if self.exclude_cols is not None:
            self.exclude_cols = self._check_error_col_list(self.df, self.exclude_cols, "exclude_cols")
        if self.include_cols is not None:
            self.include_cols = self._check_error_col_list(self.df, self.include_cols, "include_cols")

    # -----------------------------------
    def _get_fit_input_type(self):
        return self.FIT_INPUT_DF

    # -----------------------------------
    def _get_preprocessing_requirements(self):
        requirements = {}
        return requirements

    # -----------------------------------
    @abstractmethod
    def _set_scaler(self):
        """
        Creates a default or sets the scaler or transformer object that
        will be used to transform the data. This method depends on the
        type of scaler/transformer used, and therefore must be implemented
        and called by each concrete class.
        """
        pass

    # -----------------------------------
    def _fit(self, df: pd.DataFrame):
        """
        The specific operations required by a given scaler/transformer
        when the fit() method is called. This method implements a basic
        behavior that is used by most scalers/transformers, which is to
        simply call the fit() method of the particular sklearn scaler
        used. If a given scaler requires a different set of operations,
        it should simply overwrite this method.

        :param df: the full dataset containing the columns that should be scaled.
        """
        self.scaler.fit(df)

    # -----------------------------------
    def _transform(self, df: pd.DataFrame):
        """
        Similar to the _fit() method, this method implements a basic
        behavior used by most scalers when the transform() method is
        called. Here, we simply call the transform() method of the underlying
        sklearn object used. If a given scaler requires a different set
        of operations, it should simply overwrite this method.

        :param df: the full dataset containing the columns that should be scaled
        """
        scaled_df = self.scaler.transform(df)
        return scaled_df

    # -----------------------------------
    def fit(self, df: Union[pd.DataFrame, np.ndarray] = None, y: Union[pd.Series, np.ndarray] = None):
        """
        Fit method used by all concrete scalers that inherit from the
        current abstract DataScaler class. This method executes the following
        steps: (i) set the dataframe (if not already set in the constructor),
        (ii) check for errors or inconsistencies in some in the exclude_cols
        parameter and in the provided dataset, (iii) set the tranforms list,
        (iv) call the fit and transform methods of other transformations
        passed through the transform_pipe parameter (such as imputation, feature
        selection, other scalers, etc.), (v) remove the columns that should
        be ignored according the exclude_cols or include_cols, (vi) call the
        _fit() method to effectively fit the current scaler on the preprocessed
        dataset.

        :param df: the full dataset containing the columns that should be scaled;
        :param y: ignored. This exists for compatibility with the :mod:`sklearn`'s Pipeline class.
        """
        self._set_df(df, require_set=True)
        self._check_error_col_filters()
        self._set_col_filters()
        self._set_transforms(self.transform_pipe)
        self._fit_transforms(self.df)
        transf_df = self._apply_transforms(self.df)
        subset_df = self._get_df_subset(transf_df, self.scale_col)
        self._fit(subset_df)
        self.fitted = True
        return self

    # -----------------------------------
    def transform(self, df: Union[pd.DataFrame, np.ndarray]):
        """
        Transform method used by all concrete scalers that inherit from
        the current abstract DataScaler class. First, check if all columns
        that should be ignored are present in the dataset, which is useful
        to check the consistency of the dataset with the one used during fit
        and transform. In the sequence, apply all the transforms in the
        transform_pipe parameter. Finally, call the transform() method of the
        current scaler over the preprocessed dataset and return a new dataset.

        :param df: the full dataset containing the columns that should be scaled.
        :return: the transformed dataset.
        :rtype: pd.DataFrame or np.ndarray
        """
        self._check_if_fitted()
        df = self._fix_col_transform(df)
        final_df = df.copy()
        final_df = self._apply_transforms(final_df)
        subset_df = self._get_df_subset(final_df, self.scale_col)
        transf_df = self._transform(subset_df)
        transf_df = pd.DataFrame(transf_df, columns=subset_df.columns)

        for col in transf_df.columns:
            final_df[col] = transf_df[col].values.tolist()

        return final_df

    # -----------------------------------
    def _base_inverse_transform(self, df: Union[pd.DataFrame, np.ndarray]):
        """
        Implements the specific behavior for the inverse transformation for
        a given scaler that inherits from the current abstract DataScaler
        class. This method implements a basic behavior, but if a given
        scaler requires a different set of operations to invert the scaling,
        then it should simply overwrite this method. The following steps are
        executed: (i) filter the dataset to contain only the columns that
        should be scaled, (ii) call the inverse_transform() method of the
        sklearn scaler, (iii) create a new dataset containing the ignored
        columns and the scaled columns properly reversed to their original
        values.

        :param df: the dataframe that should be scaled. This
            dataset contains only the columns in the self.scale_col
            attribute. The other columns (in the self.exclude_cols parameter)
            have already been removed in the more general
            inverse_transform() method.
        """
        subset_df = self._get_df_subset(df, self.scale_col)
        columns = subset_df.columns
        subset_df = self.scaler.inverse_transform(subset_df)
        subset_df = pd.DataFrame(subset_df, columns=columns)
        for col in subset_df.columns:
            df[col] = subset_df[col].values.tolist()
        return df

    # -----------------------------------
    def _get_wrong_obj_error(self, own_class: str, outside_class: str):
        """
        Method used to generate a default error message that can then be used
        by any scaler class that inherits from DataScaler. Raises an error
        indicating that one of the parameters passed to the concrete class
        'own_class' should be an object from class 'outside_class'.

        :param own_class (str): the name of the concrete class that inherits from
            the current abstract DataScaler class;
        :param outside_class (str): the name of the class that one of the
            parameters provided to the 'own_class' constructor should belong to.
        """
        err_msg = (
            f"ERROR: the object provided to the {own_class} class through the "
            f"scaler_obj parameter must be from the {outside_class} class."
        )
        return err_msg
