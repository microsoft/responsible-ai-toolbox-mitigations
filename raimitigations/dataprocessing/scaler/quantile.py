from typing import Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from .scaler import DataScaler


class DataQuantileTransformer(DataScaler):
    """
    Concrete class that applies the :class:`~sklearn.preprocessing.QuantileTransformer` scaler over a given dataset.
    This class uses the :mod:`sklearn`'s implementation of this scaler (:class:`~sklearn.preprocessing.QuantileTransformer`)
    at its root, but also makes it more simple to be applied to a dataset. For example,
    the user can use a dataset with categorical columns and the scaler will be applied
    only to the numerical columns. Also, the user can provide a pipeline of scalers, and
    all of the scalers in the pipeline will be applied before the :class:`~sklearn.preprocessing.QuantileTransformer`
    scaler. The user can also use a list of transformations using other non-scaler
    classes implemented in this library (feature selection, encoding, imputation, etc.).
    For more details on how the :class:`~sklearn.preprocessing.QuantileTransformer` changes the data, check the official
    documentation from :mod:`sklearn`:
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html

    :param scaler_obj: an object from the :class:`~sklearn.preprocessing.QuantileTransformer` class.
        This :mod:`sklearn` scaler will be used to perform the scaling process. If None, a
        :class:`~sklearn.preprocessing.QuantileTransformer` is created using default values;

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
        scaler_obj: QuantileTransformer = None,
        df: Union[pd.DataFrame, np.ndarray] = None,
        exclude_cols: list = None,
        include_cols: list = None,
        transform_pipe: list = None,
        verbose: bool = True,
    ):
        super().__init__(df, exclude_cols, include_cols, transform_pipe, verbose)
        self._set_scaler(scaler_obj)

    # -----------------------------------
    def _set_scaler(self, scaler_obj: QuantileTransformer):
        if scaler_obj is None:
            scaler_obj = QuantileTransformer()
        if not isinstance(scaler_obj, QuantileTransformer):
            raise ValueError(self._get_wrong_obj_error("DataQuantileTransformer", "QuantileTransformer"))
        self.scaler = scaler_obj

    # -----------------------------------
    def _inverse_transform(self, df: pd.DataFrame):
        return self._base_inverse_transform(df)
