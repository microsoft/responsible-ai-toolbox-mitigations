from abc import abstractmethod
from typing import Union

import numpy as np
import pandas as pd

from ..data_processing import DataProcessing
from ..encoder import DataEncoding, EncoderOrdinal
from ..imputer import DataImputer, BasicImputer


class FeatureSelection(DataProcessing):
    """
    Base class for all feature selection subclasses. Implements basic
    functionalities that can be used by all feature selection approaches.

    :param df: the data frame to be used during the fit method.
        This data frame must contain all the features, including the label
        column (specified in the  ``label_col`` parameter). This parameter is
        mandatory if  ``label_col`` is also provided. The user can also provide
        this dataset (along with the  ``label_col``) when calling the:meth:`fit`
        method. If ``df`` is provided during the class instantiation, it is not
        necessary to provide it again when calling:meth:`fit`. It is also possible
        to use the  ``X`` and  ``y`` instead of  ``df`` and  ``label_col``, although it is
        mandatory to pass the pair of parameters (X,y) or (df, label_col) either
        during the class instantiation or during the:meth:`fit` method;

    :param label_col: the name or index of the label column. This parameter is
        mandatory if  ``df`` is provided;

    :param X: contains only the features of the original dataset, that
        is, does not contain the label column. This is useful if the user has
        already separated the features from the label column prior to calling this
        class. This parameter is mandatory if  ``y`` is provided;

    :param y: contains only the label column of the original dataset.
        This parameter is mandatory if  ``X`` is provided;

    :param transform_pipe: a list of transformations to be used as a pre-processing
        pipeline. Each transformation in this list must be a valid subclass of the
        current library (:class:`~raimitigations.dataprocessing.EncoderOrdinal`, :class:`~raimitigations.dataprocessing.BasicImputer`, etc.). Some feature selection
        methods require a dataset with no categorical features or with no missing values
        (depending on the approach). If no transformations are provided, a set of default
        transformations will be used, which depends on the feature selection approach
        (subclass dependent);

    :param in_place: indicates if the original dataset will be saved internally (``df_org``)
        or not. If True, then the feature selection transformation is saved over the
        original dataset. If False, the original dataset is saved separately (default
        value);

    :param verbose: indicates whether internal messages should be printed or not.
    """

    # -----------------------------------
    def __init__(
        self,
        df: Union[pd.DataFrame, np.ndarray] = None,
        label_col: str = None,
        X: Union[pd.DataFrame, np.ndarray] = None,
        y: Union[pd.DataFrame, np.ndarray] = None,
        transform_pipe: list = None,
        in_place: bool = False,
        verbose: bool = True,
    ):
        super().__init__(verbose)
        self.df = None
        self.df_org = None
        self.y = None
        self.in_place = in_place
        self.fitted = False
        self._set_df_mult(df, label_col, X, y)
        self._set_transforms(transform_pipe)

    # -----------------------------------
    def _get_preprocessing_requirements(self):
        """
        Several feature selection approaches require a dataset containing only numerical
        features and without any missing values. Therefore, by default, any concrete
        feature selection class that inherits from the current abstract class must
        have a ``self.transform_pipe`` with at least one object that inherits from :class:`~raimitigations.dataprocessing.DataImputer`
        and one object that inherits from DataEncoding. The default objects created in
        case one of the requirements is not met are :class:`~raimitigations.dataprocessing.BasicImputer` and :class:`~raimitigations.dataprocessing.EncoderOrdinal`
        respectively.
        """
        requirements = {
            DataImputer: BasicImputer(verbose=self.verbose),
            DataEncoding: EncoderOrdinal(verbose=self.verbose),
        }
        return requirements

    # -----------------------------------
    def _get_fit_input_type(self):
        return self.FIT_INPUT_XY

    # -----------------------------------
    @abstractmethod
    def _get_selected_features(self):
        """
        Abstract method. For a given concrete class, this method must return a list
        of column names or column indexes chosen by the feature selection method
        implemented.
        """
        pass

    # -----------------------------------
    def set_selected_features(self, selected: list = None):
        """
        Sets the ``self.selected_feat`` attribute, which indicates the currently selected
        features. Receives a list of column names that should be set as the currently
        selected features. If this list is None, then the features selected by the
        feature selection method (implemented by a concrete class that inherits
        from the current class) are used instead. This method is meant to be used
        from the outside by the user (not a private method), allowing the user to
        manually set the features they want to select in case they disagree with
        features selected by the feature selection method. Before setting the
        self.selected_feat attribute, it is checked if the provided list of features
        are all within the dataset provided for the fit method. If one of the
        features in the list is not present in the dataset, a ValueError is raised.

        :param selected: a list of column names or indexes representing the new
            selected features. If None, the features selected by the feature
            selection method are used instead.
        """
        if selected is None:
            selected = self._get_selected_features()
        if self.df is not None:
            selected = self._check_error_col_list(self.df, selected, "selected")
        self.selected_feat = selected

    # -----------------------------------
    @abstractmethod
    def _fit(self):
        """
        Abstract method. For a given concrete class, this method must execute the
        feature selection method and save the selected features and any other
        important information in a set of class-specific attributes. These
        attributes are then used by the _get_selected_features method to retrieve
        the list of selected features.
        """
        pass

    # -----------------------------------
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray] = None,
        y: Union[pd.Series, np.ndarray] = None,
        df: Union[pd.DataFrame, np.ndarray] = None,
        label_col: str = None,
    ):
        """
        Default fit method for all feature selection classes that inherit from
        the current class. The following steps are executed: (i) set the ``self.df``
        and ``self.y`` attributes, (ii) set the transform list (or create a default
        one if needed), (iii) fit and then apply the transformations in the
        ``self.transform_pipe`` attribute to the dataset, (iv) call the concrete
        class's specific :meth:`_fit` method, and (v) set the ``self.selected_feat`` attribute.

        :param X: contains only the features of the original dataset, that is, does not
            contain the label column;
        :param y: contains only the label column of the original dataset;
        :param df: the full dataset;
        :param label_col: the name or index of the label column;

        Check the documentation of the _set_df_mult method (DataProcessing class)
        for more information on how these parameters work.
        """
        self._set_df_mult(df, label_col, X, y, require_set=True)
        self._set_transforms(self.transform_pipe)
        self._fit_transforms(self.df, self.y)
        self.df = self._apply_transforms(self.df)
        if self.in_place:
            self.df_org = self.df
        self._fit()
        self.set_selected_features()
        self.fitted = True
        return self

    # -----------------------------------
    def get_selected_features(self):
        """
        Public method that returns the list of the selected features.
        The difference between this method and _get_selected_features
        is that the latter returns the list of features selected by the
        feature selection method, wheres the current method returns the
        list of selected features currently assigned to self.selected_feat,
        which can be manually changed by the user.

        :return: list containing the name of indices of the currently
            selected features.
        :rtype: list
        """
        return self.selected_feat.copy()

    # -----------------------------------
    def transform(self, df: Union[pd.DataFrame, np.ndarray]):
        """
        Default behavior for transforming the data for the different
        feature selection methods. If a concrete class requires a
        different behavior, just override this method.

        :param df: the dataset used for inference.
        :return: the transformed dataset.
        :rtype: pd.DataFrame or np.ndarray
        """
        self._check_if_fitted()
        df = self._fix_col_transform(df)
        df = self._apply_transforms(df)
        features = self.get_selected_features()
        if self.label_col_name is not None and self.label_col_name in df.columns:
            features.append(self.label_col_name)
        new_df = df[features]
        return new_df
