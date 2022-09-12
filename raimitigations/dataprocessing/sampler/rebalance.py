from typing import Union
import pandas as pd
from imblearn.base import BaseSampler
import numpy as np
from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import TomekLinks

from ..data_processing import DataProcessing
from ..encoder import DataEncoding, EncoderOHE
from ..imputer import DataImputer, BasicImputer
from ..data_utils import get_cat_cols


class Rebalance(DataProcessing):
    """
    Concrete class that uses under and oversampling approaches (implemented
    in the :mod:`imblearn` library) to rebalance a dataset. This class serves as a
    facilitation layer over the :mod:`imblearn` library: it implements several
    automation processes and default parameters, making it easier to rebalance
    a dataset using approaches implemented in the :mod:`imblearn` library.

    :param df: the dataset to be rebalanced, which is used during the fit method.
        This data frame must contain all the features, including the rebalance
        column (specified in the  ``rebalance_col`` parameter). This parameter is
        mandatory if  ``rebalance_col`` is also provided. The user can also provide
        this dataset (along with the  ``rebalance_col``) when calling the :meth:`fit`
        method. If df is provided during the class instantiation, it is not
        necessary to provide it again when calling :meth:`fit`. It is also possible
        to use the  ``X`` and  ``y`` instead of  ``df`` and  ``rebalance_col``, although it is
        mandatory to pass the pair of parameters (X,y) or (df, rebalance_col) either
        during the class instantiation or during the :meth:`fit` method;

    :param rebalance_col: the name or index of the column used to do the rebalance
        operation. This parameter is mandatory if  ``df`` is provided;

    :param X: contains only the features of the original dataset, that is, does
        not contain the column used for rebalancing. This is useful if the user has
        already separated the features from the label column prior to calling this
        class. This parameter is mandatory if  ``y`` is provided;

    :param y: contains only the rebalance column of the original dataset. The rebalance
        operation is executed based on the data distribution of this column. This parameter
        is mandatory if  ``X`` is provided;

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

    :param cat_col: a list of names or indexes of categorical columns. If None, this
        parameter will be set automatically as a list of all categorical variables
        in the dataset. These columns are used to determine the default ``SMOTE`` type that
        should be used: if ``cat_col`` is None, then use ``SMOTE``; if ``cat_col`` represents all
        columns of the dataset, then use ``SMOTEN``; if ``cat_col`` is a subset of columns of
        the dataset, then use ``SMOTENC``. If a specific ``SMOTE`` object is provided in the
        constructor (using the ``over_sampler`` parameter), then the columns in ``cat_col``
        will be automatically encoded using One-Hot encoding (:class:`~raimitigations.dataprocessing.EncoderOHE`), unless
        another encoding transformer is provided in the transform_pipe parameter;

    :param strategy_over: indicates which oversampling strategy should be used. This
        parameter can be a string, a float, or a dictionary, and their meaning are similar
        to what is used by the :mod:`imblearn` library for ``SMOTE`` classes:

            - **Float:** a value between [0, 1] that represents the desired ratio between the
              number of instances of the minority class over the majority class. The ratio 'r'
              is given by: :math:`r = N_m/N_M` where :math:`N_m` is the number of instances of the minority
              class after applying oversample and :math:`N_M` is the number of instances of the
              majority class;
            - **String:** a string value must be one of the following, which identifies preset
              oversampling strategies (explanations retrieved from the :mod:`imblearn`'s SMOTE
              documentation):

                * **'minority':** resample only the minority class;
                * **'not minority':** resample all classes but the minority class;
                * **'not majority':** resample all classes but the majority class;
                * **'all':** resample all classes;
                * **'auto':** equivalent to 'not majority'.

            - Dictionary: the dictionary must have one key for each of the possible classes
              found in the label column, and the value associated to each key represents the
              number of instances desired for that class after the oversampling process is done;

    :param k_neighbors: an integer value representing the number of neighbors that should be
        used when creating the artificial samples using the ``SMOTE`` oversampling. This value is
        only valid if no oversampling object is passed, that is, over_sampler=True;

    :param over_sampler: this parameter can be a boolean value or a sampler object from :mod:`imblearn`:

        - Boolean: if a boolean value is passed, it indicates if the current class should
          use an oversampling method or not. If True, a default ``SMOTE`` is created internally using
          the parameters provided (such as ``k_neighbors``, ``n_jobs``, ``strategy_over``, etc.), where the
          ``SMOTE`` type (``SMOTE``, ``SMOTEN``, ``SMOTENC``) used is determined automatically based on the dataset
          provided: if the dataset contains only numerical data, then SMOTE is used, if the dataset
          contains only categorical features, then ``SMOTEN`` is used, and if the dataset contains
          numerical and categorical data, ``SMOTENC`` is used;
        - BaseSampler object: if the value provided is an object that inherits from :class:`~raimitigations.dataprocessing.BaseSampler`, then
          this sampler is used instead of creating a new sampler. The preprocessing steps
          automatically applied to the dataset changes based on which SMOTE type is passed: if
          the object is ``SMOTE``, then all categorical data is encoded using one-hot encoding
          (:class:`~raimitigations.dataprocessing.EncoderOHE`) and all missing values are imputed using the BasicImputer, but if the object
          is another ``SMOTE`` type (``SMOTEN`` or ``SMOTENC``), then only the imputation preprocessing is
          applied;

    :param strategy_under: similar to strategy_over, but instead specifies the strategy to be used for
        the undersampling approach. This parameter can be a string, a float, or a dictionary, and their
        meaning are similar to what is used by the :mod:`imblearn` library for the ClusterCentroids class:

            - **Float:** a value between [0, 1] that represents the desired ratio between the
              number of instances of the minority class over the majority class after undersampling.
              The ratio 'r' is given by: :math:`r = N_m/N_M` where :math:`N_m` is the number of instances of the
              minority class and :math:`N_M` is the number of instances of the majority class after
              undersampling. Note: this parameter only works with undersampling approaches that allows
              controlling the number of instances to be undersampled, such as :class:`~imblearn.under_sampling.RandomUnderSampler`,
              :class:`~imblearn.under_sampling.ClusterCentroids` (from :mod:`imblearn`). If any other undersampler is provided in the
              ``under_sampler`` parameter along with a float value for the `strategy_under` parameter, an
              error will be raised;
            - **Dictionary:** the dictionary must have one key for each of the possible classes
              found in the label column, and the value associated to each key represents the
              number of instances desired for that class after the undersampling process is done.
              Note: this parameter only works with undersampling approaches that allow
              controlling the number of instances to be undersampled, such as :class:`~imblearn.under_sampling.RandomUnderSampler`,
              :class:`~imblearn.under_sampling.ClusterCentroids` (from :mod:`imblearn`). If any other undersampler is provided in the
              ``under_sampler`` parameter along with a float value for the ``strategy_under`` parameter,
              an error will be raised;
            - **String:** a string value must be one of the following, which identifies preset
              oversampling strategies (explanations retrieved from the :mod:`imblearn`'s :class:`~imblearn.under_sampling.ClusterCentroids`
              documentation):

                * **'majority':** resample only the majority class;
                * **'not minority':** resample all classes but the minority class;
                * **'not majority':** resample all classes but the majority class;
                * **'all':** resample all classes;
                * **'auto':** equivalent to 'not minority';

    :param under_sampler: this parameter can be a boolean value or a sampler object from :mod:`imblearn`:

        - **Boolean:** if a boolean value is passed, it indicates if the current class should
          use an undersampling method or not. If True, a default undersampler is created internally.
          There are two possible default undersamplers that can be created: (i) a :class:`~imblearn.under_sampling.ClusterCentroids`
          is created if the value provided to the ``strategy_under`` parameter is a float value or a
          dictionary (the :class:`~imblearn.under_sampling.ClusterCentroids` allows control over the number of instances that should
          be undersampled), and (ii) a TomekLinks otherwise;
        - **BaseSampler object:** if the value provided is an object that inherits from BaseSampler, then
          this sampler is used instead of creating a new sampler;

    :param n_jobs: the number of workers used to run the sampling methods. This value is only used
        when a default sampler (under or over) is created, where this parameter is provided to the
        ``n_jobs`` parameter of the :mod:`imblearn`'s classes;

    :param verbose: indicates whether internal messages should be printed or not
    """

    VALID_STRATEGY_OVER = ["minority", "not minority", "not majority", "all", "auto"]
    VALID_STRATEGY_UNDER = ["majority", "not minority", "not majority", "all", "auto"]

    SMOTE_TYPE = 0
    SMOTEN_TYPE = 1
    SMOTENC_TYPE = 2

    CONTROLLED_UNDER_CLASSES = [RandomUnderSampler, ClusterCentroids]
    UNDER_CONTROLLED = 0
    UNDER_AUTO = 1

    # -----------------------------------
    def __init__(
        self,
        df: Union[pd.DataFrame, np.ndarray] = None,
        rebalance_col: str = None,
        X: Union[pd.DataFrame, np.ndarray] = None,
        y: Union[pd.DataFrame, np.ndarray] = None,
        transform_pipe: list = None,
        in_place: bool = False,
        cat_col: list = None,
        strategy_over: Union[str, dict, float] = None,
        k_neighbors: int = 4,
        over_sampler: Union[BaseSampler, bool] = True,
        strategy_under: Union[str, dict, float] = None,
        under_sampler: Union[BaseSampler, bool] = False,
        n_jobs: int = 1,
        verbose: bool = True,
    ):
        super().__init__(verbose)
        self.df = None
        self.df_org = None
        self.y = None
        self.in_place = in_place
        self.cat_col = cat_col
        self.transform_pipe = transform_pipe
        self._set_df_mult(df, rebalance_col, X, y)
        self._set_cat_col()
        self.label_encoder = None
        self.strategy_over = strategy_over
        self.strategy_under = strategy_under
        self.over_sampler = over_sampler
        self.under_sampler = under_sampler
        self.k_nn = k_neighbors
        self.default_smote_type = self.SMOTE_TYPE
        self.default_under_type = self.UNDER_AUTO
        self.njobs = n_jobs

    # -----------------------------------
    def _check_rebalance_col(self):
        """
        Checks if the rebalance_col provided contains only integer or string values
        (float values are not allowed). This column is used to determine which classes
        should be rebalanced or not. It can be other columns different from the label
        column, but it needs the be a column containing only non-float values. Float values
        can't be interpreted as classes that should be rebalanced.
        """

        def test_if_float(value):
            if type(value) == float:
                return 1
            return 0

        has_null = self.y.isnull().values.any()
        if has_null:
            raise ValueError(
                f"ERROR: the column {self.label_col_name} provided to the 'rebalance_col' parameter contains "
                + f"null values. The 'rebalance_col' only accepts columns with integer values, which represents the "
                + f"classes to be rebalanced."
            )

        not_int = self.y.apply(test_if_float)
        not_int = np.any(not_int)
        if not_int:
            raise ValueError(
                f"ERROR: the column {self.label_col_name} provided to the 'rebalance_col' parameter contains "
                + f"float values. The 'rebalance_col' only accepts columns with non-float values, which "
                + f"represents the classes to be rebalanced."
            )

    # -----------------------------------
    def _set_cat_col(self):
        """
        Sets the columns to encode (cat_col) automatically
        if these columns are not provided. We consider that
        only categorical columns must be encoded. Therefore, we
        automatically check which columns are possibly categorical.
        """
        if self.cat_col is not None:
            return

        if self.df is None:
            return

        self.cat_col = get_cat_cols(self.df)
        if self.cat_col != []:
            self.print_message(
                f"No categorical columns specified. These columns "
                + f"have been automatically identfied as the following:\n{self.cat_col}"
            )

    # -----------------------------------
    def _check_strategy(
        self, var_name: str, strategy: Union[str, dict, float], valid_values: list, sampler: BaseSampler
    ):
        """
        Checks a given strategy parameter to see if it is valid. Can be called
        for the strategy_under or strategy_over parameters. Checks if the parameter
        value 'strategy' is a string value within the possible values (represented
        by the parameter valid_values) or if it is a float or dictionary. Other value
        types are invalid. If a valid sampler is provided (through the sampler
        parameter), then set the strategy to "auto" if no strategy is provided.

        :param var_name: the name of the parameter used for the variable passed to the
            'strategy' parameter. Can be one of the following: 'strategy_under' or
            'strategy_over';
        :param strategy: the attribute strategy_under or strategy_over;
        :param valid_values: a list of the possible valid string values that the strategy
            parameter can assume. Can be one of the following lists: VALID_STRATEGY_OVER
            or VALID_STRATEGY_UNDER
        :param sampler: the BaseSampler object provided either one of the following
            constructor's parameter: over_sampler or under_sampler.
        """
        if strategy is None:
            if sampler is not False:
                strategy = "auto"
            # else, no sampler of this type is being used
        elif type(strategy) == str and strategy not in valid_values:
            raise ValueError(
                f"ERROR: invalid value {strategy} for variable {var_name}. "
                + f"The only valid string values for {var_name} are: {valid_values}."
            )
        elif type(strategy) != str and type(strategy) != float and type(strategy) != dict:
            raise ValueError(
                f"ERROR: invalid value {strategy} for variable {var_name}. "
                + f"Expected {var_name} to be a float, a string or a dictionary."
            )
        return strategy

    # -----------------------------------
    def _set_default_smote_type(self):
        """
        Identifies which default oversampler object should be created in
        case no specific  object (created before creating this class) is
        provided in the constructor (over_sampler parameter). This method
        doesn't actually create the object, rather it simply sets an internal
        attribute that determines which oversampler should be created. If
        no categorical columns are present in the dataset, then use the
        base SMOTE type. If using an undersampler, set the undersampler to
        be SMOTE, since the existing undersamplers in :mod:`imblearn` can only
        handle numerical data (and SMOTE also only handles numerical data).
        Otherwise, if no undersampler is used and the dataset contains only
        categorical columns, use a SMOTEN object instead of SMOTE. If the
        dataset contains a mixture of numerical and categorical columns, then
        use the SMOTENC object.
        """
        # if no categorical variables, then maintain the
        # default smote type (SMOTE_TYPE)
        if self.cat_col == []:
            return

        # if using an oversampler and the data frame was already provided
        if self.over_sampler is not False and self.strategy_under is None:
            # if using undersampling as well, then all categorical data must
            # be converted beforehand. This way, use the conventional SMOTE
            if self.under_sampler is not False:
                self.default_smote_type = self.SMOTE_TYPE
            # if using only over sampling
            elif self.df is not None:
                # check if all columns in df are categorical.
                # In this case, we need to use SMOTEN
                if set(self.cat_col) == set(self.df.columns):
                    self.default_smote_type = self.SMOTEN_TYPE
                # otherwise, we must use SMOTENC
                else:
                    self.default_smote_type = self.SMOTENC_TYPE

    # -----------------------------------
    def _set_default_under_sample_type(self):
        """
        Identifies which default undersampler object should be created in
        case no specific object (created before creating this class) is
        provided in the constructor (under_sampler parameter). This method
        doesn't actually create the object, rather it simply sets an internal
        attribute that determines which undersampler should be created. This
        attribute is self.default_under_type. If the provided undersampling
        strategy is a float value or a dictionary, this means that the
        undersampler must be allowed to control the number of undersampled
        instances. In this case, self.default_under_type is set as UNDER_CONTROLLED.
        Otherwise, self.default_under_type is set to be UNDER_AUTO, which indicates
        that the undersampler doesn't need to be able to control the number of
        undersampled instances.
        """
        self.default_under_type = self.UNDER_AUTO
        if type(self.strategy_under) == dict or type(self.strategy_under) == float:
            self.default_under_type = self.UNDER_CONTROLLED

    # -----------------------------------
    def _check_valid_samplers(self):
        """
        Check for any inconsistencies between the over and undersampler's options
        and the samplers provided through the over_sampler and under_sampler parameters.
        If any inconsistency is found, an error is raised, along with a proper
        explanation of the problem.
        """
        if self.default_smote_type == self.SMOTE_TYPE:
            if isinstance(self.over_sampler, SMOTEN) or isinstance(self.over_sampler, SMOTENC):
                if self.under_sampler is not False:
                    raise ValueError(
                        "ERROR: Cannot use a SMOTEN or SMOTENC with an under sampler, since the "
                        + "latter requires only numerical data and the former requires categorical "
                        + "data to be present."
                    )
                else:
                    raise ValueError("ERROR: No categorical variables present in the dataframe provided.")
        # we get here only if default_smote_type is SMOTEN or SMOTENC
        elif isinstance(self.over_sampler, BaseSampler):
            # if the df has only categorical data but the provided
            # sampler is SMOTENC (which expects categorical and numerical data)
            if self.default_smote_type == self.SMOTEN_TYPE and isinstance(self.over_sampler, SMOTENC):
                raise ValueError(
                    "ERROR: Expected the provided sampler to be a SMOTEN object, "
                    + "but got a SMOTENC object. The provided dataframe does not contain "
                    + "numerical data, so SMOTENC cannot be used."
                )
            # if the df has categorical and numerical data, but the provided sampler
            # is SMOTEN, which expects only categorical data
            elif self.default_smote_type == self.SMOTENC_TYPE and isinstance(self.over_sampler, SMOTEN):
                raise ValueError(
                    "ERROR: Expected the provided sampler to be a SMOTENC object, "
                    + "but got a SMOTEN object. The provided dataframe contains "
                    + "numerical data, so SMOTEN cannot be used."
                )
            # if we were expecting a SMOTEN or SMOTENC sampler but got a different one
            # that deals with numerical data. In this case, we must encode all categorical
            # data before using the provided sampler
            if not isinstance(self.over_sampler, SMOTENC) and not isinstance(self.over_sampler, SMOTEN):
                self.default_smote_type = self.SMOTE_TYPE

        if isinstance(self.under_sampler, BaseSampler):
            # check if the undersampler should be capable of controlling the number of
            # instances undersampled and if the provided undersampler allows this behavior
            if (
                self.default_under_type == self.UNDER_CONTROLLED
                and self.under_sampler.__class__ not in self.CONTROLLED_UNDER_CLASSES
            ):
                class_names = [cls.__name__ for cls in self.CONTROLLED_UNDER_CLASSES]
                raise ValueError(
                    f"ERROR: Expected 'under_sampler' to be an object of one of the following classes: {class_names}. "
                    f"\nInstead got an object of class {self.under_sampler.__class__.__name__}. If the provided "
                    "'strategy_under' is a dictionary or a float value, than the resulting under sampler must allow these "
                    f"types of strategy. Only controlled under samplers allow to control the number of samples removed."
                )

        use_over = self.over_sampler is True or isinstance(self.over_sampler, BaseSampler)
        use_under = (
            self.under_sampler is True or isinstance(self.under_sampler, BaseSampler) or self.strategy_under is not None
        )
        # check if both under and oversampling are not being used
        if not use_over and not use_under:
            raise ValueError(
                "ERROR: Rebalance class is not using any under nor over sampling method. "
                + "At least one of the two approaches must be provided when instantiating this class."
            )

    # -----------------------------------
    def _check_samplers(self):
        """
        Sets all internal variables that control which under and oversampling methods
        should be used. Also, checks for any inconsistencies in these variables.
        """

        def _invalid_sampler(sampler_name):
            raise ValueError(
                f"ERROR: invalid {sampler_name} value. Expected a BaseSampler object or "
                + f"a boolean value to indicate if {sampler_name} will be used or not."
            )

        if type(self.over_sampler) != bool and not isinstance(self.over_sampler, BaseSampler):
            _invalid_sampler("over_sampler")
        if type(self.under_sampler) != bool and not isinstance(self.under_sampler, BaseSampler):
            _invalid_sampler("under_sampler")

        self._set_default_smote_type()
        self._set_default_under_sample_type()
        self._check_valid_samplers()

    # -----------------------------------
    def _check_inputs(self):
        """
        Checks for any errors in the parameters provided to the constructor
        and raise an error in case any problem is found.
        """
        self.cat_col = self._check_error_col_list(self.df, self.cat_col, "cat_col")

        # check strategy_under and strategy_over
        self.strategy_over = self._check_strategy(
            "strategy_over", self.strategy_over, self.VALID_STRATEGY_OVER, self.over_sampler
        )
        self.strategy_under = self._check_strategy(
            "strategy_under", self.strategy_under, self.VALID_STRATEGY_UNDER, self.under_sampler
        )

        # check k_neighbors
        if type(self.k_nn) != int:
            raise ValueError("ERROR: Expected k_neighbors to be an integer.")

        # check over and undersampler
        self._check_samplers()

    # -----------------------------------
    def _get_preprocessing_requirements(self):
        if self.default_smote_type == self.SMOTE_TYPE:
            requirements = {
                DataImputer: BasicImputer(
                    verbose=self.verbose,
                    numerical={"missing_values": np.nan, "strategy": "most_frequent", "fill_value": None},
                ),
                DataEncoding: EncoderOHE(col_encode=self.cat_col, verbose=self.verbose),
            }
        else:
            requirements = {DataImputer: BasicImputer(verbose=self.verbose)}

        return requirements

    # -----------------------------------
    def _cat_cols2bool(self):
        """
        Returns a list of boolean values. The list has one value for
        each column in the dataset (self.df), and each value indicates
        if that column is categorical (True) or numerical (False).
        """
        cat_col_bool = [False for _ in range(0, self.df.shape[1])]
        for col in self.cat_col:
            index = self.df.columns.get_loc(col)
            cat_col_bool[index] = True
        return cat_col_bool

    # -----------------------------------
    def _set_over_sampler(self):
        """
        Sets the default oversampler object in case no oversampler is provided
        to the over_sampler parameter. This method checks which SMOTE type should
        be created (by checking the parameter default_smote_type, set in the
        _set_default_smote_type method) and create the object using other parameters
        provided in the constructor method.
        """
        if isinstance(self.over_sampler, BaseSampler):
            self.print_message("\nOver Sampler already provided.\n")
            return

        if self.over_sampler is False:
            self.over_sampler = None
        elif self.default_smote_type == self.SMOTE_TYPE:
            self.over_sampler = SMOTE(sampling_strategy=self.strategy_over, k_neighbors=self.k_nn, n_jobs=self.njobs)
        elif self.default_smote_type == self.SMOTENC_TYPE:
            cat_col_bool = self._cat_cols2bool()
            self.over_sampler = SMOTENC(
                sampling_strategy=self.strategy_over,
                categorical_features=cat_col_bool,
                k_neighbors=self.k_nn,
                n_jobs=self.njobs,
            )
        else:
            self.over_sampler = SMOTEN(sampling_strategy=self.strategy_over, k_neighbors=self.k_nn, n_jobs=self.njobs)

    # -----------------------------------
    def _set_under_sampler(self):
        """
        Sets the default undersampler object in case no undersampler is provided
        to the under_sampler parameter. This method checks which undersampler type
        should be created (by checking the parameter default_under_type, set in the
        _set_default_under_sample_type method) and create the object using other
        parameters provided in the constructor method.
        """
        if isinstance(self.under_sampler, BaseSampler):
            return

        # If the user passes a strategy_under value but does not
        # set the under_sampler to True, then set it to True
        if self.under_sampler is False and self.strategy_under is not None:
            self.under_sampler = True

        if self.under_sampler is False:
            self.under_sampler = None
        elif self.default_under_type == self.UNDER_CONTROLLED:
            self.under_sampler = ClusterCentroids(sampling_strategy=self.strategy_under)
        else:
            self.under_sampler = TomekLinks(sampling_strategy=self.strategy_under, n_jobs=self.njobs)

    # -----------------------------------
    def _get_fit_input_type(self):
        return self.FIT_INPUT_XY

    # -----------------------------------
    def fit_resample(
        self,
        X: Union[pd.DataFrame, np.ndarray] = None,
        y: Union[pd.DataFrame, np.ndarray] = None,
        df: Union[pd.DataFrame, np.ndarray] = None,
        rebalance_col: str = None,
    ):
        """
        Runs the over and/or undersampling methods specified by the parameters provided in
        the constructor method. The following steps are performed: (i) set the dataset, (ii)
        set the list of categorical columns in the dataset, (iii) check for errors in the
        inputs provided, (iv) set, fit and apply the transforms in the ``transform_pipe`` (if any),
        (v) set the oversampler to be used, (vi) set the undersampler to be used, (vii) run
        the oversampler, (viii) run the undersampler, and finally (ix) create a new data frame
        with the modified data.

        :param X: contains only the features of the original dataset, that is, does
            not contain the column used for rebalancing. This is useful if the user has
            already separated the features from the label column prior to calling this
            class. This parameter is mandatory if  ``y`` is provided;
        :param y: contains only the rebalance column of the original dataset. The rebalance
            operation is executed based on the data distribution of this column. This parameter
            is mandatory if  ``X`` is provided;
        :param df: the dataset to be rebalanced, which is used during the :meth:`fit` method.
            This data frame must contain all the features, including the rebalance
            column (specified in the  ``rebalance_col`` parameter). This parameter is
            mandatory if  ``rebalance_col`` is also provided. The user can also provide
            this dataset (along with the  ``rebalance_col``) when calling the :meth:`fit`
            method. If ``df`` is provided during the class instantiation, it is not
            necessary to provide it again when calling :meth:`fit`. It is also possible
            to use the  ``X`` and  ``y`` instead of  ``df`` and  ``rebalance_col``, although it is
            mandatory to pass the pair of parameters (X,y) or (df, rebalance_col) either
            during the class instantiation or during the :meth:`fit` method;
        :param rebalance_col: the name or index of the column used to do the rebalance
            operation. This parameter is mandatory if  ``df`` is provided.
        :return: the transformed dataset.
        :rtype: pd.DataFrame or np.ndarray
        """
        self._set_df_mult(df, rebalance_col, X, y, require_set=True)
        self._check_rebalance_col()
        self._set_cat_col()
        self._check_inputs()
        self._set_transforms(self.transform_pipe)
        self._fit_transforms(self.df, self.y)
        self.df = self._apply_transforms(self.df)
        if self.in_place:
            self.df_org = self.df
        self._set_over_sampler()
        self._set_under_sampler()
        X_resample = None
        if self.over_sampler is not None:
            self.print_message("Running oversampling...")
            X_resample, y_resample = self.over_sampler.fit_resample(self.df, self.y)
            self.print_message("...finished")
        if self.under_sampler is not None:
            self.print_message("Running undersampling...")
            if X_resample is None:
                X_resample, y_resample = self.under_sampler.fit_resample(self.df, self.y)
            else:
                X_resample, y_resample = self.under_sampler.fit_resample(X_resample, y_resample)
            self.print_message("...finished")

        return_var = [X_resample, y_resample]
        if self.input_scheme == self.INPUT_DF:
            return_var = pd.concat([X_resample, y_resample], axis=1)

        return return_var
