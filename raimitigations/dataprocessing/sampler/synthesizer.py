import os
from typing import Union

import numpy as np
import pandas as pd
from sdv.tabular.base import BaseTabularModel
from sdv.tabular import GaussianCopula, CTGAN, CopulaGAN, TVAE
from sdv.sampling import Condition

from ..data_processing import DataProcessing


class Synthesizer(DataProcessing):
    """
    Concrete class that uses generative models (implemented in the :mod:`sdv` library)
    to create synthetic data for imbalanced datasets. This class allows the creation
    of synthetic data according to a set of conditions specified by the user or
    according to a predefined strategies based on the minority and majority
    classes (both considering the label column only).

    :param df: the dataset to be rebalanced, which is used during the :meth:`fit` method.
        This data frame must contain all the features, including the label
        column (specified in the  ``label_col`` parameter). This parameter is
        mandatory if  ``label_col`` is also provided. The user can also provide
        this dataset (along with the  ``label_col``) when calling the :meth:`fit`
        method. If df is provided during the class instantiation, it is not
        necessary to provide it again when calling :meth:`fit`. It is also possible
        to use the  ``X`` and  ``y`` instead of  ``df`` and  ``label_col``, although it is
        mandatory to pass the pair of parameters (X,y) or (df, label_col) either
        during the class instantiation or during the :meth:`fit` method;

    :param label_col: the name or index of the label column. This parameter is
        mandatory if  ``df`` is provided;

    :param X: contains only the features of the original dataset, that is, does
        not contain the label column. This is useful if the user has already
        separated the features from the label column prior to calling this class.
        This parameter is mandatory if  ``y`` is provided;

    :param y: contains only the label column of the original dataset. This parameter
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

    :param model: the model that should be used to generate the synthetic instances. Can
        be a string or an object that inherits from :class:`sdv.tabular.base.BaseTabularModel`:

            - **BaseTabularModel:** an object from one of the following classes: :class:`~sdv.tabular.ctgan.CTGAN`, :class:`~sdv.tabular.ctgan.TVAE`,
              :class:`~sdv.tabular.copulas.GaussianCopula, or :class:`~sdv.tabular.copulagan.CopulaGAN`, all from the :class:`sdv.tabular` module;
            - **str:** a string that identifies which base model should be created. The base
              models supported are: :class:`~sdv.tabular.ctgan.CTGAN`, :class:`~sdv.tabular.ctgan.TVAE`, :class:`~sdv.tabular.copulas.GaussianCopula`, and :class:`~sdv.tabular.copulagan.CopulaGAN`. The string
              values allowed associated to each of the previous models are (respectively):
              "ctgan", "tvae", "copula", and "copula_gan";

    :param epochs: the number of epochs that the model (specified by the ``model`` parameter)
        should be trained. This parameter is not used when the selected model is from the
        class GaussianCopula;

    :param save_file: the name of the file containing the data of the trained model. After
        training the model (when calling :meth:`fit`), the model's weights are saved in
        the path specified by this parameter, which can then be loaded and reused for future
        use. This is useful when training over a large dataset since this results in an
        extended training time. If the provided value is None, then a default name will be
        created based on the model's type and number of epochs. If ``load_existing`` is True,
        then this parameter will indicate which save file should be loaded;

    :param load_existing: a boolean value indicating if the model should be loaded or not.
        If False, a new save file will be created (or overwritten if the file specified in
        ``save_file`` already exists) containing the model's wights of a new model. If True, the
        model will be loaded from the file ``save_file``;

    :param verbose: indicates whether internal messages should be printed or not
    """

    VALID_STRATEGY = ["minority", "not majority", "auto"]

    VALID_MODELS = ["ctgan", "copula", "copula_gan", "tvae"]
    DEFAULT_EPOCHS = 50

    # -----------------------------------
    def __init__(
        self,
        df: pd.DataFrame = None,
        label_col: str = None,
        X: pd.DataFrame = None,
        y: pd.DataFrame = None,
        transform_pipe: list = None,
        in_place: bool = False,
        model: Union[BaseTabularModel, str] = "ctgan",
        epochs: int = DEFAULT_EPOCHS,
        save_file: str = None,
        load_existing: bool = True,
        verbose: bool = True,
    ):
        super().__init__(verbose)
        self.df = None
        self.df_org = None
        self.y = None
        self.in_place = in_place
        self.fitted = False
        self.transform_pipe = transform_pipe
        self.model = model
        self.epochs = epochs
        self.load_existing = load_existing
        self._set_df_mult(df, label_col, X, y)
        self._set_model()
        self._set_save_file(save_file)

    # -----------------------------------
    def _set_df_mult(
        self, df: pd.DataFrame, label_col: str, X: pd.DataFrame, y: pd.DataFrame, require_set: bool = False
    ):
        """
        Overwrites the _set_df_mult from the BaseClass. Here the label column is added to
        the self.df dataset.

        :param df: the full dataset;
        :param label_col: the name or index of the label column;
        :param X: contains only the features of the original dataset, that is, does not
            contain the label column;
        :param y: contains only the label column of the original dataset;
        :param require_set: if True, a ValueError will be raised if both pairs of
            variables ((df, label_col) and (X, y)) are all None.
        """
        if label_col is not None and type(label_col) == int:
            raise ValueError(
                "ERROR: the class Synthesizer only accepts data frames with proper column names. "
                + "The provided label_col parameter is an integer. Please provide the column name as string."
            )
        super()._set_df_mult(df, label_col, X, y, require_set)
        if self.df is not None:
            self.df = self.df.copy()
            self.df[self.label_col_name] = self.y

    # -----------------------------------
    def _set_save_file(self, save_file: str):
        """
        Set the save_file attribute using the save_file provided in the constructor
        method. If the provided value is None, create a default save_file name.

        :param save_file: the name of the file where the model's wights should be
            saved.
        """
        if save_file is None:
            save_file = f"{type(self.model).__name__}_{self.epochs}.pkl"
        if type(save_file) != str:
            raise ValueError("ERROR: save_file should be a string.")
        self.save_file = save_file

    # -----------------------------------
    def _get_fit_input_type(self):
        return self.FIT_INPUT_XY

    # -----------------------------------
    def _check_model_param(self):
        """
        Check if the value provided to the model parameter in the constructor
        method is valid or not.
        """
        if isinstance(self.model, BaseTabularModel):
            return
        if self.model not in self.VALID_MODELS:
            raise ValueError(
                "ERROR: Invalid 'model' parameter provided. The value provided to "
                + "this parameter must be one of the following: (i) a valid object of "
                + "the following classes from the sdv library: GaussianCopula(), "
                + "CTGAN(), CopulaGAN(), or TVAE(), or (ii) one of the following strings: "
                + f"{self.VALID_MODELS}, which corresponds to a default model from "
                + "the SDV library."
            )

    # -----------------------------------
    def _get_default_model(self):
        """
        Creates a model from the sdv.tabular module based on the model
        identifier provided in the constructor method. If the model is
        already a model object, return and do nothing. Otherwise, if the
        provided model is a string identifying which model should be
        instantiated, instantiate the model using default parameters.
        """
        if isinstance(self.model, BaseTabularModel):
            return
        if self.model == "ctgan":
            self.model = CTGAN(epochs=self.epochs)
        elif self.model == "copula_gan":
            self.model = CopulaGAN(epochs=self.epochs)
        elif self.model == "tvae":
            self.model = TVAE(epochs=self.epochs)
        elif self.model == "copula":
            self.model = GaussianCopula()

    # -----------------------------------
    def _set_model(self):
        """
        Checks for any errors in the 'model' parameter and create (if needed)
        a default model.
        """
        self._check_model_param()
        self._get_default_model()

    # -----------------------------------
    def _check_valid_df(self, df: pd.DataFrame):
        if type(df.columns[0]) == int:
            raise ValueError(
                "ERROR: the class Synthesizer only accepts data frames with proper column names. "
                + "The provided data frame has column names as integer, not strings."
            )

    # -----------------------------------
    def _preprocess_dataset(self):
        """
        Apply the transforms provided in the transform_pipe parameter of the constructor method.
        """
        self._set_transforms(self.transform_pipe)
        self._fit_transforms(self.df, self.y)
        self.df = self._apply_transforms(self.df)
        if self.in_place:
            self.df_org = self.df

    # -----------------------------------
    def _save_model(self):
        self.model.save(self.save_file)

    # -----------------------------------
    def _load_model(self):
        """
        Load an existing model located in the self.save_file attribute.
        If no file exists, just continue without loading and create a
        new model instead.
        """
        loaded = False
        if self.load_existing and os.path.exists(self.save_file):
            self.print_message(f"Loading existing sythesizer model ({self.save_file})...")
            self.model = self.model.load(self.save_file)
            self.print_message(f"LOADED model of class {type(self.model).__name__}.")
            loaded = True
        return loaded

    # -----------------------------------
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray] = None,
        y: Union[pd.Series, np.ndarray] = None,
        df: Union[pd.DataFrame, np.ndarray] = None,
        label_col: str = None,
    ):
        """
        Prepare the dataset and then call :meth:`fit`. If the model was loaded,
        then there is no need to call :meth:`fit`.

        :param X: contains only the features of the original dataset, that is, does not
            contain the label column;
        :param y: contains only the label column of the original dataset;
        :param df: the full dataset;
        :param label_col: the name or index of the label column;
        """
        self._set_df_mult(df, label_col, X, y, require_set=True)
        self._check_valid_df(self.df)

        self._preprocess_dataset()

        loaded = self._load_model()
        if not loaded:
            self.model.fit(self.df)
            self._save_model()

        self.fitted = True
        return self

    # -----------------------------------
    def _find_minority_majority_columns(self, df: pd.DataFrame):
        """
        Builds a dictionary that informs which of the label values (from the
        column self.label_col_name of dataframe df) represents the minority,
        which represents the majority, not minority, and not majority. This
        information is useful when using one of the predifined sampling
        strategies.

        :param df: the dataframe that contains the label column.
        """
        label_df = df[self.label_col_name]
        label_values = label_df.value_counts()
        strategy_dict = {}
        labels_ordered = label_values.keys()
        strategy_dict["value_counts"] = label_values
        strategy_dict["majority"] = labels_ordered[0]
        strategy_dict["minority"] = labels_ordered[-1]
        strategy_dict["not minority"] = [key for key in labels_ordered if key != strategy_dict["minority"]]
        strategy_dict["not majority"] = [key for key in labels_ordered if key != strategy_dict["majority"]]
        strategy_dict["all"] = labels_ordered
        return strategy_dict

    # -----------------------------------
    def _check_strategy(self, strategy: Union[str, dict, float]):
        """
        Checks if the strategy parameter provided to the transform method is valid.
        A valid strategy must be a string, dictionary, or a float value.

        :param strategy: this parameter can assume the following values:

            - String: one of the following predefined strategies:

              * 'minority': generates synthetic samples for only the minority class;
              * 'not majority': generates synthetic samples for all classes but the
                majority class;
              * 'auto': equivalent to 'minority';

              Note that for a binary classification problem, "minority" is similar to
              "not majority";
            - Dictionary: the dictionary must have one key for each of the possible classes
              found in the label column, and the value associated to each key represents the
              number of instances desired for that class after the undersampling process is done.
              Note: this parameter only works with undersampling approaches that allow
              controlling the number of instances to be undersampled, such as RandomUnderSampler,
              ClusterCentroids (from imblearn). If any other undersampler is provided in the
              under_sampler parameter along with a float value for the strategy_under parameter,
              an error will be raised;
            - Float: a value between [0, 1] that represents the desired ratio between
              the number of instances of the minority class over the majority class
              after undersampling. The ratio 'r' is given by: :math:`r = N_m/N_M` where
              :math:`N_m` is the number of instances of the minority class and :math:`N_M` is the
              number of instances of the majority class after undersampling. Note: this
              parameter only works with undersampling approaches that allow controlling
              the number of instances to be undersampled, such as RandomUnderSampler,
              ClusterCentroids (from imblearn). If any other undersampler is provided in
              the under_sampler parameter along with a float value for the strategy_under
              parameter, an error will be raised.
        """
        if strategy is None:
            strategy = "auto"
        elif type(strategy) == str and strategy not in self.VALID_STRATEGY:
            raise ValueError(
                f"ERROR: invalid value {strategy} for variable 'strategy'. "
                + f"The only valid string values for 'strategy' are: {self.VALID_STRATEGY}."
            )
        elif type(strategy) != str and type(strategy) != float and type(strategy) != dict:
            raise ValueError(
                f"ERROR: invalid value {strategy} for variable 'strategy'."
                + f"Expected 'strategy' to be a float, a string or a dictionary."
            )

        return strategy

    # -----------------------------------
    @staticmethod
    def _get_num_samples_to_create(m0, M0, r1):
        """
        Computes the number of samples that should be generated given a float value
        to the strategy parameter (from the transform method).
        We have that r0 = m0 / M0, where m0 is the number of samples from the minority
        label, M0 is the number of samples from the majority label, and r0 is the ratio
        of minority labels in relation to the majority class.
        For a new ratio r1, we want to create 'e' samples from the minority label such
        that: r1 = (m0 + e) / M0  ====> e = r1*M0 - m0
        """
        e = r1 * M0 - m0
        return e

    # -----------------------------------
    def _strategy_to_samples_number(self, df: pd.DataFrame, strategy: Union[str, dict, float]):
        """
        Converts the strategy parameter (a string, dictionary, or float value) to a dictionary
        that indicates how many instances should be created for each class.

        :param df: the dataset to be transformed;
        :param strategy: check the documentation for the transform method for more details.
        """
        strategy = self._check_strategy(strategy)
        strategy_dict = self._find_minority_majority_columns(df)

        samples_dict = {key: 0 for key in strategy_dict["value_counts"].keys()}
        minority_label = strategy_dict["minority"]
        majority_label = strategy_dict["majority"]
        label_values = strategy_dict["value_counts"]

        if type(strategy) == float:
            if len(strategy_dict["all"]) > 2:
                raise ValueError(
                    "ERROR: a float value for the 'strategy' parameter is only allowed "
                    + "when two labels are present in the dataset. For more lables, provide "
                    + "a dictinary with the number of samples for each label or one of the "
                    + f"predefined strategies: {self.VALID_STRATEGY}"
                )
            e = self._get_num_samples_to_create(
                m0=label_values[minority_label], M0=label_values[majority_label], r1=strategy
            )
            samples_dict[minority_label] = int(e)
        if type(strategy) == dict:
            samples_dict = strategy
        else:
            if strategy == "minority" or strategy == "auto":
                e = self._get_num_samples_to_create(
                    m0=label_values[minority_label], M0=label_values[majority_label], r1=1.0
                )
                samples_dict[minority_label] = int(e)
            elif strategy == "not majority":
                for value in strategy_dict["not majority"]:
                    e = self._get_num_samples_to_create(m0=label_values[value], M0=label_values[majority_label], r1=1.0)
                    samples_dict[value] = int(e)

        return samples_dict

    # -----------------------------------
    def _generate_samples_strategy(self, df: pd.DataFrame, strategy: Union[str, dict, float]):
        """
        Given a dataset and a sampling strategy, get the number of instances that should be
        created for each class (using the method _strategy_to_samples_number) and then create
        these instances using the trained model. For each different class, call the sample()
        method of the model using a condition that specifies that the generated samples should
        be of a specific class (considering the label column). Returns a dataset containing
        only the synthetic instances created.

        :param df: the dataset to be transformed;
        :param strategy: check the documentation for the transform method for more details.
        """
        samples_dict = self._strategy_to_samples_number(df, strategy)
        all_samples = None
        for label_value, n_sample in samples_dict.items():
            if n_sample > 0:
                condition_dict = {self.label_col_name: label_value}
                samples = self.sample(n_sample, condition_dict)
                if all_samples is None:
                    all_samples = samples
                else:
                    all_samples = pd.concat([all_samples, samples], axis=0)
        return all_samples

    # -----------------------------------
    def sample(self, n_samples: int, conditions: dict = None):
        """
        Encapsulates :meth:`sample` from the models that inherit from
        :class:`sdv.tabular.baseBaseTabularModel`. This allows users to use this method
        without requiring to directly access the model object (``self.model``).

        :param n_samples: the number of samples to be generated;
        :param conditions: a set of conditions, specified by a dictionary, that defines the
            characteristics of the synthetic instances that should be created. This parameter
            indicates the values for certain features that the synthetic instances should
            have. If None, then no restrictions will be imposed on how to generate the
            synthetic data.
        :return: a dataset containing the artificial samples.
        :rtype: pd.DataFrame
        """
        if conditions is None:
            samples = self.model.sample(num_rows=n_samples, max_tries_per_batch=200)
        else:
            conditions_obj = Condition(conditions, num_rows=n_samples)
            samples = self.model.sample_conditions(conditions=[conditions_obj], max_tries_per_batch=200)
        return samples

    # -----------------------------------
    def _arrange_transform_df(self, df: pd.DataFrame = None, X: pd.DataFrame = None, y: pd.DataFrame = None):
        """
        Arranges the data provided to the transform method to a standardized pattern, which is:
        a dataframe where the label column is named after the attribute ``self.label_col_name``, and
        the remaining columns are the feature columns. If the dataset is provided through the df
        parameter, no changes are made, but df must follow the same structure as the dataset
        provided during :meth:`fit`. If the data is provided through ``X`` and ``y``, then a new dataset
        is created such that it contains all columns in ``X`` plus the label column ``y``.

        :param df: the full dataset to be transformed, which contains the label column
            (specified during :meth:`fit`);
        :param X: contains only the features of the dataset, that is, does not contain the
            label column;
        :param y: contains only the label column of the dataset to be transformed. If the
            user provides ``df``, ``X`` and ``y`` must be left as None. Alternatively, if the user
            provides (X, y), ``df`` must be left as None;
        """
        input_mode = self.INPUT_DF
        if df is not None:
            df = self._fix_col_transform(df)
            error = False
            if len(df.columns) != len(self.df.columns):
                error = True
            if not error:
                for col in df.columns:
                    if col not in self.df.columns:
                        error = True
                        break
            if error:
                raise ValueError(
                    "ERROR: the data frame passed to the transform method does not "
                    + "follow the same structure as the one used during the fit method. "
                    + f"The data frame used during the fit method had the following columns: {self.df.columns}. "
                    + f"\nThe data frame passed to the transform method has the following columns: {df.columns}."
                )

        elif X is not None and y is not None:
            df = X.copy()
            df[self.label_col_name] = y
            df = self._fix_col_transform(df)
            input_mode = self.INPUT_XY
        else:
            input_mode = self.INPUT_NULL

        return df, input_mode

    # -----------------------------------
    def transform(
        self,
        df: Union[pd.DataFrame, np.ndarray] = None,
        X: Union[pd.DataFrame, np.ndarray] = None,
        y: Union[pd.DataFrame, np.ndarray] = None,
        n_samples: int = None,
        conditions: dict = None,
        strategy: Union[str, dict, float] = None,
    ):
        """
        Transforms a dataset by adding synthetic instances to it. The types of instances
        created depend on the number of samples provided and the set of conditions specified,
        or the chosen strategy. Returns a dataset with the original data and the synthetic
        data generated.

        :param df: the full dataset to be transformed, which contains the label column
            (specified during :meth:`fit`);
        :param X: contains only the features of the dataset, that is, does not contain the
            label column;
        :param y: contains only the label column of the dataset to be transformed. If the
            user provides ``df``, ``X`` and ``y`` must be left as None. Alternatively, if the user
            provides (X, y), ``df`` must be left as None;
        :param n_samples: the number of samples that should be created using the set of
            conditions specified by the 'conditions' parameter;
        :param conditions: a set of conditions, specified by a dictionary, that defines the
            characteristics of the synthetic instances that should be created. This parameter
            indicates the values for certain features that the synthetic instances should
            have. If None, then no restrictions will be imposed on how to generate the
            synthetic data;
        :param strategy: represents the strategy used to generate the artificial instances.
            This parameter is ignored when ``n_samples`` is provided. Strategy can assume the
            following values:

            - **String:** one of the following predefined strategies:

              * **'minority':** generates synthetic samples for only the minority class;
              * **'not majority':** generates synthetic samples for all classes but the
                majority class;
              * **'auto':** equivalent to 'minority';

              Note that for a binary classification problem, "minority" is similar to
              "not majority";
            - **Dictionary:** the dictionary must have one key for each of the possible classes
              found in the label column, and the value associated with each key represents the
              number of instances desired for that class after the undersampling process is done.
              Note: this parameter only works with undersampling approaches that allow
              controlling the number of instances to be undersampled, such as :class:`~imblearn.under_sampling.RandomUnderSampler`,
              :class:`~imblearn.under_sampling.ClusterCentroids` (from :mod:`imblearn`). If any other undersampler is provided in the
              ``under_sampler`` parameter along with a float value for the strategy_under parameter,
              an error will be raised;
            - **Float:** a value between [0, 1] that represents the desired ratio between
              the number of instances of the minority class over the majority class
              after undersampling. The ratio 'r' is given by: :math:`r = N_m/N_M` where
              :math:`N_m` is the number of instances of the minority class and :math:`N_M` is the
              number of instances of the majority class after undersampling. Note: this
              parameter only works with undersampling approaches that allow controlling
              the number of instances to be undersampled, such as :class:`~imblearn.under_sampling.RandomUnderSampler`,
              :class:`~imblearn.under_sampling.ClusterCentroids` (from :mod:`imblearn`). If any other undersampler is provided in
              the under_sampler parameter along with a float value for the ``strategy_under``
              parameter, an error will be raised;
              If None, the default value is set to "auto", which is the same as "minority".
        :return: the transformed dataset.
        :rtype: pd.DataFrame or np.ndarray
        """
        self._check_if_fitted()
        df, input_mode = self._arrange_transform_df(df, X, y)
        if df is not None:
            self._check_valid_df(df)
            df = self._apply_transforms(df)

        if n_samples is None and conditions is not None:
            raise ValueError("ERROR: if 'conditions' is provided, the parameter 'n_samples' is also required.")
        if n_samples is not None:
            samples = self.sample(n_samples, conditions)
        else:
            samples = self._generate_samples_strategy(df, strategy)

        if df is not None:
            samples = pd.concat([df, samples], axis=0)
            if input_mode == self.INPUT_XY:
                df_x = samples.drop(columns=[self.label_col_name])
                df_y = samples[self.label_col_name]
                return df_x, df_y

        return samples
