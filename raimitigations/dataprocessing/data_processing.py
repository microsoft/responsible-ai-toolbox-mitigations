from abc import ABC, abstractmethod
from typing import Union

import random
import pandas as pd
import numpy as np


class DataProcessing(ABC):
    """
    Base class for all classes present in the dataprocessing module
    of the RAIMitigation library. Implements basic functionalities
    that can be used throughout different mitigations.

    :param verbose: indicates whether internal messages should be printed or not.
    """

    FIT_INPUT_DF = 0
    FIT_INPUT_XY = 1

    COL_NAME = 0
    COL_INDEX = 1

    INPUT_NULL = 0
    INPUT_DF = 1
    INPUT_XY = 2

    DEFAULT_LABEL_NAME = "label_col"

    # -----------------------------------
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.label_col_name = None
        self.y = None
        self.column_type = self.COL_NAME
        self.in_pipe = False

    # -----------------------------------
    def print_message(self, text: str):
        if self.verbose:
            print(text)

    # -----------------------------------
    @abstractmethod
    def _get_fit_input_type(self):
        """
        Abstract method. Returns either FIT_INPUT_DF or FIT_INPUT_XY, which indicates
        the main interface that the child class uses:

        - FIT_INPUT_DF: the concrete class requires only a single dataset and it doesn't
        require any knowledge of which is the label column;

        - FIT_INPUT_XY: the concrete class requires a dataset and the name of the label
        column, or two separate datasets: (i) a dataset X containing the features and
        (ii) a dataset Y containing the labels.
        """
        pass

    # -----------------------------------
    def _check_error_df(self, df):
        # Check consistency of the df param
        if df is not None and type(df) != pd.DataFrame and type(df) != np.ndarray:
            class_name = type(self).__name__
            raise ValueError(
                f"ERROR: expected parameter 'df' of {class_name} to be of type pandas.DataFrame "
                + f"or numpy.ndarray, but got a parameter of type {type(df)} instead."
            )

    # -----------------------------------
    def _fix_num_col(self, df: pd.DataFrame, label_col: Union[int, str] = None):
        """
        Checks if the column names of the dataset are present or not. If not, create
        valid column names using the index number (converted to string) of each column.
        Also check if the label_col parameter is provided as an index or as a column
        name. In the former case, convert the label_col to a column name. Finally,
        create a dictionary that maps each column index to the column name. This is
        used when a the object uses the transform_pipe parameter. In these cases,
        the dataset might change its column structure, but we need to map these changes
        and guarantee that the indices provided by the user for a given transformation
        (provided before any transformation is applied) are maped to the correct columns
        even if these columns are changed by other transforms in the transform_pipe.

        :param df: the full dataset;
        :param label_col: the name or index of the label column;
        :return: if label_col is None, returns only the fixed dataset. Otherwise, return
            a tuple (df, label_col) containing the fixed dataset and the fixed label
            column, respectively.
        :rtype: pd.DataFrame or a tuple
        """
        column_type = self.COL_NAME
        if type(df.columns[0]) != str:
            column_type = self.COL_INDEX
            df.columns = [str(i) for i in range(df.shape[1])]
            if label_col is not None:
                label_col = str(label_col)

        if not self.in_pipe:
            self.col_index_to_name = {i: df.columns[i] for i in range(df.shape[1])}
            self.column_type = column_type

        if label_col is not None:
            if self.column_type == self.COL_INDEX:
                label_col = str(label_col)
            else:
                if type(label_col) == int:
                    label_col = self._get_column_from_index(label_col)
            return df, label_col

        return df

    # -----------------------------------
    def _numpy_array_to_df(self, df: Union[pd.DataFrame, np.ndarray]):
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df)
        return df

    # -----------------------------------
    def _fix_col_transform(self, df: Union[pd.DataFrame, np.ndarray]):
        """
        Checks if the column names of the dataset are present or not. If not, create
        valid column names using the index number (converted to string) of each column.

        :param df: the full dataset;
        :return: the fixed dataset.
        :rtype: pd.DataFrame
        """
        if isinstance(df, np.ndarray):
            df = self._numpy_array_to_df(df)
        invalid = False
        if type(df.columns[0]) != str:
            if self.column_type == self.COL_NAME:
                invalid = True
            else:
                df.columns = [str(i) for i in range(df.shape[1])]

        if invalid:
            raise ValueError(
                "ERROR: the columns of the dataset provided to the transform() method from class "
                + f"{type(self).__name__} does not match with the columns provided during the fit() method."
            )
        return df

    # -----------------------------------
    def _set_col_index_to_name(self, col_index_to_name: dict, column_type: int, label_col_name: str):
        """
        Set a few parameters associated to the dataset column structure based on an outside
        information. This is used by transform objects inside a transform_pipe, where these
        objects must always use the same dataset column structure used by the main transform
        object.

        :param col_index_to_name: a dictionary mapping each column index to their respective
            column names. This is useful when a transformation changes the dataset structure;
        :param column_type: indicates if the indexing is done by column names or column index.
            This depends if the dataset has a header or not.
        :param label_col_name: the label column name.
        """
        self.col_index_to_name = col_index_to_name
        self.column_type = column_type
        self.label_col_name = label_col_name
        self.in_pipe = True

    # -----------------------------------
    def _get_column_from_index(self, column_index: int):
        """
        Get the column name associated to a given column index.

        :param column_index: the column index.
        :return: the column name associated to the column specified by the index column_index.
        :rtype: str
        """
        if column_index not in self.col_index_to_name.keys():
            raise ValueError(
                f"ERROR: invalid index provided to the class {type(self).__name__}.\n"
                + f"Error caused by the following index: {column_index}"
            )
        return self.col_index_to_name[column_index]

    # -----------------------------------
    def _check_error_col_list(self, df: pd.DataFrame, col_list: list, col_var_name: str):
        """
        For a given dataset df, check if all column names in col_list are present
        in df. col_list can be a list of column names of column indexes. If one of
        the column names or indexes is not present in df, a ValuError is raised. If
        the col_list parameter is made up of integer values (indices) and the dataframe
        has column names, return a new column list using the column names instead.

        :param df: the dataframe that should be checked;
        :param col_list: a list of column names or column indexes;
        :param col_var_name: a name that identifies where the error occurred (if a
            ValueError is raised). This method can be called from many child classes,
            so this parameter shows the name of the parameter from the child class
            that caused the error.
        :return: the col_list parameter. If the col_list parameter is made up of integer
            values (indices) and the dataframe has column names, return a new column list
            using the column names instead.
        :rtype: list
        """
        if type(col_list) != list:
            raise ValueError(
                f"ERROR: the parameter '{col_var_name}' must be a list of column names."
                + f" Each of these columns must be present in the DataFrame 'df'."
            )

        if col_list == []:
            self.do_nothing = True
        elif df is not None:
            if type(col_list[0]) != int and type(col_list[0]) != str:
                raise ValueError(f"ERROR: '{col_var_name}' must be a list of strings or a list of integers.")

            if type(col_list[0]) == int:
                if self.column_type == self.COL_NAME:
                    col_list = [self._get_column_from_index(index) for index in col_list]
                else:
                    col_list = [str(val) for val in col_list]

            missing = [value for value in col_list if value not in df.columns]
            if missing != []:
                err_msg = (
                    f"ERROR: at least one of the columns provided in the '{col_var_name}' param is "
                    f"not present in the 'df' dataframe. The following columns are missing:\n{missing}"
                )
                raise ValueError(err_msg)
        return col_list

    # -----------------------------------
    def _get_df_subset(self, df: pd.DataFrame, col_list: list):
        """
        For a given dataset df and a list of column names or column indexes in col_list,
        this method returns a subset of df containing only the columns specified by
        col_list. self.column_type indicates if the indexing is done by column names
        or column index. This depends if the dataset has a header or not.

        :param df: the full dataset;
        :param col_list: list of column names or indexes that should be present in the
            subset of df.
        :return: a dataset containing only the columns in col_list.
        :rtype: pd.DataFrame
        """
        if type(col_list) != list:
            raise ValueError("ERROR: calling the _get_df_subset method with an invalid col_list parameter.")
        if len(col_list) > 0 and type(col_list[0]) == int:
            if self.column_type == self.COL_NAME:
                col_list = [self._get_column_from_index(i) for i in col_list]
            else:
                col_list = [str(i) for i in col_list]

        df_valid = df[col_list]

        return df_valid

    # -----------------------------------
    def _check_if_fitted(self):
        if not self.fitted:
            raise ValueError(
                f"ERROR: trying to call the transform() method from an instance of the {self.__class__.__name__} class "
                + "before calling the fit() method. "
                + "Call the fit() method before using this instance to transform a dataset."
            )

    # -----------------------------------
    def _check_df_input_format(self, df: pd.DataFrame, label_col: str, X: pd.DataFrame, y: pd.DataFrame):
        """
        Checks the consistency of the input scheme chosen by the user. There are two ways
        to provide the dataset to some of the concrete classes (classes that require the
        label column):
            - (df, label_col): the user provides the full dataset df and the name or index
            of the label column;
            - (X, y): the user provides a dataset X containing only the features and a dataset
            y containing only the label column;
        For the concrete classes that use this input approach, the user must choose one of the
        two approaches above. If the user provides (df, label_col), X and y must be left as None.
        Alternatively, if the user provides (X, y), df and label_col must be left as None.

        :param df: the full dataset;
        :param label_col: the name or index of the label column;
        :param X: contains only the features of the original dataset, that is, does not
            contain the label column;
        :param y: contains only the label column of the original dataset.
        :return: an integer value representing the input scheme used by the fit() method of the
            child class. Can be one of the following values:

                * INPUT_DF: when the fit() method requires only the full dataframe;
                * INPUT_XY: when the fit() method requires a dataframe containing only the features
                  (X) and a dataframe with only the label column (Y);
                * INPUT_NULL: when the fit() method doesn't require any input.

        :rtype: one of the following values: self.INPUT_DF, self.INPUT_XY, self.INPUT_NULL
        """

        def error_message(param1: str, param2: str):
            raise ValueError(
                f"ERROR: '{param1}' and '{param2}' must be provided together, "
                + f"but only '{param1}' is valid parameter."
            )

        input_scheme = self.INPUT_DF
        if df is not None or label_col is not None:
            if df is None:
                error_message("label_col", "df")
            elif label_col is None:
                error_message("df", "label_col")
            if type(label_col) != str and type(label_col) != int:
                raise ValueError("ERROR: 'label_col' must be a string or an integer.")
            label_err_str = type(label_col) == str and label_col not in df.columns.values.tolist()
            label_err_int = type(label_col) == int and label_col not in range(0, df.shape[1])
            if label_err_str or label_err_int:
                raise ValueError(
                    f"ERROR: label_col = {label_col} not present in the dataframe. "
                    + f"'df' has the following columns: {df.columns}"
                )
        elif X is not None or y is not None:
            if X is None:
                error_message("y", "X")
            elif y is None:
                error_message("X", "y")
            if type(X) != pd.DataFrame or (type(y) != pd.DataFrame and type(y) != pd.core.series.Series):
                raise ValueError("ERROR: 'X' and 'y' must be a pandas DataFrame.")
            input_scheme = self.INPUT_XY
        else:
            input_scheme = self.INPUT_NULL

        return input_scheme

    # -----------------------------------
    def _set_df(self, df: Union[pd.DataFrame, np.ndarray], require_set: bool = False):
        """
        Sets the current dataset self.df using a new dataset df. If both
        self.df and df are None, then a ValueError is raised. df can be None
        if a valid self.df has already been set beforehand.

        :param df: the full dataset;
        :param require_set: a boolean value indicating if the df parameter must
            be a valid dataframe or not. If true and df is None, an error is raised.
        """
        self._check_error_df(df)
        if self.df is None and df is None and require_set:
            raise ValueError(
                "ERROR: dataframe not provided. Provide the dataframe "
                + "through the class constructor or through the fit() method."
            )
        if df is not None:
            if isinstance(df, np.ndarray):
                df = self._numpy_array_to_df(df)
            self.df = self._fix_num_col(df)

    # -----------------------------------
    def _set_df_mult(
        self,
        df: pd.DataFrame,
        label_col: str,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray],
        require_set: bool = False,
    ):
        """
        Sets the current dataset self.df and the current label column self.y.
        This method is reserved for classes that require a label column. Otherwise,
        use the _set_df method. For the current method, first it is checked which
        input scheme is used by the user using the  _check_df_input_format method
        (check the documentation of this method for more info). In the sequence,
        some attributes are set according to the input scheme used.

        :param df: the full dataset;
        :param label_col: the name or index of the label column;
        :param X: contains only the features of the original dataset, that is, does not
            contain the label column;
        :param y: contains only the label column of the original dataset;
        :param require_set: if True, a ValueError will be raised if both pairs of
            variables ((df, label_col) and (X, y)) are all None.
        """
        self._check_error_df(df)
        self._check_error_df(X)
        # convert to pd.DataFrame if input is a numpy array
        if X is not None and isinstance(X, np.ndarray):
            X = self._numpy_array_to_df(X)
        if y is not None and isinstance(y, np.ndarray):
            y = pd.Series(y)
        if df is not None and isinstance(df, np.ndarray):
            df = self._numpy_array_to_df(df)

        input_scheme = self._check_df_input_format(df, label_col, X, y)
        if input_scheme == self.INPUT_DF:
            df, label_col = self._fix_num_col(df, label_col)
            self.df = df.drop(columns=[label_col])
            self.y = df[label_col]
            self.input_scheme = input_scheme
            self.label_col_name = label_col
        elif input_scheme == self.INPUT_XY:
            X = self._fix_num_col(X)

            self.df = X
            self.y = y

            if type(self.y) == pd.core.series.Series:
                self.label_col_name = self.y.name
            else:
                label_col_name = self.DEFAULT_LABEL_NAME
                while label_col_name in self.df.columns:
                    label_col_name += str(random.randint([0, 9]))
                self.label_col_name = label_col_name

            self.input_scheme = input_scheme
        self.df_org = self.df

        if (self.df is None or self.y is None) and require_set:
            raise ValueError(
                "ERROR: dataframe not provided. Provide the dataframe through the 'df' and "
                + "'label_col' parameters or through the 'X' and 'y' parameters during the "
                + "instantiation of the class or during the call of the .fit() method."
            )

    # -----------------------------------
    def _get_preprocessing_requirements(self):
        """
        Returns the preprocessing steps required by a given concrete class.
        The default behavior (implemented in the current abstract class)
        is to return an empty set of preprocessing steps. Concrete classes
        should override this method if they require any preprocessing step
        prior to their fit and transform methods.
        NOTE: the preprocessing steps are represented by a dictionary, where
        the keys are references of abstract classes (that inherits from the
        current class) and the values are the concrete classes that inherit
        from the classes represented by their respective keys. All class
        references must be from classes from the current dataprocessing library.
        For example, the dictionary {AbstracClass: ConcreteClass} indicates that
        the current class (that overrides the current method) requires a
        transformation class that inherits from AbstractClass. If the list of
        transformations provided by the user (check _set_transforms form more
        information) does not include any concrete class that inherits from
        AbstractClass, then a new class will be added to the transformations
        list. The class that will be created is given by the value associate
        to AbstractClass, that is, ConcreteClass in our example.
        To see an example, search for any concrete class that overrides this
        method.
        """
        return {}

    # -----------------------------------
    def _create_default_transforms(self):
        """
        Creates a list of concrete class references based on the required
        classes returned by the _get_preprocessing_requirements method.
        This method is only called if the user doesn't specify any preprocessing
        transformations.

        :return: a list containing default objects from the current class, which
            represents the required preprocessing steps for the current child class.
        :rtype: list
        """
        tf_list = []
        requirements = self._get_preprocessing_requirements()
        for req in requirements.items():
            tf_list.append(req[1])
        return tf_list

    # -----------------------------------
    def _check_transforms(self):
        """
        Checks if the list of transformations provided by the user contains
        all mandatory classes specified by _get_preprocessing_requirements.
        If one of the mandatory transformations are missing, the concrete
        class specified by _get_preprocessing_requirements (given
        by the value of the dictionary returned by _get_preprocessing_requirements)
        is added to the transformations list.
        """
        for transform in self.transform_pipe:
            has_pipe = hasattr(transform, "transform_pipe")
            if has_pipe and transform.transform_pipe != []:
                raise ValueError(
                    f"ERROR: transformers in the 'transform_pipe' parameter are not allowed to have "
                    + f"their own pipeline (that is, 'transform_pipe' must be None for all transformers "
                    + f"in another class' transform_pipe'). The transform_pipe from class "
                    + f"{self.__class__.__name__} includes a transformer from class "
                    + f"{transform.__class__.__name__} that has its own transform_pipe."
                )

        requirements = self._get_preprocessing_requirements()
        new_tf_list = []
        for req in requirements.items():
            missing = True
            class_req = req[0]
            for transform in self.transform_pipe:
                if isinstance(transform, class_req):
                    missing = False
                    break
            if missing:
                new_tf_list.append(req[1])
        self.transform_pipe = new_tf_list + self.transform_pipe

    # -----------------------------------
    def _set_transforms(self, transform_pipe: list):
        """
        Sets the transform_pipe attribute based on the list of transformations
        provided by the user. If no transformation list is provided, a list
        is created from the default classes specified by the values of the
        dictionary returned by _get_preprocessing_requirements. If a valid
        list is provided but one of the required classes (given by the keys
        in the dictionary returned by _get_preprocessing_requirements) is
        not present in it, then an object of the class given by the value
        associated with this key is added to the transformation list. Check
        the documentation of _get_preprocessing_requirements for more details.

        :param transform_pipe: a list of objects of classes from the current
            dataprocessing library that will be used as a preprocessing step,
            that is, will be applied to transform the dataset before calling
            the fit and transform methods. For example, [BasicImputer] is a
            valid transformation list that specifies that the fit and transform
            methods of the BasicImputer class should be applied before the fit
            and transform methods of the current class.
        """
        if transform_pipe is None:
            transform_pipe = []
        if type(transform_pipe) != list:
            raise ValueError("ERROR: 'transform_pipe' must be a list of transform objects.")
        if transform_pipe == [] and self.df is not None:
            transform_pipe = self._create_default_transforms()
        self.transform_pipe = transform_pipe
        self._check_transforms()

    # -----------------------------------
    def _fit_transforms(self, df: pd.DataFrame, y: pd.DataFrame = None):
        """
        Calls the fit method for all transformations in the self.transform_pipe
        attribute (check _set_transforms for more information). This method
        goes through each object in the self.transform_pipe call their fit and
        transform methods over the dataset df and label column y. For each object,
        it is necessary to first check which input method is used by the class:
        FIT_INPUT_DF or FIT_INPUT_XY, given by the _get_fit_input_type method.
        Note that the fit and transforms are called sequentially for each object
        in self.transform_pipe, that is, the fit and transform methods of the
        last object in self.transform_pipe is called over the dataset that
        results from the transform method of the last but one object in
        self.transform_pipe.
        """
        for tf in self.transform_pipe:
            fit_params = tf._get_fit_input_type()
            tf._set_col_index_to_name(self.col_index_to_name, self.column_type, self.label_col_name)
            if fit_params == self.FIT_INPUT_DF:
                tf.fit(df)
                df = tf.transform(df)
            elif fit_params == self.FIT_INPUT_XY:
                if y is None:
                    raise ValueError(
                        f"ERROR: using the tranformation class {type(tf).__name__} "
                        + "that requires an X and Y datasets as a preprocessing step "
                        + "inside another class that does not require the separation "
                        + "a Y dataset (whcich contains only the labels)."
                    )
                tf.fit(df, y)
                df = tf.transform(df)
            else:
                raise NotImplementedError("ERROR: Unknown fit input order.")

    # -----------------------------------
    def _apply_transforms(self, df: pd.DataFrame):
        """
        Applies the preprocessing transformations defined by the transform_pipe
        attribute. The current method considers that the objects in
        self.transform_pipe have already been fitted. The transform method of each
        of these objects is called in sequential order, that is, the
        transformation of the last object is applied over the dataset that
        results from the transform method of the last but one object in the
        transform_pipe. Check the documentation of _fit_transforms for more details.

        :param df: the dataset to which the transformations should be applied.
        :return: the dataset df after calling the transform() method of all objects
            in the self.transform_pipe internal parameter.
        :rtype: pd.DataFrame or np.ndarray
        """
        for tf in self.transform_pipe:
            df = tf.transform(df)
        return df

    # -----------------------------------
    def inverse_transform(self, df: pd.DataFrame):
        """
        Implements the behavior for the inverse transformation. This method
        first checks if the current class can be reversed. All transformation
        classes that can be reversed have a private _inverse_transform() method.
        If the current class doesn't have this method, an error is raised. If
        it does, then we call this method. The next step is to reverse all other
        reversible transformations in the transf_pipe parameter up until we reach
        the first reversible transformation. The following steps are executed:
        (i) call the _inverse_transform() method for the current class, which
        returns a new dataset, (ii) call the _inverse_transform() method for all
        other transformers in the transform_pipe parameter (note that these methods
        are called in reverse order to guarantee a correct behavior, and that the
        inverse_transform is called only for the transformations that appear after
        the last non-reversible transformer object in the transform_pipe parameter),
        (iii) return the reversed dataset.

        :param df: the dataframe to be scaled containing all
            original columns, that is, all columns that should be ignored
            and those that should be scaled.
        :return: the dataset df after calling the _inverse_transform() method of
            all objects in the self.transform_pipe internal parameter (in reversed
            order).
        :rtype: pd.DataFrame or np.ndarray
        """
        has_inverse = hasattr(self.__class__, "_inverse_transform")
        reversible = has_inverse and callable(getattr(self.__class__, "_inverse_transform"))
        if not reversible:
            raise ValueError(f"ERROR: the class {self.__class__.__name__} is not reversible.")

        df = self._fix_col_transform(df)
        df = self._inverse_transform(df)

        has_pipe = hasattr(self, "transform_pipe")
        if has_pipe:
            for transform in reversed(self.transform_pipe):
                has_inverse = hasattr(transform.__class__, "_inverse_transform")
                if not has_inverse:
                    break
                df = transform._inverse_transform(df)

        return df
