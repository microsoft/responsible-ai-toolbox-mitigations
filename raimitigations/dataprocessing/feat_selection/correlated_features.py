from typing import Union
import json
import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
import statsmodels.api as sm
from statsmodels.formula.api import ols
import researchpy as rp

from .selector import FeatureSelection
from ..encoder import EncoderOrdinal
from ..data_utils import get_cat_cols, ordinal_to_onehot, err_float_01


"""
TODO: for the Jensen method:
    - save the histograms in a folder specified by the user (create a new parameter for that) and add the path
of each histogram in their associated results in the JSON file;
    - allow the user to pass a dict to the jensen_n_bins parameter that specifies the number of bins for each
categorical variable;

TODO: Allow the user to save the selected features for future use
"""


def _get_exact_matches(col1: list, col2: list):
    """
    Computes the percentage of exact matches between two lists.
    Returns a string with the percentage of exact matches.

    :param col1: the first list;
    :param col2: the second list.
    """
    list1 = col1.values.tolist()
    list2 = col2.values.tolist()
    if len(list1) != len(list2):
        raise ValueError("ERROR: _get_exact_matches called with lists with different sizes.")

    matches = 0
    for i in range(len(list1)):
        if list1[i] == list2[i]:
            matches += 1

    pct_match = matches / len(list1)
    pct_match = f"{pct_match*100}%"

    return pct_match


# -----------------------------------
def freedman_diaconis(data: pd.Series):
    """
    Computes the optimal number of bins for a set of data using the
    Freedman Diaconis rule.

    :param data: the data column used to compute the number of bins.
    """
    data = np.asarray(data, dtype=np.float_)
    iqr = stats.iqr(data, rng=(25, 75), scale=1.0, nan_policy="omit")
    N = data.size
    bw = (2 * iqr) / np.power(N, 1 / 3)

    min_val, max_val = data.min(), data.max()
    datrng = max_val - min_val
    result = int((datrng / bw) + 1)
    return result


# -----------------------------------
class CorrelatedFeatures(FeatureSelection):
    """
    Concrete class that measures the correlation between variables
    (numerical x numerical, categorical x numerical, and categorical x categorical)
    and drop features that are correlated to another feature.

    :param df: the data frame to be used during the fit method.
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

    :param X: contains only the features of the original dataset, that is, does not
        contain the label column. This is useful if the user has already separated
        the features from the label column prior to calling this class. This parameter
        is mandatory if  ``y`` is provided;

    :param y: contains only the label column of the original dataset.
        This parameter is mandatory if  ``X`` is provided;

    :param transform_pipe: a list of transformations to be used as a pre-processing
        pipeline. Each transformation in this list must be a valid subclass of the
        current library (:class:`~raimitigations.dataprocessing.EncoderOrdinal`, :class:`~raimitigations.dataprocessing.BasicImputer`, etc.). Some feature selection
        methods require a dataset with no categorical features or with no missing
        values (depending on the approach). If no transformations are provided, a set
        of default transformations will be used, which depends on the feature selection
        approach (subclass dependent);

    :param in_place: indicates if the original dataset will be saved internally
        (``df_org``) or not. If True, then the feature selection transformation is saved
        over the original dataset. If False, the original dataset is saved separately
        (default value);

    :param cor_features: a list of the column names or indexes that should have their
        correlations checked. If None, all columns are checked for correlations, where
        each correlation is checked in pairs (all possible column pairs are checked);

    :param method_num_num: the method used to test the correlation between numerical
        variables. Must be a list containing one or more methods (limited to the
        number of available methods). The available methods are:
        ["spearman", "pearson", "kendall"]. If None, none of the correlations between
        two numerical variables will be tested;

    :param num_corr_th: the correlation coefficient value used as a threshold for
        considering if there is a correlation between two numerical variables.
        That is, given two variables with a correlation coefficient of 'x' (depends on
        the correlation used, specified by ``method_num_num``), a correlation is considered
        only if abs(x) >= method_num_num and if the associated p-value 'p' is smaller than
        'p' <= num_pvalue_th;

    :param num_pvalue_th: the p-value used as a threshold when considering if there is a
        correlation between two variables. That is, given two variables with a correlation
        coefficient of 'x' (depends on the correlation used, specified by ``method_num_num``),
        a correlation is considered only if abs(x) >= method_num_num and if the associated
        p-value 'p' is smaller than 'p' <= num_pvalue_th;

    :param method_num_cat: the method used to compute the correlation between a categorical and
        a numerical variable. There are currently three approaches implemented:

            * **'anova':** uses the ANOVA test to identify a correlation. First, we use the Levene
              test to see if the numerical variable has a similar variance across the
              different values of the categorical variable (Homoscedastic data). If
              the test passes (that is if the p-value of the Levene test is greater
              than ``levene_pvalue``), then we can perform the ANOVA test, in which we
              compute the F-statistic to see if there is a correlation between the
              numerical and categorical variables and its associated p-value. We also
              compute the omega-squared metric. If the p-value is less than ``anova_pvalue``
              and the omega-squared is greater than ``omega_th``, then both variables
              are considered to be correlated;
            * **'jensen':** first we clusterize the numerical values according to their respective
              values of the categorical data. We then compute the probability density
              function of the numerical variable for each cluster (we approximate the
              PDF with the histogram using ``jensen_n_bins`` different bins). The next
              step is to compute the Jensen-Shannon Distance metric between the distribution
              functions of each pair of clusters. This distance metric varies from 0 to 1,
              where values closer to 0 mean that both distributions tested are similar and
              values closer to 1 mean that the distributions are different. If all pairs
              of distributions tested are considered different (a Jensen-Shannon metric above
              ``jensen_th`` for all pairs tested), then both variables are considered to be
              correlated;
            * **'model':** trains a simple decision tree using the numerical variable and predicts the
              categorical variable. Both variables are first divided into a training and
              test set (70% and 30% of the size of the original variables, respectively). The
              training set is used to train the decision tree, where the only feature used
              by the model is the numerical variable and the predicted label is the different
              values within the categorical variable. After training, the model is used to
              predict the values of the test set and a set of metrics is computed to assess the
              performance of the model (the metrics computed are defined by ``model_metrics``).
              If all metrics computed are above the threshold ``metric_th``, then both variables
              are considered to be correlated;

        If set to None, then none of the correlations between numerical and categorical variables will
        be tested;

    :param levene_pvalue: the threshold used to check if a set of samples are homoscedastic (similar
        variances across samples). This condition is necessary for the ANOVA test. This check is done
        using the Levene test, which considers that all samples have similar variances as the null
        hypothesis. If the p-value associated with this test is high, then the null hypothesis is accepted,
        thus allowing the ANOVA test to be carried out. This parameter defines the threshold used by the
        p-value of this test: if p-value > levene_pvalue, then the data is considered to be homoscedastic.
        This parameter is ignored if method_num_cat != 'anova';

    :param anova_pvalue: threshold used by the p-value associated with the F-statistic computed by the ANOVA
        test. If the p-value < anova_pvalue, then we consider that there is a statistically significant
        difference between the numerical values of different clusters (clusterized according to the values
        of the categorical variable). This implies a possible correlation between the numerical and
        categorical variables, although the F-statistic doesn't tell us the magnitude of this difference.
        For that, we use the Omega-Squared metric. This parameter is ignored if method_num_cat != 'anova';

    :param omega_th: the threshold used for the omega squared metric. The omega squared is a metric that
        varies between 0 and 1 that indicates the effect of a categorical variable over the variance of
        a numerical variable. A value closer to 0 indicates a weak effect, while values closer to 1 show
        that the categorical variable has a significant impact on the variance of the numerical variable,
        thus showing a high correlation. If the omega squared is greater than omega_th, then both variables
        being analyzed are considered to be correlated. This parameter is ignored if method_num_cat != 'anova';

    :param jensen_n_bins: the number of bins used for creating the histogram of each cluster of data when
        method_num_cat = 'jensen'. For this method, we cluster the numerical data according to the categorical
        variable. For each cluster, we compute a histogram, which is used to approximate the Probability
        Density Function of that cluster. This parameter controls the number of bins used during the creation
        of the histogram. This parameter is ignored if method_num_cat != 'jensen'. If None, the best number
        of bins for the numerical variable being analyzed is computed using the Freedman Diaconis rule;

    :param jensen_th: when method_num_cat = 'jensen', we compare the distribution of each cluster of data
        using the Jensen-Shannon distance metric. If the distance is close to 1, then the distributions are
        considered different. If all pairs of clusters have a high distance, then both variables being analyzed
        are considered to be correlated. This parameter indicates the threshold used to check if a distance
        metric is high or not: if distance > jensen_th, then the distributions being compared are considered
        different. Must be a float value within [0, 1]. This parameter is ignored if method_num_cat != 'jensen';

    :param model_metrics: a list of metric names that should be used when evaluating if a model trained using
        a single numerical variable to predict a categorical variable is good enough. If the trained model
        presents a good performance for the metrics in model_metrics, then both variables being analyzed are
        considered to be correlated. This parameter must be a list, and the allowed values in this list are:
        ["f1", "auc", "accuracy", "precision", "recall"]. This parameter is ignored if
        method_num_cat != 'model';

    :param metric_th: given the metrics provided by model_metrics, a pair of variables being analyzed are only
        considered correlated if all metrics in model_metrics achieve a score greater than metric_th over the
        test set (the variables being analyzed are split into training and test set internally). This parameter
        is ignored if method_num_cat != 'model';

    :param method_cat_cat: the method used to test the correlation between two categorical variables. There is only
        one option implemented:

            * **'cramer':** performs the Cramer's V test between two categorical variables. This test returns a value
              between 0 and 1, where values near 1 indicate a high correlation between the variables
              and a p-value associated with this metric. If the Cramer's V correlation coefficient is
              greater than cat_corr_th and its p-value is smaller than cat_pvalue_th, then both variables
              are considered to be correlated.

        If set to None, then none of the correlations between two categorical variables will be tested;

    :param cat_corr_th: the threshold used for the Cramer's V correlation coefficient. Values greater than ``cat_corr_th``
        indicates a high correlation between two variables, but only if the p-value associated with this coefficient
        is smaller than ``cat_pvalue_th``;

    :param cat_pvalue_th: check the description for the parameter ``cat_corr_th`` for more information;

    :param tie_method: the method used to choose the variable to remove in case a correlation
        between them is identified. This is used for all types of correlations:
        numerical x numerical, categorical x numerical, and categorical x categorical. The
        possible values are:

            * **"missing":** chooses the variable with the least number of missing values;
            * **"var":** chooses the variable with the largest data dispersion (std / (V - v),
              where std is the standard deviation of the variable, V and v are the
              maximum and minimum values observed in the variable, respectively).
              Works only for numerical x numerical analysis. Otherwise, it uses the
              cardinality approach internally;
            * **"cardinality":** chooses the variable with the most number of different values present;

        In all three cases, if both variables are tied (same dispersion, same number of missing
        values, or same cardinality), the variable to be removed will be selected randomly;

    :param save_json: if True, the summary jsons are saved according to the paths ``json_summary``, ``json_corr``, and
        ``json_uncorr``. If False, these json files are not saved;

    :param json_summary: when calling the fit method, all correlations will be computed according to the many
        parameters detailed previously. After computing all this data, everything is saved in a JSON file,
        which can then be accessed and analyzed carefully. We recommend using a JSON viewing tool for this.
        This parameter indicates the name of the file where the JSON should be saved (including the path
        to the file). If set to None no JSON file will be saved;

    :param json_corr: similar to ``json_summary``, but corresponds to the name of the JSON file that contains only
        the information of the pairs of variables considered to be correlated (with no repetitions);

    :param json_uncorr: similar to ``json_summary``, but corresponds to the name of the JSON file that contains only
        the information of the pairs of variables considered NOT to be correlated (with no repetitions);

    :param compute_exact_matches: if True, compute the number of exact matches between two variables and save
        this information in the ``json_summary``, ``json_uncorr``, and ``json_corr``;

    :param verbose: indicates whether internal messages should be printed or not.
    """

    NUM_COR_TH = 0.85
    NUM_PVALUE_TH = 0.05
    PVALUE_LEVENE = 0.01
    PVALUE_ANOVA = 0.05
    OMEGA_ANOVA = 0.9
    METRIC_TH = 0.9
    CAT_COR_TH = 0.85
    CAT_PVALUE_TH = 0.05
    VALID_NUM_COR = {"spearman": stats.spearmanr, "pearson": stats.pearsonr, "kendall": stats.kendalltau}
    VALID_NUM_CAT = ["anova", "jensen", "model"]
    VALID_CAT = ["cramer"]
    VALID_METRICS = ["f1", "auc", "accuracy", "precision", "recall"]
    MIN_NUM_INST = 10

    VALID_TIE_METHODS = ["missing", "var", "cardinality"]

    TYPE_CORR_KEY = "Correlation Type"
    TYPE_CORR_NN = "numerical x numerical"
    TYPE_CORR_NC = "numerical x categorical"
    TYPE_CORR_CC = "categorical x categorical"

    PVAL_KEY = "p-value"
    PVAL_LEVENE_KEY = "p-value (Levene Test)"
    CORR_KEY = "correlation"
    CORRELATED_KEY = "correlated?"
    FINAL_CORR_KEY = "Final Correlation"

    CRAMER_KEY = "Cramer's V"
    ANOVA_VALID_KEY = "Valid Anova (Homoscedastic data)"
    ANOVA_KEY = "ANOVA F-Value"
    OMEGA_KEY = "Omega^2"
    JENSEN_KEY = "Jensen-Shannon Distance"
    METRIC_KEY = "Model Metrics"
    EXACT_MATCH_KEY = "Exact Matches"
    COR_TH_KEY = "Correlation threshold used"
    PVALUE_TH_KEY = "P-Value threshold used"
    METRIC_TH_KEY = "Metric threshold used"
    METRIC_USED_KEY = "Metrics used"
    PVAL_TH_LEVENE_KEY = "Levene P-Value threshold used"
    PVAL_ANOVA_KEY = "ANOVA P-Value threshold used"
    OMEGA_TH_KEY = "Omega^2 threshold used"
    JENSEN_BIN_KEY = "Number of bins used"
    JENSEN_TH_KEY = "Jensen-Shannon Distance threshold used"

    CORR_VAR_KEY = "correlated variables"
    DEGREE_KEY = "degree"
    BLOCK_KEY = "blocked"

    MODEL = DecisionTreeClassifier(max_features="sqrt")
    TEST_SIZE = 0.3

    # -----------------------------------
    def __init__(
        self,
        df: Union[pd.DataFrame, np.ndarray] = None,
        label_col: str = None,
        X: Union[pd.DataFrame, np.ndarray] = None,
        y: Union[pd.DataFrame, np.ndarray] = None,
        transform_pipe: list = None,
        in_place: bool = False,
        cor_features: list = None,
        method_num_num: list = ["spearman"],
        num_corr_th: float = NUM_COR_TH,
        num_pvalue_th: float = NUM_PVALUE_TH,
        method_num_cat: str = "model",
        levene_pvalue: float = PVALUE_LEVENE,
        anova_pvalue: float = PVALUE_ANOVA,
        omega_th: float = OMEGA_ANOVA,
        jensen_n_bins: int = None,
        jensen_th: float = 0.8,
        model_metrics: list = ["f1", "auc"],
        metric_th: float = METRIC_TH,
        method_cat_cat: str = "cramer",
        cat_corr_th: float = CAT_COR_TH,
        cat_pvalue_th: float = CAT_PVALUE_TH,
        tie_method: str = "missing",
        save_json: bool = True,
        json_summary: str = "summary.json",
        json_corr: str = "corr_pairs.json",
        json_uncorr: str = "uncorr_pairs.json",
        compute_exact_matches: bool = True,
        verbose: bool = True,
    ):
        super().__init__(df, label_col, X, y, transform_pipe, in_place, verbose)
        self.cor_features = None
        self._set_numerical_corr_param(method_num_num, num_corr_th, num_pvalue_th)
        self._set_num_cat_corr_param(
            method_num_cat, levene_pvalue, anova_pvalue, omega_th, jensen_n_bins, jensen_th, model_metrics, metric_th
        )
        self._set_cat_corr_param(method_cat_cat, cat_corr_th, cat_pvalue_th)
        self._set_tie_method(tie_method)
        self._set_possible_cor_features(cor_features)
        self.missing_count = None
        self.dispersion_index = None
        self.feat_nunique = None
        self.save_json = save_json
        self.json_summary = json_summary
        self.json_corr = json_corr
        self.json_uncorr = json_uncorr
        self.compute_exact_matches = compute_exact_matches

    # -----------------------------------
    def _get_preprocessing_requirements(self):
        requirements = {}
        return requirements

    # -----------------------------------
    def _set_numerical_corr_param(self, method_num_num: list, num_corr_th: float, num_pvalue_th: float):
        """
        Sets the attributes associated with the numerical x numerical correlation analysis. Also
        checks for any errors with the parameters provided.

        :param method_num_num: the method used to test the correlation between numerical variables.
            Must be a list containing one or more methods (limited to the number of available
            methods). The available methods are: ["spearman", "pearson", "kendall"];
        :param num_corr_th: the correlation coefficient value used as a threshold for considering if
            there is a correlation between two numerical variables. Must be a value between [0, 1];
        :param num_pvalue_th: the p-value used as a threshold when considering if there is a correlation
            between two variables.
        """
        base_err_str = (
            f"ERROR: invalid value for parameter 'method_num_num'. "
            f"Expected a list with the possible values: {self.VALID_NUM_COR.keys()}."
        )
        if method_num_num is not None:
            if type(method_num_num) != list:
                raise ValueError(base_err_str)
            else:
                invalid = [name for name in method_num_num if name not in self.VALID_NUM_COR.keys()]
                if invalid != []:
                    raise ValueError(base_err_str + f" Found the invalid values: {invalid}")

        err_float_01(num_corr_th, "num_corr_th")
        err_float_01(num_pvalue_th, "num_pvalue_th")

        self.method_num_num = method_num_num
        self.num_corr_th = num_corr_th
        self.num_pvalue_th = num_pvalue_th

    # -----------------------------------
    def _set_num_cat_corr_param(
        self,
        method_num_cat: str,
        levene_pvalue: float,
        anova_pvalue: float,
        omega_th: float,
        jensen_n_bins: int,
        jensen_th: float,
        model_metrics: list,
        metric_th: float,
    ):
        """
        Sets the attributes associated with the numerical x categorical correlation analysis. Also
        checks for any errors with the parameters provided.

        :param method_num_cat: the method used to compute the correlation between a categorical
            and a numerical variable. There are currently three approaches implemented: 'anova',
            'jensen', and 'model';
        :param levene_pvalue: the threshold used to check if a set of samples are homoscedastic
            (similar variances across samples);
        :param anova_pvalue: threshold used by the p-value associated with the F-statistic computed
            by the ANOVA test;
        :param omega_th: the threshold used for the omega squared metric;
        :param jensen_n_bins: the number of bins used for creating the histogram of each cluster
            of data when method_num_cat = 'jensen';
        :param jensen_th: when method_num_cat = 'jensen', we compare the distribution of each
            cluster of data using the Jensen-Shannon distance metric. This parameter represents the
            threshold used to check if a distance metric is high or not;
        :param model_metrics: a list of metric names that should be used when evaluating if a model
            trained using a single numerical variable to predict a categorical variable is good enough.
            If the trained model presents a good performance for the metrics in model_metrics, then
            both variables being analyzed are considered to be correlated. This parameter must be a
            list, and the allowed values in this list are: ["f1", "auc", "accuracy", "precision",
            "recall"]. This parameter is ignored if method_num_cat != 'model';
        :param metric_th: given the metrics provided by model_metrics, a pair of variables being
            analyzed are only considered correlated if all metrics in model_metrics achieve a score
            greater than metric_th over the test set (the variables being analyzed are split into
            training and test set internally). This parameter is ignored if method_num_cat != 'model';
        """
        if method_num_cat is not None and method_num_cat not in self.VALID_NUM_CAT:
            raise ValueError(
                f"ERROR: invalid value for parameter 'method_num_cat'. Expected a list with the possible values: {self.VALID_NUM_CAT}."
            )

        err_float_01(levene_pvalue, "levene_pvalue")
        err_float_01(anova_pvalue, "anova_pvalue")
        err_float_01(omega_th, "omega_th")
        err_float_01(metric_th, "metric_th")
        err_float_01(jensen_th, "jensen_th")

        if jensen_n_bins is not None and jensen_n_bins < 0:
            raise ValueError(
                f"ERROR: parameter 'jensen_n_bins' must be a positive integer. Instead, got value {jensen_n_bins}."
            )

        base_err_str = f"ERROR: invalid value for parameter 'model_metrics'. Expected a list with the possible values: {self.VALID_METRICS}."
        if type(model_metrics) != list:
            raise ValueError(base_err_str)
        else:
            invalid = [name for name in model_metrics if name not in self.VALID_METRICS]
            if invalid != []:
                raise ValueError(base_err_str + f" Found the invalid values: {invalid}")

        self.method_num_cat = method_num_cat
        self.levene_pvalue = levene_pvalue
        self.anova_pvalue = anova_pvalue
        self.omega_th = omega_th
        self.jensen_n_bins = jensen_n_bins
        self.jensen_th = jensen_th
        self.model_metrics = model_metrics
        self.metric_th = metric_th

    # -----------------------------------
    def _set_cat_corr_param(self, method_cat_cat: str, cat_corr_th: float, cat_pvalue_th: float):
        """
        Sets the attributes associated with the categorical x categorical correlation analysis. Also
        checks for any errors with the parameters provided.

        :param method_cat_cat: the method used to test the correlation between two categorical
            variables. Can be one of the following: "cramer" or None. If set to None, then none of
            the correlations between two categorical variables will be tested;
        :param cat_corr_th: the threshold used for the Cramer's V correlation coefficient. Values
            greater than cat_corr_th indicates a high correlation between two variables, but only
            if the p-value associated with this coefficient is smaller than cat_pvalue_th;
        :param cat_pvalue_th: check the description for the parameter cat_corr_th for more
            information;
        """
        if method_cat_cat is not None and method_cat_cat not in self.VALID_CAT:
            raise ValueError(
                f"ERROR: invalid value for parameter 'method_num_cat'. Expected a list with the possible values: {self.VALID_CAT}."
            )

        err_float_01(cat_corr_th, "cat_corr_th")
        err_float_01(cat_pvalue_th, "cat_pvalue_th")

        self.method_cat_cat = method_cat_cat
        self.cat_corr_th = cat_corr_th
        self.cat_pvalue_th = cat_pvalue_th

    # -----------------------------------
    def _set_tie_method(self, tie_method: str):
        if tie_method not in self.VALID_TIE_METHODS:
            raise ValueError(
                f"ERROR: invalid value for parameter 'tie_method'. Expected a string within the possible values: {self.VALID_TIE_METHODS}"
            )
        self.tie_method = tie_method

    # -----------------------------------
    def _set_possible_cor_features(self, cor_features: list = None):
        """
        Set the list of columns that will be checked for correlations.

        :param cor_features: a list of the column names or indexes
            that should have their correlations checked. If None, all
            columns are checked for correlations, where each correlation
            is checked in pairs (all possible column pairs are checked);
        """
        if cor_features is not None:
            self.cor_features = cor_features

        if self.df is not None:
            if self.cor_features is None:
                self.cor_features = self.df.columns.values.tolist()
            else:
                self.cor_features = self._check_error_col_list(self.df, self.cor_features, "cor_features")

            cat_col = get_cat_cols(self.df)
            self.feature_types = dict((feat, str) if feat in cat_col else (feat, int) for feat in self.df.columns)

    # -----------------------------------
    def _num_num_correlation(self, num_x: pd.Series, num_y: pd.Series):
        """
        Computes the different correlation metrics for each of the methods specified by
        the method_num_num parameter (from the constructor method). Returns a dictionary
        with the results for each of these methods.

        :param num_x: the first numerical column;
        :param num_y: the second numerical column.
        """
        if self.method_num_num is None:
            return None

        result = {
            self.TYPE_CORR_KEY: self.TYPE_CORR_NN,
        }
        for method in self.method_num_num:
            func = self.VALID_NUM_COR[method]
            corr, p = func(num_x, num_y)
            is_correlated = False
            if abs(corr) > self.num_corr_th and p < self.num_pvalue_th:
                is_correlated = True
            result[method] = {self.CORR_KEY: corr, self.PVAL_KEY: p, self.CORRELATED_KEY: is_correlated}

        is_correlated = True
        for method in self.method_num_num:
            if not result[method][self.CORRELATED_KEY]:
                is_correlated = False
                break

        result[self.COR_TH_KEY] = self.num_corr_th
        result[self.PVALUE_TH_KEY] = self.num_pvalue_th

        if self.compute_exact_matches:
            result[self.EXACT_MATCH_KEY] = _get_exact_matches(num_x, num_y)
        result[self.FINAL_CORR_KEY] = is_correlated

        return result

    # -----------------------------------
    def _print_num_num_summary(self, result: dict):
        for method in self.method_num_num:
            print(
                f"\t* {method} correlation = {result[method][self.CORR_KEY]} with a p-value of {result[method][self.PVAL_KEY]}"
            )

    # -----------------------------------
    def _anova_test_omega_squared(self, anova_table: pd.DataFrame):
        """
        Ideas extracted from pythonfordatascience.org/anova-python/

        Computes the F statistic, p-value, and the omega squared metric given
        an ANOVA table.

        :param anova_table: a table containing the results of the ANOVA test.
            This parameter must be the table returned by sm.stats.anova_lm.
        """
        anova_table["mean_sq"] = anova_table[:]["sum_sq"] / anova_table[:]["df"]
        anova_table["eta_sq"] = anova_table[:-1]["sum_sq"] / sum(anova_table["sum_sq"])

        anova_table["omega_sq"] = (
            anova_table[:-1]["sum_sq"] - (anova_table[:-1]["df"] * anova_table["mean_sq"][-1])
        ) / (sum(anova_table["sum_sq"]) + anova_table["mean_sq"][-1])

        cols = ["sum_sq", "df", "mean_sq", "F", "PR(>F)", "eta_sq", "omega_sq"]
        anova_table = anova_table[cols]
        omega = anova_table["omega_sq"].values.tolist()[0]
        pvalue = anova_table["PR(>F)"].values.tolist()[0]
        F = anova_table["F"].values.tolist()[0]
        return F, pvalue, omega

    # -----------------------------------
    def _num_cat_correlation_anova(self, num_x: pd.DataFrame, cat_y: pd.DataFrame):
        """
        Computes the numerical x categorical correlation metrics using the ANOVA test.
        Returns a dictionary with the results of the ANOVA test.

        :param num_x: the numerical column;
        :param cat_y: the categorical column.
        """
        result = {self.TYPE_CORR_KEY: self.TYPE_CORR_NC}
        df = pd.DataFrame()
        df["num"] = num_x
        df["cat"] = cat_y
        sample_list = []
        categories = df["cat"].unique()
        for cat in categories:
            sample_list.append(df[df["cat"] == cat]["num"])

        is_correlated = False
        # test the homoscedasticity of the data, that is, if each category
        # has a similar variance. This is a condition for the ANOVA test
        # B, p = stats.bartlett(*sample_list)
        B, levene_p = stats.levene(*sample_list)

        # high p-values indicates that the null hypothesis is accepted,
        # where the null hypothesis is that each category has a similar variance
        valid_anova = False
        if levene_p > self.levene_pvalue:
            valid_anova = True

        # Perform the ANOVA test
        model = ols("num ~ C(cat)", data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        F, pvalue, omega = self._anova_test_omega_squared(anova_table)

        # Correlation exists if the data is homoscedastic, the ANOVA p-value
        # is lower then a given threshold, and the omega squared is above a threshold
        if valid_anova and pvalue < self.anova_pvalue and abs(omega) > self.omega_th:
            is_correlated = True

        result[self.ANOVA_KEY] = {
            self.PVAL_LEVENE_KEY: levene_p,
            self.ANOVA_VALID_KEY: valid_anova,
            self.CORR_KEY: F,
            self.PVAL_KEY: pvalue,
            self.OMEGA_KEY: omega,
        }

        result[self.PVAL_TH_LEVENE_KEY] = self.levene_pvalue
        result[self.PVAL_ANOVA_KEY] = self.anova_pvalue
        result[self.OMEGA_TH_KEY] = self.omega_th

        result[self.FINAL_CORR_KEY] = is_correlated

        return result

    # -----------------------------------
    def _num_cat_correlation_jensen(self, num_x: pd.Series, cat_y: pd.Series):
        """
        Computes the numerical x categorical correlation metrics using the
        Jensen-Shannon Distance metric between the distribution functions
        of the numerical feature (num_x) of each pair of clusters (each
        cluster is assigned to a different value of the categorical feature
        cat_y). Returns a dictionary with the results of the Jensen metric.

        :param num_x: the numerical column;
        :param cat_y: the categorical column.
        """
        # initialize variables
        result = {self.TYPE_CORR_KEY: self.TYPE_CORR_NC}
        df = pd.DataFrame()
        df["num"] = num_x
        df["cat"] = cat_y
        sample_list = []
        hist_list = []
        min_value = num_x.min()
        max_value = num_x.max()
        categories = df["cat"].unique()

        # set the number of bins used to create the data distributions
        bins = self.jensen_n_bins
        if bins is None:
            bins = freedman_diaconis(num_x)

        # create the distribution of numerical variable num_x for each
        # possible categorical value in cat_y
        for cat in categories:
            sample = df[df["cat"] == cat]["num"]
            hist, bin_sep = np.histogram(sample, bins=bins, range=(min_value, max_value), density=True)
            sample_list.append(sample)
            hist_list.append(hist)

        # compute the Jensen-Shanon distance metric between each pair
        # of distributions created previously. For each distance, test
        # if the distance is smaller than the threshold self.jensen_th.
        # In this case, no correlation is considered
        is_correlated = True
        summary = {}
        for i in range(len(hist_list) - 1):
            for j in range(i + 1, len(hist_list)):
                p = hist_list[i]
                q = hist_list[j]
                coef = distance.jensenshannon(p, q, base=2.0)
                summary[f"jensen {categories[i]} x {categories[j]}"] = coef
                if coef < self.jensen_th:
                    is_correlated = False

        result[self.JENSEN_KEY] = summary
        result[self.JENSEN_BIN_KEY] = bins
        result[self.JENSEN_TH_KEY] = self.jensen_th
        result[self.FINAL_CORR_KEY] = is_correlated

        return result

    # -----------------------------------
    def _num_cat_correlation_model(self, num_x: pd.Series, cat_y: pd.Series):
        """
        Computes the numerical x categorical correlation metrics based on
        a model trained to predict the categorical value of an instance
        considering only the numerical feature. Models that achieve a good
        score (according the the metrics specified by the model_metrics
        parameter of the constructor method) indicate a high correlation
        between the numerical and categorical features. Returns a dictionary
        with the results of the model approach.

        :param num_x: the numerical column;
        :param cat_y: the categorical column.
        """
        # initialize variables
        result = {self.TYPE_CORR_KEY: self.TYPE_CORR_NC}
        df = pd.DataFrame()
        df["num"] = num_x
        df["cat"] = cat_y
        n_class = cat_y.nunique()

        # encode the categorical column using ordinal encoding
        encode = EncoderOrdinal(df=df, col_encode=["cat"])
        encode.fit()
        df = encode.transform(df)

        # split the two columns into train and test
        # sets and train the model
        error = False
        try:
            train_x, test_x, train_y, test_y = train_test_split(
                df[["num"]], df["cat"], test_size=self.TEST_SIZE, stratify=df["cat"]
            )
            model = self.MODEL
            model.fit(train_x, train_y)
        except:
            error = True

        if not error:
            # use the model to predict the test set and compute all metrics
            pred = model.predict(test_x)
            test_y_oh = ordinal_to_onehot(test_y, n_class)
            pred_oh = ordinal_to_onehot(pred, n_class)
            if df["cat"].nunique() > 2:
                roc_auc = roc_auc_score(test_y_oh, pred_oh, multi_class="ovr")
            else:
                roc_auc = roc_auc_score(test_y, pred)
            p, r, f1, s = precision_recall_fscore_support(test_y, pred, average="macro", zero_division=0)
            acc = accuracy_score(test_y, pred)
            # save all the metrics
            result[self.METRIC_KEY] = {"precision": p, "recall": r, "f1": f1, "auc": roc_auc, "accuracy": acc}
            result[self.METRIC_TH_KEY] = self.metric_th
            result[self.METRIC_USED_KEY] = f"{self.model_metrics}"
            # test for all the metrics being used (self.model_metrics) if all
            # of them all above the threshold specified (self.metric_th)
            is_correlated = True
            for metric in self.model_metrics:
                if result[self.METRIC_KEY][metric] < self.metric_th:
                    is_correlated = False
                    break
            result[self.FINAL_CORR_KEY] = is_correlated
        else:
            result[self.METRIC_KEY] = "Error while splitting data. Too few data for one of the classes."
            result[self.FINAL_CORR_KEY] = False

        return result

    # -----------------------------------
    def _print_num_cat_summary(self, result: dict):
        """
        Prints the results obtained for a numerical x categorical correlation
        analysis. The format depends on which correlation method is chosen.

        :param result: dictionary containing the correlation metrics.
        """
        if self.method_num_cat == "anova":
            print(
                "\tANOVA results:\n"
                + f"\tP-Value for the Levene's Test of Homoscedasticity = {result[self.ANOVA_KEY][self.PVAL_LEVENE_KEY]}\n"
                + f"\tValid Anova (Homoscedasticity data)?: {result[self.ANOVA_KEY][self.ANOVA_VALID_KEY]}\n"
                + f"\tANOVA F-Value = {result[self.ANOVA_KEY][self.CORR_KEY]}\n"
                + f"\tP-Value for the ANOVA F-Value = {result[self.ANOVA_KEY][self.PVAL_KEY]}\n"
                + f"\tOmega Squared = {result[self.ANOVA_KEY][self.OMEGA_KEY]}"
            )
        elif self.method_num_cat == "jensen":
            print("\tJensen-Shannon results:")
            for key in result[self.JENSEN_KEY].keys():
                print(f"\t{key} = {result[self.JENSEN_KEY][key]}")
        else:
            print("\tModel metrics:")
            for key in result[self.METRIC_KEY].keys():
                print(f"\t{key} = {result[self.METRIC_KEY][key]}")

    # -----------------------------------
    def _check_min_instances(self, cat_col: pd.Series):
        """
        Checks if a given categorical column is valid. Valid categorical columns
        are those with 2 or more classes with more than the ​minimum number of
        instances (self.MIN_NUM_INST). If a column has only one class with the
        minimum number of instances, the correlation results could be invalid.

        :param cat_col: the categorical column to be analyzed.
        """
        counts = cat_col.value_counts().values
        valid_class = 0
        for count in counts:
            if count >= self.MIN_NUM_INST:
                valid_class += 1
        if valid_class > 1:
            return True
        return False

    # -----------------------------------
    def _num_cat_correlation(self, num_x: pd.Series, cat_y: pd.Series):
        """
        Computes the numerical x categorical correlation metrics based on
        the method chosen, defined by the method_num_cat parameter passed
        to the constructor method. Returns a dictionary with the results
        based on the selected method.

        :param num_x: the numerical column;
        :param cat_y: the categorical column.
        """
        if self.method_num_cat is None:
            return None

        valid_y = self._check_min_instances(cat_y)
        if not valid_y:
            result = {self.TYPE_CORR_KEY: self.TYPE_CORR_NC}
            result[self.METRIC_KEY] = "Categorical feature has a low class variation."
            result[self.FINAL_CORR_KEY] = False
            return result

        if self.method_num_cat == "anova":
            result = self._num_cat_correlation_anova(num_x, cat_y)
        elif self.method_num_cat == "jensen":
            result = self._num_cat_correlation_jensen(num_x, cat_y)
        else:
            result = self._num_cat_correlation_model(num_x, cat_y)

        return result

    # -----------------------------------
    def _cat_cat_correlation(self, cat_x: pd.Series, cat_y: pd.Series):
        """
        Computes the categorical x categorical correlation metrics using
        the Cramer's V method. Returns a dictionary with the results
        obtained with Cramer's V.

        :param cat_x: the first categorical column;
        :param cat_y: the second categorical column.
        """
        if self.method_cat_cat is None:
            return None

        valid_x = self._check_min_instances(cat_x)
        valid_y = self._check_min_instances(cat_y)
        if not valid_x or not valid_y:
            result = {self.TYPE_CORR_KEY: self.TYPE_CORR_NC}
            result[self.CRAMER_KEY] = "At least one of the categorical features has a low class variation."
            result[self.FINAL_CORR_KEY] = False
            return result

        result = {self.TYPE_CORR_KEY: self.TYPE_CORR_CC}
        ctab, chitest = rp.crosstab(cat_x, cat_y, test="chi-square")
        cramer = chitest["results"][2]
        pvalue = chitest["results"][1]
        is_correlated = False
        if cramer > self.cat_corr_th and pvalue < self.cat_pvalue_th:
            is_correlated = True

        result[self.CRAMER_KEY] = {self.CORR_KEY: cramer, self.PVAL_KEY: pvalue}
        result[self.COR_TH_KEY] = self.cat_corr_th
        result[self.PVALUE_TH_KEY] = self.cat_pvalue_th
        if self.compute_exact_matches:
            result[self.EXACT_MATCH_KEY] = _get_exact_matches(cat_x, cat_y)
        result[self.FINAL_CORR_KEY] = is_correlated

        return result

    # -----------------------------------
    def _print_cat_cat_summary(self, result: dict):
        print(
            f"\t* {self.CRAMER_KEY} = {result[self.CRAMER_KEY][self.CORR_KEY]} "
            + f"with a p-value of {result[self.CRAMER_KEY][self.PVAL_KEY]}"
        )

    # -----------------------------------
    def _print_summary(self, result: dict, force_print: bool = False):
        """
        Print the results of a given correlation analysis. A specific
        method is called to print the results based on the correlation
        type (numerical x numerical, numerical x categorical, or
        categorical x categorical).

        :param result: the dictionary containing the results of the
            correlation metrics;
        :param force_print: if True, print the results only if the
            result dictionary indicates a correlation between the
            two variables analyzed. Otherwise, nothing is printed.
            If False, print results even if no correlation is
            detected.
        """
        if not result[self.FINAL_CORR_KEY] and not force_print:
            return
        if result[self.TYPE_CORR_KEY] == self.TYPE_CORR_NN:
            self._print_num_num_summary(result)
        elif result[self.TYPE_CORR_KEY] == self.TYPE_CORR_NC:
            self._print_num_cat_summary(result)
        else:
            self._print_cat_cat_summary(result)

    # -----------------------------------
    def _build_correlation_summary(self):
        """
        Builds a series of internal attributes that stores information regarding the
        different correlations found in the dataset. self.corr_dict stores the result
        dictionary for each pair of variables tested. self.feat_info is a dictionary
        that stores, for each variable, the number of variables correlated to it and
        a list of these variables. self.edge_list is another internal variable created
        that stores a list of edges of the correlation graph: each variable represents
        a different node in this graph and an edge between two variables exists if these
        two variables are correlated. This graph is later used to determine which variables
        should be removed from each pair of correlated variables.
        """
        self.feat_info = {}
        for feat in self.cor_features:
            self.feat_info[feat] = {self.DEGREE_KEY: 0, self.CORR_VAR_KEY: [], self.BLOCK_KEY: False}
        self.edge_list = []
        self.corr_dict = {feat: {} for feat in self.cor_features}
        for i, feat_x in enumerate(self.cor_features):
            for j in range(i, len(self.cor_features)):
                if i == j:
                    # don't analyze the correlation between a variable and itself
                    continue
                feat_y = self.cor_features[j]
                x_type = self.feature_types[feat_x]
                y_type = self.feature_types[feat_y]

                # Create a temporary dataframe and drop all rows with Nan
                df_temp = self.df[[feat_x, feat_y]]
                df_temp = df_temp.dropna()
                if df_temp.shape[0] == 0:
                    self.corr_dict[feat_x][feat_y] = {self.TYPE_CORR_KEY: self.TYPE_CORR_CC, self.FINAL_CORR_KEY: False}
                    self.corr_dict[feat_y][feat_x] = {self.TYPE_CORR_KEY: self.TYPE_CORR_CC, self.FINAL_CORR_KEY: False}
                    continue

                # Numerical x Numerical Correlation
                if x_type == int and y_type == int:
                    result = self._num_num_correlation(df_temp[feat_x], df_temp[feat_y])
                # Numerical x Categorical Correlation
                elif x_type == int and y_type == str:
                    result = self._num_cat_correlation(df_temp[feat_x], df_temp[feat_y])
                elif x_type == str and y_type == int:
                    result = self._num_cat_correlation(df_temp[feat_y], df_temp[feat_x])
                # Categorical x Categorical Correlation
                else:
                    result = self._cat_cat_correlation(df_temp[feat_x], df_temp[feat_y])

                if result is None:
                    continue

                self.corr_dict[feat_x][feat_y] = result
                self.corr_dict[feat_y][feat_x] = result

                if result[self.FINAL_CORR_KEY]:
                    self.feat_info[feat_x][self.DEGREE_KEY] += 1
                    self.feat_info[feat_x][self.CORR_VAR_KEY].append(feat_y)
                    self.feat_info[feat_y][self.DEGREE_KEY] += 1
                    self.feat_info[feat_y][self.CORR_VAR_KEY].append(feat_x)
                    self.edge_list.append((feat_x, feat_y))

        # Removes any variables with an empty dictionary
        key_remove = []
        for key in self.corr_dict.keys():
            # Check if the dict associated to key is an empty dict
            if not bool(self.corr_dict[key]):
                key_remove.append(key)

        for key in key_remove:
            self.corr_dict.pop(key)

    # -----------------------------------
    def _build_auxiliary_dicts(self):
        """
        Creates two dictionaries:

            - self.corr_pairs: stores information and correlation metrics for all
              pairs of correlated variables. Each key of this
              dictionary follows the pattern "{key1} x {key2}";
            - self.uncorr_pairs: stores information and correlation metrics for all
              pairs of uncorrelated variables. Each key of this
              dictionary follows the pattern "{key1} x {key2}".

        These dictionaries are later saved in a json file, which can then be
        consulted for a more in-depth analysis by the user.
        """
        corr_pairs = {}
        uncorr_pairs = {}
        for feat_x in self.corr_dict.keys():
            for feat_y in self.corr_dict[feat_x].keys():
                key1 = f"{feat_x} x {feat_y}"
                key2 = f"{feat_y} x {feat_x}"
                if self.corr_dict[feat_x][feat_y][self.FINAL_CORR_KEY]:
                    if key1 not in corr_pairs.keys() and key2 not in corr_pairs.keys():
                        corr_pairs[key1] = self.corr_dict[feat_x][feat_y]
                else:
                    if key1 not in uncorr_pairs.keys() and key2 not in uncorr_pairs.keys():
                        uncorr_pairs[key1] = self.corr_dict[feat_x][feat_y]

        self.corr_pairs = corr_pairs
        self.uncorr_pairs = uncorr_pairs

    # -----------------------------------
    def _get_num_correlations(self, cor_list: list = None):
        """
        Get the number of correlations found. cor_list represents a list
        of correlations. If cor_list is not provided, then use the list
        of correlations previously computed (self.edge_list).

        :param cor_list: list of correlations. If None, use the list of
            correlations previously computed (self.edge_list).
        """
        if cor_list is None:
            cor_list = self.edge_list
        return len(cor_list)

    # -----------------------------------
    def _get_most_correlated_feature(self, feat_degree: dict):
        """
        Returns the feature with the greatest number of correlations.
        There are two values returned:

            - selected: the unblocked feature identifier with the largest
              amount of correlations;
            - selected: the feature identifier (blocked or unblocked) with
              the largest amount of correlations;

        All features start as unblocked. Whenever a feature f_x is selected
        to be removed, all the other features f_y correlated to f_x are
        blocked. This avoids removing both features of a correlated pair,
        which could result in losing some information. This could happen,
        for example, if we have the correlated pairs (f_1, f_2), (f_1, f_3),
        and (f_2, f_4), and then we select to remove features f_1 from the
        first pair (f_1, f_2) and feature f_2 from the pair (f_2, f_4). Since
        f_1 was already removed, no other feature needs to be removed for the
        pair (f_1, f_3). In this scenario, we removed both f_1 and f_2 from
        the pair (f_1, f_2), which means that we could lose some information
        by dropping both of the correlated variables. The blocking mechanism
        is to try to avoid this scenario.

        :param feat_degree: a dictionary that has one key for each feature in
            the dataset, and the value of each key is the degree of each feature.
            The degree of a feature represents the number of other features that
            are correlated to it.
        """
        max_degree = 0
        selected = None
        max_degree_blocked = 0
        selected_blocked = None
        for feat in feat_degree.keys():
            degree = feat_degree[feat]
            blocked = self.feat_info[feat][self.BLOCK_KEY]
            if degree > max_degree and not blocked:
                max_degree = degree
                selected = feat
            if degree > max_degree_blocked:
                max_degree_blocked = degree
                selected_blocked = feat

        return selected, selected_blocked

    # -----------------------------------
    def _get_feature_more_missing(self, feat_x: str, feat_y: str):
        """
        Given two features (feat_x and feat_y), return the feature that
        has more missing values. This is used to determine which feature
        should be dropped (considering that feat_x and feat_y are correlated).

        :param feat_x: the first feature name being compared;
        :param feat_x: the second feature name being compared.
        """
        if self.missing_count[feat_x] > self.missing_count[feat_y]:
            return feat_x
        return feat_y

    # -----------------------------------
    def _get_feature_least_cadinality(self, feat_x: str, feat_y: str):
        """
        Given two features (feat_x and feat_y), return the feature that
        has the lowest ​cardinality, that is, different number of unique
        values. This is used to determine which feature should be dropped
        (considering that feat_x and feat_y are correlated).

        :param feat_x: the first feature name being compared;
        :param feat_x: the second feature name being compared.
        """
        if self.feat_nunique[feat_x] < self.feat_nunique[feat_y]:
            return feat_x
        return feat_y

    # -----------------------------------
    def _get_feature_least_variance(self, feat_x: str, feat_y: str):
        """
        Given two features (feat_x and feat_y), return the feature that
        has the lowest variance. If one of the features is categorical
        and the other is numerical, keep the numerical variable, since
        numerical variables usually have a greater cardinality. If both
        features are categorical, call the _get_feature_least_cadinality.
        A greater variance could mean more information being provided by
        the variable, and so it is better to remove the one with the least
        variance. This is used to determine which feature should be dropped
        (considering that feat_x and feat_y are correlated).

        :param feat_x: the first feature name being compared;
        :param feat_x: the second feature name being compared.
        """
        selected = feat_x
        x_type = self.feature_types[feat_x]
        y_type = self.feature_types[feat_y]
        # If one variable is numerical and the other categorical, keep the numerical one
        if x_type != y_type:
            if x_type == str:
                return feat_x
            return feat_y
        # If both variables are categorical, use the cardinality method instead
        elif x_type == str:
            selected = self._get_feature_least_cadinality(feat_x, feat_y)
        else:
            if self.dispersion_index[feat_x] > self.dispersion_index[feat_y]:
                selected = feat_y

        return selected

    # -----------------------------------
    def _select_feature_tie(self, feat_x: str, feat_y: str):
        """
        Select which feature should be removed in case there are
        two features with the same degree in the correlations graph
        (check the documentation of the _build_correlation_summary
        method for more details about this graph). There are three
        tie resolution methods: (i) remove the variable with the
        largest number of missing values, (ii) remove the variable
        with the lowest cardinality, or (iii) remove the variable
        with the lowest variance.

        :param feat_x: the first correlated feature;
        :param feat_y: the second correlated feature.
        """
        if self.tie_method == "missing":
            return self._get_feature_more_missing(feat_x, feat_y)
        elif self.tie_method == "cardinality":
            return self._get_feature_least_cadinality(feat_x, feat_y)
        return self._get_feature_least_variance(feat_x, feat_y)

    # -----------------------------------
    def _select_best_feat_remove(self, base_feature: str, feat_degree: dict):
        """
        Check if there are other features ​apart from the selected feature
        (base_feature) that has the same number of correlated features. If
        that is the case, check which of these two features should be removed
        using the _select_feature_tie method.

        :param base_feature: the identifier of the feature being analyzed;
        :param feat_degree: an updated dictionary that stores the degree
            of a feature considering the current correlations graph.
        """
        selected = base_feature
        for feat in self.feat_info[base_feature][self.CORR_VAR_KEY]:
            if feat not in feat_degree.keys():
                continue
            if feat_degree[base_feature] == feat_degree[feat]:
                selected = self._select_feature_tie(base_feature, feat)
                return selected
        return selected

    # -----------------------------------
    def _remove_edges(self, edge_list: list, vertex: str):
        """
        Given a correlation graph (given by edge_list) and a feature
        that we want to remove from this graph (vertex), remove this
        feature and block all features correlated to the feature
        identified by the vertex parameter. Returns an updated
        graph (edge_list) with 'vertex' removed.

        :param edge_list: an updated list of edges in the correlations
            graph;
        :param vertex: the vertex (feature) that will be removed
            from the correlations graph.
        """
        rem_indices = []
        for i, edge in enumerate(edge_list):
            if edge[0] == vertex or edge[1] == vertex:
                rem_indices.append(i)
                if edge[0] != vertex:
                    self.feat_info[edge[0]][self.BLOCK_KEY] = True
                else:
                    self.feat_info[edge[1]][self.BLOCK_KEY] = True

        for i in sorted(rem_indices, reverse=True):
            edge_list.pop(i)
        return edge_list

    # -----------------------------------
    def _get_features_to_remove(self):
        """
        Returns the list of features that should be removed according to the
        analysis of the correlations graph (check the documentation for the
        _build_correlation_summary method for more information on this graph).
        The main idea is the following: using the correlations graph, select
        the feature (vertices) with the highest degree (features with the
        largest number of other correlated features) and remove it. If there
        are features with the same degree, use the _select_best_feat_remove
        method to solve the tie. When a feature is removed (using the
        _remove_edges method), all of its neighbors are blocked (check the
        documentation of the _get_most_correlated_feature method for more
        information about blocked features), and all edges connected to it are
        removed.
        """
        feature_to_remove = []
        # If there are no correlations, just exit
        if self._get_num_correlations() == 0:
            self.print_message("No correlations detected. Nothing to be done here.")
            return feature_to_remove

        # Create a copy of the degree of each feature and a copy of the edges list
        feat_degree_aux = {feat: self.feat_info[feat][self.DEGREE_KEY] for feat in self.feat_info.keys()}
        edges_temp = self.edge_list.copy()

        while self._get_num_correlations(edges_temp) > 0:
            selected, selected_blocked = self._get_most_correlated_feature(feat_degree_aux)
            if selected is None:
                selected = selected_blocked
                if selected_blocked is None:
                    raise ValueError("DEBUG ERROR: both selected and selected_blocked are None.")

            selected = self._select_best_feat_remove(selected, feat_degree_aux)

            edges_temp = self._remove_edges(edges_temp, selected)
            for feat in self.feat_info[selected][self.CORR_VAR_KEY]:
                if feat in feat_degree_aux.keys():
                    feat_degree_aux[feat] -= 1
            feat_degree_aux.pop(selected)

            feature_to_remove.append(selected)

        return feature_to_remove

    # -----------------------------------
    def _save_jsons(self):
        if not self.save_json:
            return

        if self.json_summary is not None:
            with open(self.json_summary, "w") as json_file:
                json.dump(self.corr_dict, json_file)

        if self.json_corr is not None:
            with open(self.json_corr, "w") as json_file:
                json.dump(self.corr_pairs, json_file)

        if self.json_uncorr is not None:
            with open(self.json_uncorr, "w") as json_file:
                json.dump(self.uncorr_pairs, json_file)

    # -----------------------------------
    def _fit(self):
        """
        Steps for running the fit method for the current class.
        """
        if self.tie_method == "missing":
            self.missing_count = self.df.isna().sum()
        elif self.tie_method == "var":
            self.dispersion_index = self.df.std(numeric_only=True) / (
                self.df.max(numeric_only=True) - self.df.min(numeric_only=True)
            )
            self.feat_nunique = self.df.nunique()
        else:
            self.feat_nunique = self.df.nunique()
        self._set_possible_cor_features()
        self._build_correlation_summary()
        self._build_auxiliary_dicts()
        self._save_jsons()

    # -----------------------------------
    def _get_selected_features(self):
        """
        Overwrites the _get_selected_features from the FeatureSelection
        class. Returns the list of features that should be kept in the
        dataset according to the correlation graph analysis method (check
        the documentation of the _get_features_to_remove method for more
        information on this approach).
        """
        feat_remove = self._get_features_to_remove()
        if self.df is None:
            raise ValueError(
                "ERROR: trying to set the selected features without providing a dataset. "
                + "Use the fit method prior to getting the selected features."
            )
        features = [feat for feat in self.df.columns if feat not in feat_remove]
        return features

    # -----------------------------------
    def _update_thresholds_num_num(self, num_corr_th: float = None, num_pvalue_th: float = None):
        if num_corr_th is None and num_pvalue_th is None:
            return False

        if num_corr_th is not None:
            self.num_corr_th = num_corr_th
        if num_pvalue_th is not None:
            self.num_pvalue_th = num_pvalue_th

        self._set_numerical_corr_param(self.method_num_num, self.num_corr_th, self.num_pvalue_th)

        return True

    # -----------------------------------
    def _update_thresholds_num_cat(
        self,
        levene_pvalue: float = None,
        anova_pvalue: float = None,
        omega_th: float = None,
        jensen_th: float = None,
        model_metrics: float = None,
        metric_th: float = None,
    ):
        param_mod = False
        if levene_pvalue is not None:
            param_mod = True
            self.levene_pvalue = levene_pvalue
        if anova_pvalue is not None:
            param_mod = True
            self.anova_pvalue = anova_pvalue
        if omega_th is not None:
            param_mod = True
            self.omega_th = omega_th
        if jensen_th is not None:
            param_mod = True
            self.jensen_th = jensen_th
        if model_metrics is not None:
            param_mod = True
            self.model_metrics = model_metrics
        if metric_th is not None:
            param_mod = True
            self.metric_th = metric_th

        if not param_mod:
            return False

        self._set_num_cat_corr_param(
            self.method_num_cat,
            self.levene_pvalue,
            self.anova_pvalue,
            self.omega_th,
            self.jensen_n_bins,
            self.jensen_th,
            self.model_metrics,
            self.metric_th,
        )

        return True

    # -----------------------------------
    def _update_thresholds_cat_cat(self, cat_corr_th: float = None, cat_pvalue_th: float = None):
        if cat_corr_th is None and cat_pvalue_th is None:
            return False

        if cat_corr_th is not None:
            self.cat_corr_th = cat_corr_th
        if cat_pvalue_th is not None:
            self.cat_pvalue_th = cat_pvalue_th

        self._set_cat_corr_param(self.method_cat_cat, self.cat_corr_th, self.cat_pvalue_th)

        return True

    # -----------------------------------
    def _update_correlation_info_num_num(self, result: dict):
        if self.method_num_num is None:
            return result

        final_corr = True
        for method in self.method_num_num:
            corr = result[method][self.CORR_KEY]
            p = result[method][self.PVAL_KEY]
            if abs(corr) > self.num_corr_th and p < self.num_pvalue_th:
                result[method][self.CORRELATED_KEY] = True
            if not result[method][self.CORRELATED_KEY]:
                final_corr = False

        result[self.COR_TH_KEY] = self.num_corr_th
        result[self.PVALUE_TH_KEY] = self.num_pvalue_th
        result[self.FINAL_CORR_KEY] = final_corr

        return result

    # -----------------------------------
    def _update_correlation_info_anova(self, result: dict):
        valid_anova = False
        levene_p = result[self.ANOVA_KEY][self.PVAL_LEVENE_KEY]
        if levene_p > self.levene_pvalue:
            valid_anova = True

        pvalue = result[self.ANOVA_KEY][self.PVAL_KEY]
        omega = result[self.ANOVA_KEY][self.OMEGA_KEY]

        is_correlated = False
        if valid_anova and pvalue < self.anova_pvalue and abs(omega) > self.omega_th:
            is_correlated = True

        result[self.ANOVA_KEY][self.ANOVA_VALID_KEY] = valid_anova
        result[self.PVAL_TH_LEVENE_KEY] = self.levene_pvalue
        result[self.PVAL_ANOVA_KEY] = self.anova_pvalue
        result[self.OMEGA_TH_KEY] = self.omega_th
        result[self.FINAL_CORR_KEY] = is_correlated

        return result

    # -----------------------------------
    def _update_correlation_info_jensen(self, result: dict):
        is_correlated = True
        for key, coef in result[self.JENSEN_KEY].items():
            if coef < self.jensen_th:
                is_correlated = False
                break

        result[self.JENSEN_TH_KEY] = self.jensen_th
        result[self.FINAL_CORR_KEY] = is_correlated

        return result

    # -----------------------------------
    def _update_correlation_info_model(self, result: dict):
        result[self.METRIC_TH_KEY] = self.metric_th
        result[self.METRIC_USED_KEY] = f"{self.model_metrics}"

        is_correlated = True
        for metric in self.model_metrics:
            if result[self.METRIC_KEY][metric] < self.metric_th:
                is_correlated = False
                break
        result[self.FINAL_CORR_KEY] = is_correlated

        return result

    # -----------------------------------
    def _update_correlation_info_num_cat(self, result: dict):
        if self.method_num_cat is None:
            return result

        if self.method_num_cat == "anova":
            result = self._update_correlation_info_anova(result)
        elif self.method_num_cat == "jensen":
            result = self._update_correlation_info_jensen(result)
        else:
            result = self._update_correlation_info_model(result)
        return result

    # -----------------------------------
    def _update_correlation_info_cat_cat(self, result: dict):
        if self.method_cat_cat is None:
            return result

        is_correlated = False
        cramer = result[self.CRAMER_KEY][self.CORR_KEY]
        pvalue = result[self.CRAMER_KEY][self.PVAL_KEY]
        if cramer > self.cat_corr_th and pvalue < self.cat_pvalue_th:
            is_correlated = True

        result[self.COR_TH_KEY] = self.cat_corr_th
        result[self.PVALUE_TH_KEY] = self.cat_pvalue_th
        result[self.FINAL_CORR_KEY] = is_correlated

        return result

    # -----------------------------------
    def _update_correlation_info(self, feat_x: str, feat_y: str):
        x_type = self.feature_types[feat_x]
        y_type = self.feature_types[feat_y]
        result = self.corr_dict[feat_x][feat_y]

        # Numerical x Numerical Correlation
        if x_type == int and y_type == int:
            result = self._update_correlation_info_num_num(result)
        # Categorical x Categorical Correlation
        elif x_type == str and y_type == str:
            result = self._update_correlation_info_cat_cat(result)
        # Numerical x Categorical Correlation
        else:
            result = self._update_correlation_info_num_cat(result)

        return result

    # -----------------------------------
    def _update_correlation_dict(self):
        self.feat_info = {}
        for feat in self.cor_features:
            self.feat_info[feat] = {self.DEGREE_KEY: 0, self.CORR_VAR_KEY: [], self.BLOCK_KEY: False}
        self.edge_list = []

        for feat_x in self.corr_dict.keys():
            for feat_y in self.corr_dict[feat_x].keys():
                result = self._update_correlation_info(feat_x, feat_y)
                self.corr_dict[feat_x][feat_y] = result
                self.corr_dict[feat_y][feat_x] = result

                if result[self.FINAL_CORR_KEY]:
                    self.feat_info[feat_x][self.DEGREE_KEY] += 1
                    self.feat_info[feat_x][self.CORR_VAR_KEY].append(feat_y)
                    self.feat_info[feat_y][self.DEGREE_KEY] += 1
                    self.feat_info[feat_y][self.CORR_VAR_KEY].append(feat_x)
                    self.edge_list.append((feat_x, feat_y))

    # -----------------------------------
    def update_selected_features(
        self,
        num_corr_th: float = None,
        num_pvalue_th: float = None,
        levene_pvalue: float = None,
        anova_pvalue: float = None,
        omega_th: float = None,
        jensen_th: float = None,
        model_metrics: float = None,
        metric_th: float = None,
        cat_corr_th: float = None,
        cat_pvalue_th: float = None,
        save_json: bool = None,
        json_summary: str = None,
        json_corr: str = None,
        json_uncorr: str = None,
    ):
        """
        Update different parameters associated to the different types of correlations and
        recomputes the selected features using these new parameter values without recomputing
        the correlations. This method allow users to change certain thresholds and metrics
        without requiring to recompute all of the correlations, which can be computationally
        expensive depending on the size of the dataset. The only parameters allowed to be
        changed are the ones accepted by this method. If another parameter not listed here
        needs to be changed, then it is necessary to instantiate a different object and call
        the :meth:`fit` method again.

        :param num_corr_th: the correlation coefficient value used as a threshold for
            considering if there is a correlation between two numerical variables.
            That is, given two variables with a correlation coefficient of 'x' (depends on
            the correlation used, specified by ``method_num_num``), a correlation is considered
            only if abs(x) >= method_num_num and if the associated p-value 'p' is smaller than
            'p' <= num_pvalue_th;

        :param num_pvalue_th: the p-value used as a threshold when considering if there is a
            correlation between two variables. That is, given two variables with a correlation
            coefficient of 'x' (depends on the correlation used, specified by ``method_num_num``),
            a correlation is considered only if abs(x) >= method_num_num and if the associated
            p-value 'p' is smaller than 'p' <= num_pvalue_th;

        :param levene_pvalue: the threshold used to check if a set of samples are homoscedastic (similar
            variances across samples). This condition is necessary for the ANOVA test. This check is done
            using the Levene test, which considers that all samples have similar variances as the null
            hypothesis. If the p-value associated with this test is high, then the null hypothesis is accepted,
            thus allowing the ANOVA test to be carried out. This parameter defines the threshold used by the
            p-value of this test: if p-value > levene_pvalue, then the data is considered to be homoscedastic.
            This parameter is ignored if method_num_cat != 'anova';

        :param anova_pvalue: threshold used by the p-value associated with the F-statistic computed by the ANOVA
            test. If the p-value < anova_pvalue, then we consider that there is a statistically significant
            difference between the numerical values of different clusters (clusterized according to the values
            of the categorical variable). This implies a possible correlation between the numerical and
            categorical variables, although the F-statistic doesn't tell us the magnitude of this difference.
            For that, we use the Omega-Squared metric. This parameter is ignored if method_num_cat != 'anova';

        :param omega_th: the threshold used for the omega squared metric. The omega squared is a metric that
            varies between 0 and 1 that indicates the effect of a categorical variable over the variance of
            a numerical variable. A value closer to 0 indicates a weak effect, while values closer to 1 show
            that the categorical variable has a significant impact on the variance of the numerical variable,
            thus showing a high correlation. If the omega squared is greater than ``omega_th``, then both variables
            being analyzed are considered to be correlated. This parameter is ignored if method_num_cat != 'anova';

        :param jensen_th: when method_num_cat = 'jensen', we compare the distribution of each cluster of data
            using the Jensen-Shannon distance metric. If the distance is close to 1, then the distributions are
            considered different. If all pairs of clusters have a high distance, then both variables being analyzed
            are considered to be correlated. This parameter indicates the threshold used to check if a distance
            metric is high or not: if distance > jensen_th, then the distributions being compared are considered
            different. Must be a float value within [0, 1]. This parameter is ignored if method_num_cat != 'jensen';

        :param model_metrics: a list of metric names that should be used when evaluating if a model trained using
            a single numerical variable to predict a categorical variable is good enough. If the trained model
            presents a good performance for the metrics in ``model_metrics``, then both variables being analyzed are
            considered to be correlated. This parameter must be a list, and the allowed values in this list are:
            ["f1", "auc", "accuracy", "precision", "recall"]. This parameter is ignored if
            method_num_cat != 'model';

        :param metric_th: given the metrics provided by ``model_metrics``, a pair of variables being analyzed are only
            considered correlated if all metrics in ``model_metrics`` achieve a score greater than metric_th over the
            test set (the variables being analyzed are split into training and test set internally). This parameter
            is ignored if method_num_cat != 'model';

        :param cat_corr_th: the threshold used for the Cramer's V correlation coefficient. Values greater than ``cat_corr_th``
            indicates a high correlation between two variables, but only if the p-value associated with this coefficient
            is smaller than ``cat_pvalue_th``;

        :param cat_pvalue_th: check the description for the parameter ``cat_corr_th`` for more information;

        :param save_json: if True, the summary jsons are saved according to the paths ``json_summary``, ``json_corr``, and
            ``json_uncorr``. If False, these json files are not saved;

        :param json_summary: when calling the fit method, all correlations will be computed according to the many
            parameters detailed previously. After computing all this data, everything is saved in a JSON file,
            which can then be accessed and analyzed carefully. We recommend using a JSON viewing tool for this.
            This parameter indicates the name of the file where the JSON should be saved (including the path
            to the file). If set to None no JSON file will be saved;

        :param json_corr: similar to ``json_summary``, but corresponds to the name of the JSON file that contains only
            the information of the pairs of variables considered to be correlated (with no repetitions);

        :param json_uncorr: similar to ``json_summary``, but corresponds to the name of the JSON file that contains only
            the information of the pairs of variables considered NOT to be correlated (with no repetitions);
        """
        update_num_num = self._update_thresholds_num_num(num_corr_th, num_pvalue_th)
        update_num_cat = self._update_thresholds_num_cat(
            levene_pvalue, anova_pvalue, omega_th, jensen_th, model_metrics, metric_th
        )
        update_cat_cat = self._update_thresholds_cat_cat(cat_corr_th, cat_pvalue_th)

        if not update_num_num and not update_num_cat and not update_cat_cat:
            return

        self._update_correlation_dict()
        self._build_auxiliary_dicts()
        self.set_selected_features()

        if save_json is not None:
            self.save_json = save_json
        if json_summary is not None:
            self.json_summary = json_summary
        if json_corr is not None:
            self.json_corr = json_corr
        if json_uncorr is not None:
            self.json_uncorr = json_uncorr

        self._save_jsons()

    # -----------------------------------
    def get_summary(self, print_summary: bool = True):
        """
        Fetches three internal dictionaries:

            - self.corr_dict: stores information and correlation metrics for
              each variable. There is one key for each variable
              analyzed;
            - self.corr_pairs: stores information and correlation metrics for all
              pairs of correlated variables. Each key of this
              dictionary follows the pattern "{key1} x {key2}";
            - self.uncorr_pairs: stores information and correlation metrics for all
              pairs of uncorrelated variables. Each key of this
              dictionary follows the pattern "{key1} x {key2}".

        :param print_summary: if True, print the values stored in the correlated
            and the uncorrelated dictionary. If False, just return the three
            dictionaries previously mentioned.
        :return: three internal dictionaries that summurizes the correlations found.
        :rtype: tuple
        """
        if print_summary:
            print("\nCORRELATION SUMMARY\n")
            cont = 0
            for pair in self.corr_pairs.keys():
                cont += 1
                print(f"{cont} - {pair}:")
                self._print_summary(self.corr_pairs[pair])

            print("\nNOT CORRELATED VARIABLES SUMMARY\n")
            for pair in self.uncorr_pairs.keys():
                print(f"{pair}:")
                self._print_summary(self.uncorr_pairs[pair], force_print=True)

        return self.corr_dict.copy(), self.corr_pairs.copy(), self.uncorr_pairs.copy()

    # -----------------------------------
    def get_correlated_pairs(self):
        """
        Returns a copy of the dictionary mapping all correlated pairs
        found.

        :return: a copy of the dictionary mapping all correlated pairs
            found.
        :rtype: dict
        """
        return self.corr_pairs.copy()
