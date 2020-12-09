#! /usr/bin/env python3
# -*- coding:utf-8 -*-


###############################################################################
# IMPORTS

import collections
import re
import os
import sys
import scipy
import datetime
import warnings

from copy import deepcopy

import numpy as np
import pandas as pd

from statsmodels.stats.multitest import multipletests


import sklearn.preprocessing
import sklearn.metrics
import sklearn.pipeline
from sklearn.externals import joblib

import imblearn.under_sampling
import rfpimp


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pydotplus
import matplotlib.ticker as plticker


mpl.rcParams.update({'figure.autolayout':True})

sns.set(style="whitegrid")
husl = sns.color_palette("husl", 32)
sns.palplot(husl)
plt.close()
plt.style.use('seaborn-talk')

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

mpl.rc('font', size=SMALL_SIZE)          # controls default text sizes
mpl.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
mpl.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
mpl.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
mpl.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
mpl.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
mpl.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



###############################################################################
# MISC


# Used for renaming columns
def human_readable_columns(name):
    if name.startswith('conservation_') or name.startswith('enhtarg_')\
            or name.startswith('chromatinaccess'):
        return name.split('_')[-1]
    if name.startswith('chromatinstates_') or name.startswith('tf_')\
            or name.startswith('genomeAnnots_') or name.startswith('interactions_'):
        return name.split('_',1)[1]
    return name

def replace_nth(string, sub, wanted, n):
    """ Replace the <n>th occurence of <sub> by <wanted> in a string.
    """
    where = [m.start() for m in re.finditer(sub, string)][n - 1]
    before = string[:where]
    after = string[where:]
    after = after.replace(sub, wanted)
    newString = before + after
    return newString



###############################################################################

# PART0: DATAFRAME PREPARATION AND FEATURE EXPLORATION
# ====================================================


def balanced_accuracy_score(y_true, y_pred, sample_weight=None, adjusted=False):
    """Compute the balanced accuracy
    The balanced accuracy in binary and multiclass classification problems to
    deal with imbalanced datasets. It is defined as the average of recall
    obtained on each class.
    The best value is 1 and the worst value is 0 when ``adjusted=False``.
    Read more in the :ref:`User Guide <balanced_accuracy_score>`.
    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) target values.
    y_pred : 1d array-like
        Estimated targets as returned by a classifier.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    adjusted : bool, default=False
        When true, the result is adjusted for chance, so that random
        performance would score 0, and perfect performance scores 1.
    Returns
    -------
    balanced_accuracy : float
    See also
    --------
    recall_score, roc_auc_score
    Notes
    -----
    Some literature promotes alternative definitions of balanced accuracy. Our
    definition is equivalent to :func:`accuracy_score` with class-balanced
    sample weights, and shares desirable properties with the binary case.
    See the :ref:`User Guide <balanced_accuracy_score>`.
    References
    ----------
    .. [1] Brodersen, K.H.; Ong, C.S.; Stephan, K.E.; Buhmann, J.M. (2010).
           The balanced accuracy and its posterior distribution.
           Proceedings of the 20th International Conference on Pattern
           Recognition, 3121-24.
    .. [2] John. D. Kelleher, Brian Mac Namee, Aoife D'Arcy, (2015).
           `Fundamentals of Machine Learning for Predictive Data Analytics:
           Algorithms, Worked Examples, and Case Studies
           <https://mitpress.mit.edu/books/fundamentals-machine-learning-predictive-data-analytics>`_.
    Examples
    --------
    >>> from sklearn.metrics import balanced_accuracy_score
    >>> y_true = [0, 1, 0, 0, 1, 0]
    >>> y_pred = [0, 1, 0, 0, 0, 1]
    >>> balanced_accuracy_score(y_true, y_pred)
    0.625
    """
    C = sklearn.metrics.confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(C) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        warnings.warn('y_pred contains classes not in y_true')
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    if adjusted:
        n_classes = len(per_class)
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    return score


def balance_classes(y):
    """ Return an array of indexes from y such as classes in y are balanced.
    Note: index is reset if y is provided as a vector.
    
    In:
        y (pd.Series or np.array): labels of samples.
    Out:
        np.array: array of indexes of samples
    """
    y = pd.Series(y) if not isinstance(y,pd.Series) else y.reset_index(drop=True)
    class_counts = y.value_counts() 
    min_size = class_counts.min()
    
    indexes = []
    for cl, count in class_counts.iteritems():
        if count>min_size:
            indexes.append(y.loc[y==cl].sample(min_size).index.values)
        else:
            indexes.append(y.loc[y==cl].index.values)
        
    return np.concatenate(indexes)


def summarize_annotations_per_group(X, bool_selected, cols_type, normalize=False):
    """ Return a dataframe of shape N_feature x 2, with summarized values per category.
    
    For each feature in X (which should have an associated column type in <cols_type>),
    samples are separated according to <bool_selected>, a summarizing operation is applied.
    
    For binary: return the ratio of samples with the annotation.
    For discrete / continuous: return the mean.
    
    To be noted that it is possible to summarize the full dataframe without selected any samples,
    by providing an array of True for all samples as <bool_selected>.
    
    In:
        X (pandas.DataFrame): dataframe of N_samples x N_features, with features to summarize.
        bool_selected (pandas.Series): boolean array used to separate samples.
        cols_type (dict): map column types (binary, discrete, continuous) to list of features
        normalize ([bool, default=False]): robust scaling (1perc ignored) of
                                           the discrete / continuous features
                                           before calculating means.
        
    Out:
        pandas.DataFrame: N_features x 2, with columns being ['selected', 'others'].
    """
    all_cols_with_coltypes = list(cols_type.keys())
    
    summarizing_columns = []
    for c in X.columns:
        if c not in all_cols_with_coltypes:
            print("Column '{}' not associated to a col_type. Ignored.".format(c))
        else:
            summarizing_columns.append(c)

    all_summarized_cols = []
    
    for type_col in set(list(cols_type.values())):
        tmp_cols_set = [c for c in summarizing_columns if cols_type[c] == type_col]
        
        if not len(tmp_cols_set)>0: continue
            
        if type_col == 'binary':
            # Return ratios
            for c in tmp_cols_set:
                try:
                    tmp = pd.concat((X.loc[bool_selected,c].value_counts().rename('selected').div(bool_selected.sum()),
                                     X.loc[~bool_selected,c].value_counts().rename('others').div((~bool_selected).sum()),
                                    ), axis=1).reindex([0,1]).replace(np.nan,0).loc[[1],:].rename(index={1:c})

                except Exception as e:
                    print(e)
                    print(c)
                    raise e
                
                all_summarized_cols.append(tmp)
        else:
            # Return means.
            if normalize:
                if isinstance(normalize,str) and normalize=='strong':
                    quant_range = (20,80)
                else:
                    quant_range = (1,99)
                # Robust scaling of the full dataset first, then means are
                # calculated independently.
                tmp = sklearn.preprocessing.robust_scale(X.loc[:,tmp_cols_set],
                                                         axis=0,
                                                         quantile_range=quant_range)
                tmp = pd.DataFrame(tmp, columns=tmp_cols_set)

                tmp = pd.concat((tmp.loc[bool_selected, tmp_cols_set].mean().rename('selected'),
                                 tmp.loc[~bool_selected, tmp_cols_set].mean().rename('others')
                                ),axis=1).replace(np.nan,0)
            else:
                tmp = pd.concat((X.loc[bool_selected, tmp_cols_set].mean().rename('selected'),
                                 X.loc[~bool_selected, tmp_cols_set].mean().rename('others')
                                ),axis=1).replace(np.nan,0)
            
            all_summarized_cols.append(tmp)
        
    all_summarized_cols = pd.concat(all_summarized_cols)
    
    # Sorting the dataframe index with the original column order.
    all_summarized_cols = all_summarized_cols.loc[summarizing_columns,:]
    return all_summarized_cols   


def relative_change(a, b, method=None):
    """ Returns the relative difference of a against b, 
    
    References:
    * https://en.wikipedia.org/wiki/Relative_change_and_difference
    * https://stats.stackexchange.com/questions/86708/how-to-calculate-relative-error-when-the-true-value-is-zero
    """
    map_method = {
                    'ref':b,
                    'mean':np.mean((a,b)),
                    'abs_mean':np.mean((abs(a),abs(b))),
                    'abs_ref':abs(b),
                    'abs_alt':abs(a),
                    'min':min(a,b),
                    'max':max(a,b),
                    'abs_exp':b
    }

   
    if not method:
        method = 'ref'
        
    try:
        denominator = map_method[method]
    except KeyError as e:
        print("\tMethod '{}' not available ; chose among: {}".format(method, map_method.keys()))
        raise e

    if method=='abs_exp':
        # Taken from:
        # https://math.stackexchange.com/questions/500723/how-to-compute-the-relative-difference-between-two-numbers
        r = 100
        return ((a - b) / denominator) * \
                (1 - np.exp(- np.abs(a - b)/r))
    else:
        return (a - b) / denominator


def add_labels_pvals(pvals_df, column_pval):
    """ Add a 'pval_label' column of '*' basing on the <column_pval> values.
    
    pvals_df can be obtained using the following command:
        ```
        # Apply a multi-test operation to all features in the dataframe, separating samples according to the true class.
        pvals_truey_array = ML_tests.apply_df_multi_tests(X=X.iloc[:,start_annots:],
                                                   bool_selected = tmp_bool_selected,
                                                   cols_type = cols_type, # The test is determined by the col type ; discrete / continuous = Mannwithney U-test, binary = Chi2
                                                   adjust=True)

        pvals_truey_df = pvals_truey_array.reset_index().set_axis(['feature','corrected_pval'],axis=1,inplace=False)
        ```
        
    """
    pvals_df['pval_label'] = ['*' if pval<5e-2 else '' for pval in pvals_df[column_pval]]
    pvals_df['pval_label'] = ['* *' if pval<1e-2 else label for pval,label in zip(pvals_df[column_pval], pvals_df.pval_label)]
    pvals_df['pval_label'] = ['* * *' if pval<1e-3 else label for pval, label in zip(pvals_df[column_pval], pvals_df.pval_label)]
    return pvals_df


def cohen_phi(prop):
    return 2 * np.arcsin(np.sqrt(prop))


def calculate_effect_size(mean_values_df, features_type, pooled_var):
    """ Get the effect size of col 'selected' vs 'other' for each row feature.

    For each feature (rows in <mean_values_df>), calculate the effect size as
    the "normalized" quantity difference between the "selected" column value,
    and the "other" column value.

    The "normalized quantity difference" is calculated according to the
    type of distribution associated to the feature:
    - if continuous or discrete, then it's a comparison of means divided by the
      pooled variances. 
    - if proportions, then its Cohen's Phi difference : phi=2(arcsin(sqrt(P)))

    """
    assert list(mean_values_df.columns.values) == ['selected','others'], \
            "Columns of <mean_values_df> should be ['selected','others']"

    features = mean_values_df.index.values

    all_effectsize_feats = []

    for type_feat in set(list(features_type.values())):
        tmp_feat_set = [c for c in features if features_type[c] == type_feat]
        
        if not len(tmp_feat_set)>0: continue
            
        if type_feat == 'binary':
            # We compare proportions.
            tmp = mean_values_df.loc[tmp_feat_set].applymap(cohen_phi
                    ).apply(lambda row: row['selected'] - row['others'],axis=1)

            all_effectsize_feats.append(tmp)
            
        else:
            # We compare means.
            tmp = mean_values_df.loc[tmp_feat_set].apply(
                    lambda row: (row['selected'] - row['others'])/ \
                                 np.sqrt(pooled_var[row.name]),
                                 axis=1
                                 )
            
            all_effectsize_feats.append(tmp)
        
    all_effectsize_feats = pd.concat(all_effectsize_feats)
    
    # Sorting the dataframe index with the original column order.
    all_effectsize_feats = all_effectsize_feats.loc[features]
    return all_effectsize_feats


def calculate_pooled_var(X, bool_select):
    """ Pooled variance from two groups defined by the bool vector.
    """
    var_set1 = X.loc[bool_select,:].var().replace(np.nan,0)
    var_set2 = X.loc[~bool_select,:].var().replace(np.nan,0)
    N1 = (bool_select).sum()
    N2 = (~bool_select).sum()
    pooled_var = ((N1-1) * var_set1 + (N2-1) * var_set2 ) / (N1+N2 - 2)
    return pooled_var


def apply_df_multi_tests(X, bool_selected, cols_type, adjust=True):
    """ Apply tests on <bool_selected> for each column in <X>
    
    Apply on each column of X a test for comparison between rows selected by 
    <bool_selected> and other rows.
    
    Tests applied depend on the col_type defined in the <cols_type> dict:
    - continuous: mannwhitney U-test
    - discrete: mannwhitney U-test
    - binary: chi2 contingency
    
    pvals are corrected by bonferroni correction.
    
    In:
        X (pandas.DataFrame): dataframes with columns on which tests are applied
        bool_selected (np.array): bools to select rows from <X> for the tests
        cols_type (dict): map colnames to types (binary,discrete, or continuous)
    
    Out:
        pd.Series: corrected pvals for all columns.
    """
    map_name_test_options = {
        'ttest':(scipy.stats.ttest_ind,{'equal_var':False}),
        'mannwhitneyu':(scipy.stats.mannwhitneyu,{}),
        'chi2_contingency':(scipy.stats.chi2_contingency,{}),
        'fisher_exact':(scipy.stats.fisher_exact,{}),
        'ks_test':(scipy.stats.ks_2samp,{})
    }
    
    all_cols_with_coltypes = list(cols_type.keys())
    
    summarizing_columns = []
    for c in X.columns:
        if c not in all_cols_with_coltypes:
            pass
        else:
            if cols_type[c]!='other':
                summarizing_columns.append(c)


    all_tests = []
    # Continuous columns : t-test (for a sufficiently large number of samples, TCL says that mean of 2 groups ~N)
    
    for test, type_col in [('mannwhitneyu','continuous'),
                           ('mannwhitneyu','discrete'),
                           ('chi2_contingency','binary')]:
                           #('fisher_exact','binary')]:
        test_f, options = map_name_test_options[test]
        cols = [c for c in summarizing_columns if cols_type[c] == type_col]
        
        if not (len(cols) > 0): continue

        if type_col == 'binary':
            pvals = []
            for c in cols:
                tmp = pd.concat((X.loc[bool_selected,c].value_counts().rename('selected'),
                                 X.loc[~bool_selected,c].value_counts().rename('others')),axis=1).replace(np.nan,0)
                try:
                    res_test = test_f(tmp, **options)[1] 
                except ValueError as e:
                    print("Error for column '{}':".format(c))
                    print(e)
                    print("Marked as non-significant")
                    res_test = 1

                pvals.append(res_test)
                
            all_tests.append(pd.Series(pvals, cols))
                
        else:
            pvals = []
            for c in cols:
                try:
                    res_test = test_f(X.loc[bool_selected,c],
                                     X.loc[~bool_selected,c],
                                     **options)[1]
                except ValueError as e:
                    print("Error for column '{}':".format(c))
                    print(e)
                    print("Marked as non-significant")
                    res_test = 1

                pvals.append(res_test)
                
            all_tests.append(pd.Series(pvals, cols))
                
    pvals = pd.concat(all_tests).loc[summarizing_columns]

    if adjust==True:
        p_adjusted = pd.Series(multipletests(pvals, method='holm')[1],index=pvals.index.values)
    else:
        p_adjusted = pvals

    return p_adjusted    


def multiple_relative_changes(df_list, bool_select_list, cols_type, normalize=False, independant_corrections=False,
                               method_change='abs_mean'):
    """ Get the summarized features, relative changes, and pvals for a list of dataframes.
    
    In:
        df_list (list): contains multiple pandas.DataFrames of features annotations for multiple samples
        bool_select_list (list): contains for each dataframe of <df_list>
                                 a boolean array to select samples against the others for summarizing
        cols_type (dict): dictionary mapping features to distribution types
        normalize ([bool, default=False]): robust scaling (1perc ignored) of
                                           the discrete / continuous features
                                           before calculating means.
        independant_corrections [(bool, False)]: wether to perform a Bonferroni correction
                                                 per dataframe or overall (default).

        method_change [(str, 'abs_mean')]: method use to measure the relative change
                                           between selected and others samples.
        
    Out:
        (list, list, list): three lists, each of N_df elements.
                                - First is the list of summarized features (N_features x 2) ;
                                - Second is the list of relative changes (N_features) ;
                                - Third is the list of pvalues (N_features x 4 or 5)
    """

    list_summarized = []
    list_relative_changes = []
    list_pvals_df = []
    
    # Adjustment will be performed over all pvals_df rather than independantly. (N_tests = N_df * N_features)
    for i, (X, bool_select) in enumerate(zip(df_list, bool_select_list)):
        summarized_features = summarize_annotations_per_group(X, bool_select,
                                                              cols_type, normalize)
        
        if independant_corrections:
            name_pval = 'corrected_pval'
            adjust=True
        else:
            name_pval = 'pval'
            adjust=False


        # Calculate the pooled variances for the features.
        # This will be used for the effect size calculation.
        columns_selected = [c for c in X.columns
                            if cols_type.get(c,'other')!='other']

        pooled_var = calculate_pooled_var(X.loc[:,columns_selected], bool_select)

        # The test is determined by the col type ; discrete / continuous = Mannwithney U-test, binary = Chi2
        pvals_array = apply_df_multi_tests(X=X.loc[:,columns_selected],
                                           bool_selected = bool_select,
                                           cols_type = cols_type,
                                           adjust=adjust)
        
        pvals_df = pvals_array.reset_index().set_axis(['feature',name_pval],
                                                      axis=1,
                                                      inplace=False)

        # Get the effect size for each feature, as a measure of distance
        # between 'selected' and 'others' values.
        effect_size_feats = calculate_effect_size(summarized_features, cols_type, pooled_var)

        list_summarized.append(summarized_features)
        #list_relative_changes.append(relative_changes)
        list_relative_changes.append(effect_size_feats)
        list_pvals_df.append(pvals_df)
        
        
    if not independant_corrections:
        merged_pvals_df = pd.concat([d.assign(index_df=i) for i,d in enumerate(list_pvals_df)])
        merged_pvals_df['corrected_pval'] = multipletests(merged_pvals_df.pval, method='holm')[1]

    else:
        merged_pvals_df = pd.concat([d.assign(index_df=i) for i, d in enumerate(list_pvals_df)])
    
    list_pvals_df = []
    for g, d in sorted(merged_pvals_df.groupby('index_df')):
        list_pvals_df.append(d.drop('index_df',axis=1))
        
    # Now for each pval_df, add the labels (a string with stars representing
    # the level of significance).
    [add_labels_pvals(pvals_df, column_pval='corrected_pval') for pvals_df in list_pvals_df]
        
    return (list_summarized, list_relative_changes, list_pvals_df)


# PART1: CLASSIFICATION EVALUATION
# ================================

def named_confusion_matrix(y_target, y_pred, labels=[0,1]):
    confusion_mat = pd.DataFrame(sklearn.metrics.confusion_matrix(y_target,y_pred, labels=labels),
                                columns=["predicted 0","predicted 1"],
                                index=["actual 0","actual 1"])
    return confusion_mat


def export_classifier(clf, X, y, col_set, name, outputdir):
    """ Export a classifier trained on the full {X,y} dataset in outputdir.
    
    If outputdir does not exists, it gets created.
    The name is used to name the .pkl model, and all other datasets.
    
    Multiple elements are exported:
        - The classifier itself (with joblib.dump) /!\ The classifier should be
          fitted!!
        - The training set (last column is the y vector)
        - The set of columns used for training

    Return nothin.
    """
    if not os.path.exists(outputdir): os.makedirs(outputdir)
        
    path_model = outputdir+"/{name}.pkl".format(name=name)
    path_cols = outputdir+"/{name}.pkl_COLUMN_USED.tsv".format(name=name)
    path_dataset = outputdir+"/{name}.pkl_DATASET.tsv.gz".format(name=name)

    joblib.dump(clf,path_model,compress=9)
    pd.Series(col_set).to_csv(path_cols,header=True,index=False,sep="\t")
    X.assign(TARGET=y.values).to_csv(path_dataset,header=True,index=False,sep="\t",compression='gzip')
    
    print("Created:")
    print("- {}\n- {}\n- {}".format(path_model,path_cols,path_dataset))

    return None


Clf_scores = collections.namedtuple("classifier_scores",
                                    ["accuracy", "balanced_accuracy",
                                     "recall", "precision",
                                     "specificity","f1"])

def evaluate_classifier_preds(y_target,y_pred):
    """ Return a namedtuple of different scores on predictions.

    This only works for binary classes, labeled as 0 and 1.
    """
    accuracy = sklearn.metrics.accuracy_score(y_target, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_target, y_pred, adjusted=False)
    recall = sklearn.metrics.recall_score(y_target,y_pred)
    precision = sklearn.metrics.precision_score(y_target,y_pred)
    specificity = sklearn.metrics.recall_score(1-y_target,1-y_pred)
    f1 = sklearn.metrics.f1_score(y_target,y_pred)

    return Clf_scores(accuracy, balanced_accuracy,
                      recall,precision,specificity,f1)



def stratifiedkfold_predict(clf, X, y, features, splitter, get_proba=False,
                            test_imb_ratio=None):
    """ Return a list of (y,y_predict,index) from train/test splits.
    
    In:
        clf: the classifier to use (should have a predict_proba method)
        X (numpy.ndarray): (N,F) matrix of samples
        y (numpy.ndarray): (N,) array of target classes
        features (list): array of feature names
        splitter: object to use for splitting the dataset into train/test sets
        get_proba (bool, default=False): whether to get the probabilities
                                         or the class.
    
    Out:
        list: contains a tuple for each fold, with:
              * y as first element
              * y_predict as second
              * test sample indices as third
              * train sample indices
              * the classifier trained on the kfold
              * (NOT IMPLEMENTED)oob_score if requested, None otherwise.
    """
    kfold_predict = []

    if isinstance(X, pd.DataFrame):
        X = X.loc[:,features]
        pass
        # Do not reset the index as the test_index is reported.
        # Any index issue should be dealt with before the function call.
        #X = X.reset_index(drop=True)

    elif isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=features)

    else:
        print(("Error: unrecognized type for X (should be pandas DataFrame "
               "or numpy.ndarray)"))
        return None

    for i,(train_index, test_index) in enumerate(splitter.split(X,y)):
        X_train, X_test = X.loc[train_index,:], X.loc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]

        if test_imb_ratio is not None:
            rus = imblearn.under_sampling.RandomUnderSampler(
                        sampling_strategy=test_imb_ratio)
            _ = rus.fit_resample(X_test.replace(np.nan,0), y_test)
            test_index = test_index[rus.sample_indices_]
            X_test, y_test = X_test.loc[test_index,:], y_test[test_index]

        train_imb = y_train[y_train==1].sum() / y_train.size
        test_imb = y_test[y_test==1].sum() / y_test.size
        print(("{} - split {} : "
               "train size: {:,} ((+)={:.3}%) ; "
               "test size: {:,} ((+)={:.3}%)"
              ).format(datetime.datetime.now().replace(microsecond=0),
                       i+1,
                       len(train_index),
                       train_imb*100,
                       len(test_index),
                       test_imb*100
              ))
        clf.fit(X_train, y_train)

        print("\t{} - Test evaluation.".format(datetime.datetime.now().replace(microsecond=0)))
        if get_proba:
            y_pred = clf.predict_proba(X_test)

        else:
            y_pred = clf.predict(X_test)
            
        oob_score = None
        try:
            if clf.oob_score:
                oob_score = clf.oob_score_
                print("\tOut-of-bag accuracy estimate: {}".format(clf.oob_score_))
        except AttributeError:
            pass

        kfold_predict.append((y_test,y_pred,test_index,train_index,deepcopy(clf)))
        print("\n")
        
    return kfold_predict



def merge_kfold_test_samples(X, kf_results, threshold=0.5, assignment_method='threshold'):
    """ Merge test samples from k-folds indexes, along with prediction scores and predicted class.
    
    In:
        X (pandas.DataFrame): 
        kf_results (list): tuples of k-fold results, with (true_y, pred_scores, indexes, trained_model)
        threshold ([float,default=0.5]): threshold to use to assign samples to classes.
        assignment_method ([str, default='threshold']): choices are ['threshold','max'] ; 
                                                        assigned class for a sample is done either
                                                        using a threshold on the 'class 1' score 
                                                        (binary classification), or assigning to the
                                                        class with max_score.
                                                        
    Return:
        pandas.DataFrame: of size (N_sample x  (N_features + 3 + N_classes)) ;
                          the 3 columns are ['true_class','kfold','predicted_class'] ;
                          last columns are named 'score_{class}''
                          
    """
    assert assignment_method in ['max','threshold'], print("Unknown assignment_method")

    test_df_kfolds = []

    for i, kf_res in enumerate(kf_results):
        y_kf, preds_kf, index_test_kf, clf_kf = kf_res

        # Table of predicted score for each class (N_columns = 2 if binary classification)
        cols = ['score_{}'.format(j) for j in range(len(preds_kf[0]))]
        tb_preds = pd.DataFrame(preds_kf).set_axis(cols, axis=1, inplace=False)
        tb_preds.index = index_test_kf

        if assignment_method=='threshold':
            predicted_class = [int((row>threshold).idxmax().strip('score_'))
                               for _, row in tb_preds.iterrows()]

        elif assignment_method=='max':
            predicted_class = [int(row.idxmax().strip('score_'))
                               for _, row in tb_preds.iterrows()]

        tmp = X.loc[index_test_kf,:].assign(true_class=y_kf,
                                             kfold='kfold_{}'.format(i),
                                             predicted_class=predicted_class
                                             )

        tmp = pd.concat([tmp, tb_preds],axis=1)

        test_df_kfolds.append(tmp)
        print("kfold {}".format(i))

    test_df_kfolds = pd.concat(test_df_kfolds)

    return test_df_kfolds.sort_index()


def label_majority_class_assignment(annotated_X, labels):
    """ Assign to each sample the most represented PREDICTED class in its label group.
    
    In:
        annotated_X (pandas.DataFrame): Should contain the column 'predicted_class'
        labels (array): one per sample ; the samples will be grouped according to this
        
    Out:
        array of corrected classes (one per sample), based on the most represented "True_class"
        in each label.
    """
    corrected_classes = []
    for label, label_df in annotated_X.assign(label=labels).groupby('label'):
        majority_class = label_df.predicted_class.value_counts().idxmax()
        
        corrected_classes.append(label_df.assign(class_label=majority_class))
        
    corrected_classes = pd.concat(corrected_classes).loc[:,'class_label']
    return corrected_classes.loc[annotated_X.index].values



# PART2: FOREST AND TREES PROPERTIES
# ==================================

TreeStructure = collections.namedtuple('TreeStruct',('n_nodes',
                                                     'children_left',
                                                     'children_right',
                                                     'node_is_right_child',
                                                     'node_depth',
                                                     'node_n_samples',
                                                     'is_leaf_node',
                                                     'feature',
                                                     'threshold',
                                                     'node_class_props'
                                                    ))


def get_tree_structure(tree):
    """ From a decision tree estimator, get different information on its structure into a namedtuple.
    In:
        tree (sklearn.tree.tree.DecisionTreeClassifier)
    
    Out:
        TreeStructure (namedtuple): formatted information on the structure of the tree.
    """
    
    n_nodes = tree.tree_.node_count
    node_n_samples = tree.tree_.n_node_samples
    # Vectors with a value for each of the n_nodes.
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold

    # These will store for each node the information on the depth, and wether it's a leaf or not.
    node_depth = np.zeros(shape=n_nodes)
    node_is_right_child = np.zeros(shape=n_nodes)
    is_leaf = np.zeros(shape=n_nodes, dtype=bool)

    # Start at the root node.
    # node_id, depth, is_right_child
    stack = [(0, -1, -1)]

    # For each node, store information on its left and right children: get the depth, and whether it's a leaf or not.
    # Cases where the node is a leaf are identifiable by 
    while len(stack) > 0:
        node_id, parent_depth, is_right_child = stack.pop()
        current_node_depth = parent_depth + 1
        node_depth[node_id] = current_node_depth
        node_is_right_child[node_id] = is_right_child

        # If we have a test node: add each children to the stack, with current node depth.
        # Else: mark the node as a leaf in is_leaf
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], current_node_depth, 0))
            stack.append((children_right[node_id], current_node_depth, 1))
        else:
            # No children found => leaf.
            is_leaf[node_id] = True

    class_props_per_node = np.array([(v/v.sum())[0] for v in tree.tree_.value])
    
    return TreeStructure(n_nodes,
                         children_left,
                         children_right,
                         node_is_right_child.astype(int),
                         node_depth.astype(int),
                         node_n_samples,
                         is_leaf,
                         feature,
                         threshold,
                         class_props_per_node)


def print_tree_structure(tree_structure):
    print("The binary tree structure has {} nodes and has "
          "the following tree structure:".format(tree_structure.n_nodes))
    for i in range(tree_structure.n_nodes):
        if tree_structure.is_leaf_node[i]:
            print("{}node={} leaf node.".format(tree_structure.node_depth[i] * "\t", i))
        else:
            print("{}node={} test node: go to node {} if X[:, {}] <= {:.2f} else to "
                  "node {}.".format(tree_structure.node_depth[i] * "\t",
                                     i,
                                     tree_structure.children_left[i],
                                     tree_structure.feature[i],
                                     tree_structure.threshold[i],
                                     tree_structure.children_right[i],
                                     ))
    print()
    

def draw_tree(decision_tree,feature_names=None,class_names=None):
    """ Uses the graphviz exporter and pydotplus module to plot a tree.
    
    The coloring is basing on the purity of a node for one class or the other.
    """
    dot_data = sklearn.tree.export_graphviz(decision_tree,
                                            out_file=None,
                                            proportion=False,
                                            node_ids=True,
                                            filled=True,
                                            rounded=True,
                                            special_characters=True,
                                            feature_names=feature_names,
                                            class_names=class_names)
    graph = pydotplus.graph_from_dot_data(dot_data)
    return graph


# NOTE: the rfpimp is working on the test dataset here, so this is not
# comparable to the feature importances learnt from the TRAIN dataset...
def feature_importances(kf_results, X, metric=None):
    """ Return both the raw and corrected feature importances from a list of k-fold results.
    
    k-fold results are of the form (y_true, y_preds, index_test, clf)
    
    For each k-fold in the list, calculate both the raw feature importance (stored in the model),
    and the corrected feature importances, based on permutations.
    
    In:
        kf_results (list)
        
    Return:
        pandas.DataFrame of long format, with columns:
            ['feature','kfold','value','importance_type']
    """
    corrected_feature_importances = []
    raw_feature_importances = []

    for i, kf_res in enumerate(kf_results):
        print("Feature importance from k-fold {}".format(i+1))
        y_true, y_preds, index_test, kf_clf = kf_res

        if isinstance(kf_clf, sklearn.pipeline.Pipeline):
            # apply the operations of the pipeline, learnt on the TRAIN
            # dataset.
            X_tmp = sklearn.pipeline.Pipeline(kf_clf.steps[:-1]).fit_transform(X) 
            kf_clf_tmp = kf_clf.steps[-1][1]

        else:
            X_tmp = X
            kf_clf_tmp = kf_clf

        tmp = rfpimp.importances(kf_clf_tmp, X_tmp.loc[index_test, :], 
                                 y_true,
                                 n_samples=-1,
                                 metric=metric)
        corrected_feature_importances.append(tmp['Importance'].rename("kfold_{}".format(i)))
        raw_feature_importances.append(pd.Series(kf_clf_tmp.feature_importances_,
                                                 index=X.columns,
                                                 name="kfold_{}".format(i)))

    corrected_feature_importances = pd.concat(corrected_feature_importances,axis=1, sort=False).sort_index()
    raw_feature_importances = pd.concat(raw_feature_importances,axis=1).sort_index()
    
    full_long_df = pd.concat([corrected_feature_importances.reset_index().melt(id_vars=['index']).assign(importance_type='corrected'),
                              raw_feature_importances.reset_index().melt(id_vars=['index']).assign(importance_type='raw')
                             ],
                             axis=0).set_axis(['feature','kfold','value','importance_type'],axis=1,inplace=False)
    
    return full_long_df


