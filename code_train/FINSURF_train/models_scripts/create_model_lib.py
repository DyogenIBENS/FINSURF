#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
"""

import os
import sys
import glob

import pandas as pd
import numpy as np
from sklearn.externals import joblib
from copy import deepcopy

import sklearn.model_selection
import sklearn_pandas

import imblearn

import datetime


def build_mapper_features_preprocessing(cat_features, X, reduced_num_cols, cols_to_drop = []):
    """ Create a preprocessing object from hard-coded lists of features.
    
    Here are hardcoded lists of features that are grouped according to the method for NaN replacement.
    
    For instance a "median_feats" list groups all feature names from my annotations tables that should see their NaN values being replaced
    by the median value from a subset of rows.
    
    These list are build from the <cat_features> and <X> variables.
    
    Args:
        cat_features (dict) : dict that maps lists of features to categories, such as "CONSERVATION"
        X (pandas.DataFrame) : table of numeric features of annotated variants
        reduced_num_cols (list) : subset of the X columns to be actually used for classification. Only features in this list will be kept.
        cols_to_drop (list) : used to filter out features.

    Returns: 
        sklearn_pandas.DataFrameMapper

    """
    list_features_preprocessing = []

    # "Default to median" features.
    median_feats = [feat for feat in cat_features['CONSERVATION']
                    if ((feat in reduced_num_cols) and (feat!='gerpElem'))
                   ] + \
                   ['H3K4me1_medFC','H3K4me3_medFC','H3K27ac_medFC'] + \
                   ['vartrans.ord']
    
    # Remove unwanted features.
    median_feats = set([feat for feat in median_feats
                        if ((feat in reduced_num_cols) & (feat not in cols_to_drop))
                       ])

    median_feats_strat = sklearn_pandas.gen_features(columns = [[f] for f in median_feats],
                                                     classes = [{'class':sklearn.impute.SimpleImputer,
                                                                 'strategy':'median'}]
                                                     )

    # "Default to 0" features.
    zeros_feats = set(['gerpElem','CpGisland','dnaseClust','CGdinit',
                       'ratio_shared_targets','targets_associations'] + \
                      list(X.head().filter(regex='tfbs.').columns.values) + \
                      list(cat_features['ENHANCERS']) +\
                      list(X.head().filter(regex='^roadmap.').columns.values)
                     )

    zeros_feats = [feat for feat in zeros_feats
                   if ((feat in reduced_num_cols) & (feat not in cols_to_drop))
                  ]
        
    zeros_feats_strat = sklearn_pandas.gen_features(columns = [[f] for f in zeros_feats],
                                                    classes = [{'class':sklearn.impute.SimpleImputer,
                                                                'strategy':'constant',
                                                                'fill_value':0}]
                                                   )

    # Reorder the list of feature preprocessing according to the original order.
    list_features_preprocessing = sorted(median_feats_strat+zeros_feats_strat,
                                         key=lambda v:
                                         reduced_num_cols.index(v[0][0]))

    print("Columns without assigned Impute strategy (will be lost if no strategy set):")
    display([c for c in reduced_num_cols if c not in [v[0][0] for v in list_features_preprocessing]])
            
    mapper = sklearn_pandas.DataFrameMapper(list_features_preprocessing, #default=None,
                                             df_out=True)
    
    return mapper


def export_kfolds_results(kfold_results, path):
    """ Export the elements from kfold results, to the indicated <path>.
    
    Expected elements per unit in the <kfold_results> list are:
    - y_test : true label of the test samples
    - y_pred : predicted label (or proba) of the test samples
    - test_index : indices of the test samples
    - train_index : indices of the train samples
    - model : trained model for the k-fold.
    
    For each element, a new directory named "kfold_{index}" is created.
    All indices and label variables are exported as tables (one per variable).
    The model is exported with joblib.dump.
    """
    created_files = []
    
    for i, kf_res in enumerate(kfold_results):
        path_kf = path + '/kfold_{:02d}/'.format(i)
        if os.path.isdir(path_kf):
            raise ValueError(('Path "{}" already exists ; please empty the main '
                              'directory before exporting.').format(path_kf))
        else:
            os.makedirs(path_kf)
        
        y_test, y_pred, test_index, train_index, clf = kf_res
        
        for var, name in [(y_test,'y_test.txt'),
                          (y_pred,'y_pred.txt'),
                          (test_index,'test_index.txt'),
                          (train_index,'train_index.txt')]:

            path_file = path_kf+'/'+name
            pd.DataFrame(var).to_csv(path_file,header=False,index=False,sep="\t")
            created_files.append(path_file)

        # And export the associated model.
        joblib.dump(clf, path_kf+'/model.pkl', compress=9)
        created_files.append(path_kf+'/model.pkl')
        

    for f in created_files:
        print('Created file "{}"'.format(f))

    return None


def find_best_threshold(y_kfold, y_proba_kfold, scorer):
    """ Find the best threshold on <scorer>, from labels and predictions.
    
    CAUTION : binary classification only...
    
    Thresholds are explored from a score from 0 to 1, with step of 0.1
    
    The best threshold is determined by scoring the y_true against y_pred
    (converted to binary class with the threshold).

    In:
        y_kfold (list) : list of arrays of true y label
        y_proba_kfold (list) : list of 2d arrays of predicted
                               probabilities for each of y labels
        scorer (function) : scorer on which the best threshold will be
                            evaluated through a range(0,1,0.1) iteration.
                            This function should be chosen from the sklearn methods.
    
    Return :
        mean threshold, calculated from the thresholds from the k-folds.
    """
    range_thresh = np.arange(0,1,0.1)
    best_thresh_kfold = []
    for y, y_prob in zip(y_kfold, y_proba_kfold):
        scores_kfold = [scorer(y, [yp[1]>thresh for yp in y_prob])
                         for thresh in range_thresh
                       ]
        best_thresh_kfold.append(range_thresh[np.argmax(scores_kfold)])
        
    return np.mean(best_thresh_kfold)



def categorical_stratified_kfold_predict(model, X, y, features, use_cols,
                                         category,
                                         K,
                                         get_proba=True,
                                         test_imb_ratio=None):
    """ Train-test split evaluation of <model> on <X>, splitting on the column <category>.

    Rather than relying on the classical `model_selection.StratifiedKFold`, or
    similar methods, we want here to split the samples into <K> groups based on
    the values present in the <category> colummn from the <X> table.

    So first are identified the unique categories, on which the samples are
    grouped. The classes from <y> are used to assess the splits over the
    categories do not yield a grouped devoid of one class (CAUTION : BINARY
    CLASSIFICATION ONLY).

    A stratified k-fold is applied on the unique categories, and the samples
    from <X> are retrieved for the train and test sets.

    The <test_imb_ratio> may be used to reduce the imbalance in the test set,
    and thus get a better view of the quality of the predictor.

    In :
        model (random forest classifier)
        X (pd.DataFrame)
        y (pd.Series)
        features (list): all column names of the <X> dataframe.
        use_cols (list): columns to use for the training.
        category (string)
        K (int)
        get_proba (bool, default=True)
        test_imb_ratio (float, default=None)


    Out:
        list of k-fold results, each composed of:
        - y_test : true labels for the test set
        - y_pred : predictions for the test set
        - test_index : indices of samples for the test set
        - train_index : indices of samples for the train set
        - model : sklearn model obtained for the fold.

    Caution : if <test_imb_ratio> is provided, the total number of samples
    from <test_index> and <train_index> may not be equal to the full size
    of the <X> dataframe, because one of the sample classes will have been
    reduced.

    """

    kfold_predict = []

    # Check X datatype
    if isinstance(X, pd.DataFrame):
        pass
    elif isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=features)
    else:
        print("Unrecognized structure for X")
        return 

    # Check y datatype
    if isinstance(y, pd.Series):
        pass
    elif isinstance(X, np.ndarray):
        y = pd.Series(y, name='y')
    else:
        raise ValueError("Unrecognized structure for y (should be Series or np.ndarray)")

    # Check that required category is among X columns.
    if not category in features:
        raise ValueError('Category "{}" not found among columns.'.format(category))


    # This splitter will be used to separate samples on the categories.
    splitter = sklearn.model_selection.StratifiedKFold(K,shuffle=True)

    # The split will be operated on this table: the X table is reduced to the
    # unique categories represented in it, along the class in y.
    tmpX_categories = (pd.concat([X[category], y],
                                 axis=1
                                 ).set_axis([category,'y'],
                                            axis=1,
                                            inplace=False
                                 ).groupby(category)['y'].sum()>0
                      ).astype(int).reset_index()

    tmpy = tmpX_categories['y']

    for i, (train_idx, test_idx) in enumerate(splitter.split(tmpX_categories, tmpy)):
        # First part : split according to the categories representation in X.
        category_train = tmpX_categories.loc[train_idx,category].values
        category_test = tmpX_categories.loc[test_idx,category].values

        full_X_indices_train = X.loc[X[category].isin(category_train),:].index.values
        full_X_indices_test = X.loc[X[category].isin(category_test),:].index.values

        X_train, y_train = X.loc[full_X_indices_train,:], y[full_X_indices_train]
        X_test, y_test = X.loc[full_X_indices_test,:], y[full_X_indices_test]


        # Second part : if test_imb_ratio is provided, resample the X_test to
        # match the y class representation to the required test_imb_ratio.
        if test_imb_ratio is not None:
            rus = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=test_imb_ratio)
            _ = rus.fit_resample(X_test.replace(np.nan,0), y_test)

            full_X_indices_test = full_X_indices_test[rus.sample_indices_]
            X_test, y_test = X.loc[full_X_indices_test,:], y[full_X_indices_test]


        train_imb = y_train[y_train==1].sum() / y_train.size
        test_imb = y_test[y_test==1].sum() / y_test.size

        print(("{} - split {} : "
               "train size : {:,} ((+)={:.3}%) | "
               "test size : {:,} ((+)={:.3}%)"
              ).format(datetime.datetime.now().replace(microsecond=0),
                       i+1,
                       len(full_X_indices_train),
                       train_imb*100,
                       len(full_X_indices_test),
                       test_imb*100)
              )


        model.fit(X_train.loc[:,use_cols], y_train)

        print("\t{} - Test evaluation.".format(datetime.datetime.now().replace(microsecond=0)))

        if get_proba:
            y_pred = model.predict_proba(X_test.loc[:,use_cols])
        else:
            y_pred = model.predict(X_test.loc[:,use_cols])

        oob_score = None
        try:
            if model.oob_score:
                oob_score = model.oob_score_
                print("\tOut-of-bar accuracy estimate: {}".format(model.oob_score_))

        except AttributeError:
            pass

        kfold_predict.append((y_test,
                              y_pred,
                              full_X_indices_test,
                              full_X_indices_train,
                              deepcopy(model)))

    return kfold_predict




class KfoldResults(object):
    def __init__(self, name, path):
        self.name = name
        self.path = path
        print('Loading all kfolds results from "{}"'.format(self.path))
        self.load()


    def load(self):
        """ Load all elements from path.
        """
        self.loaded=True

        self.test_y = []
        self.test_pred = []
        self.test_indices = []
        self.train_indices = []
        self.models = []

        for kfold_path in glob.glob(self.path+'/kfold_*'):
            self.test_y.append(pd.read_table(kfold_path+'/y_test.txt',
                                             header=None,
                                             sep="\t"
                                             ).iloc[:,0].values)

            self.test_pred.append(pd.read_table(kfold_path+'/y_pred.txt',
                                             header=None,
                                             sep="\t"
                                             ).values)

            self.test_indices.append(pd.read_table(kfold_path+'/test_index.txt',
                                             header=None,
                                             sep="\t"
                                             ).iloc[:,0].values)

            self.train_indices.append(pd.read_table(kfold_path+'/train_index.txt',
                                             header=None,
                                             sep="\t"
                                             ).iloc[:,0].values)

            self.models.append(joblib.load(kfold_path+'/model.pkl'))


        return 
            



    def get_kfold_res_structure(self):
        """ Return the different self variables into the list structure.

        The list structure is sometimes expected by different functions.
        So in order to use them, you can rely on this function to return
        the variables in the correct format, being a list of tuple, each
        with:
        - test_y
        - test_pred
        - test_indices
        - train_indices
        - model
        """
        return zip(self.test_y,
                   self.test_pred,
                   self.test_indices,
                   self.train_indices,
                   self.models)

