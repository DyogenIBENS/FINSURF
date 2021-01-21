#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
"""


###############################################################################
# IMPORTS

import sys
import os

import pandas as pd
import numpy as np
import sklearn

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import functools
import multiprocessing
import datetime

from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, "../utils/")
    
from ML_tests import label_majority_class_assignment



###############################################################################
# DEFINITIONS


# Feature contributions
# ---------------------


def create_feature_contributions_table(predictions, biases, contributions,
                                      features, sample_index):
    """ Associate feature contributions to samples in X.
    
    For now, only tested for independant contributions, with a classifier random forest.
    
    In:
        
    out:
        pandas.DataFrame
    """
    if not isinstance(features, pd.Series):
        features = pd.Series(features)
    
    # NOW: create multi index tables.
    # -------------------------------
    classes = [k for k in range(len(predictions[0]))]
    tb_preds_tmp = pd.DataFrame(predictions).set_axis(classes, axis=1, inplace=False)
    
    # Transform this table into a multi index one.
    rename_cols = {'index':'sample',
                 'variable':'class',
                 'value':'pred_score'}
    
    tb_preds_tmp = tb_preds_tmp.set_index(sample_index
                                         ).reset_index(
                                ).melt(id_vars='index'
                                      ).rename(columns=rename_cols
                                               ).set_index(['sample','class'])

    tb_bias_tmp = pd.DataFrame(biases
                        ).set_axis(classes,
                                   axis=1,
                                   inplace=False)

    # Transform this table into a multi index one.
    rename_cols = {'index':'sample',
                 'variable':'class',
                 'value':'bias'}

    tb_bias_tmp = tb_bias_tmp.set_index(sample_index
                                       ).reset_index(
                                    ).melt(id_vars='index'
                                          ).rename(columns=rename_cols
                                                   ).set_index(['sample','class'])

    # Now for the contributions dataframe.
    contribs_df = []
    for index_sample, ctb in zip(sample_index, contributions):
        ctb_df = pd.DataFrame(ctb).set_axis(classes,axis=1,inplace=False)
        ctb_df = ctb_df.set_index(features).T.assign(sample=index_sample
                                                    ).reset_index().rename(
                                                        columns={'index':'class'}
                                                        ).set_index(['sample','class'])
        
        contribs_df.append(ctb_df)
        
    tb_contribs_tmp = pd.concat(contribs_df)


    # Merge them altogether
    contribs_df = pd.concat([tb_contribs_tmp, tb_bias_tmp, tb_preds_tmp],axis=1)

    return contribs_df


def create_merged_featcontribs_kfolds(kf_results, X, ncores=1):
    """ Merge feature contributions from k-fold test samples.
    
    For each k-fold in the kf_results structure, apply the trained model on
    the test samples. Contributions are thus specific to a model and a k-fold.
    
    Contributions toward all classes are kept, and the different k-fold calculated
    tables are aggregated into a single dataframe.
    
    Each sample will thus appear N_classes times, with for each class a row of
    feature contributions toward that class, along with the final prediction score,
    the bias, and the k_fold it belongs to.
    
    In:
        kf_results (list)
        X (pandas.DataFrame): dataframe of samples with features as column, from which
                              train and test samples will be retrieved basing on the index.
        ncores (int, default=1)
        
    Return:
        pandas.DataFrame with columns ['sample', 'class'] + ['features']*N_features + 
                                      ['bias', 'pred_score', 'kfold']
        The number of rows is N_sample * N_classes
    """
    contribs_kfolds = []
    for i, kf_res in enumerate(kf_results):
        print("kfold {}".format(i))
        y_kf, preds_kf, index_test_kf, clf_kf = kf_res

        contribs_df= create_feature_contributions_table(clf_kf, X.loc[index_test_kf,:], ncores=ncores)

        contribs_kfolds.append(contribs_df.assign(kfold='kfold_{}'.format(i)))
        
    # Now we can merge them into a single dataframe.
    
    merged_contribs_kfolds = pd.concat(contribs_kfolds).reset_index()
    return merged_contribs_kfolds


# CLUSTERING
# ----------


# There can be either k-means clustering, or prototype definition.
# Prototypes are identified from the percentage of shared leaves between two
# samples across all trees.


def get_perc_shared_leaves(rdf, df_samples):
    """ Return the list of maximum proportion of samples sharing the same leaf for each tree.
    For each tree in a forest, get the maximum proportion of samples falling in the same leaf.
    This is calculated by getting the index of the leaf in which each sample ends.
    Then these indexes are counted, and the proportion from the one with most samples is reported.
    
    In:
        rdf (sklearn.ensemble.RandomForestClassifier): a fitted random forest classifier.
        df_samples (pandas.DataFrames): samples to pass through the random forest.
    
    Out:
        pandas.Series: values for each tree of the highest proportions of sample in the same leaf.
    """
    max_count_in_leaf_per_tree = [pd.Series(t.apply(df_samples)).value_counts().max()
                                  for t in rdf.estimators_]
    max_prop_in_leaf_per_tree = np.array([v / df_samples.shape[0]
                                          for v in max_count_in_leaf_per_tree])
    return max_prop_in_leaf_per_tree

# pd.Series(get_perc_shared_leaves(clf2.steps[1][1],transformed_train_X)).plot(kind='box')
# plt.show()





def clustering_analysis(X_kmeans, n_jobs, max_K, step, N_repeat_silhouette):
    """ Perform a K-means clustering on the X_kmeans dataframe.
    
    For a range of 2:max_K:step centroids, perform K-means clustering.
    Quality is assessed both by the reduction in the inertia, as well as the silhouette.
    
    Silhouette is evaluated by repeating the silhouette calculus N_repeat_silhouette times,
    on a subsample of the dataset.
    
    
    In:
        X_kmeans (pandas.DataFrame)
    
    Return:
        tuple(list clusters, inertia list, df_silhouette, list of labels): 
            - list clusters is the same shape as the number of evaluated K.
            - inertia list is the same shape as the number of evaluated K.
            - df_silhouette: dataframe of shape N_evaluated_K elements x <N_repeat_silhouette>
            - list of labels contains N_evaluated_K elements, each of shape X_kmeans.n_rows
        
    """
    kmeans_k = range(2, max_K,step)

    if X_kmeans.shape[0]>10000:
        perc = 0.2
    else:
        perc = 1
        
    sample_size_silh = int(X_kmeans.shape[0]*perc)
    
    partial_silhouette_score = functools.partial(sklearn.metrics.silhouette_score,
                                                 sample_size = int(X_kmeans.shape[0]*perc),
                                                )

    silhouettes = []
    inertia = []
    labels = []

    pool = multiprocessing.Pool(n_jobs)

    for k in kmeans_k:
        print("K={}".format(k))
        print("\tPerforming K-means: {}".format(datetime.datetime.now()))
        #kmeans = sklearn.cluster.MiniBatchKMeans(
        kmeans = sklearn.cluster.KMeans(
                                        n_clusters=k,
                                        random_state=10,
                                        n_init=5,
                                        init='k-means++',
                                        n_jobs=n_jobs,
                                        copy_x=True,
                                        algorithm='full',
                                        precompute_distances=False,
                                        #batch_size=5000
                                       )
        cluster_labels = kmeans.fit_predict(X_kmeans)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        print("\tPerforming silhouette: {}".format(datetime.datetime.now()))

        # We are performing 10k silhouette calculus to account for the sampling.


        silhouette_avg = pool.starmap(partial_silhouette_score,
                                      [(X_kmeans, cluster_labels)
                                       for i in range(N_repeat_silhouette)])

        inertia.append(kmeans.inertia_)
        silhouettes.append(silhouette_avg)
        labels.append(cluster_labels)

    pool.close()
    pool.join()
    
    #format the list of list of silhouettes into a dataframe.
    df_silhouettes = pd.DataFrame(silhouettes, index=kmeans_k
                                    ).set_axis(['silhouette' for i in range(N_repeat_silhouette)],
                                               axis=1, inplace=False
                                              )
    
    return (list(kmeans_k), inertia, df_silhouettes, labels)


def select_labels_cluster_analysis(selected_K, list_labels, annotated_X):
    """ Return list of selected labels per sample, along quality measures per label.
    
    In:
        selected_K (int): number of requested clusters
        list_labels (list): full list of all labels per K ; obtained from `clustering_analysis`
        annotated_X (pandas.DataFrame): dataframe with features AND metadata such as 'true_class'
                                        column
                                        
    Return:
        tuple with :
        - selected_labels (list of shape N_sample)
        - set_selected_labels (ordered list of shape <selected_K>)
        - list_bool_selected_clusters (pandas.DataFrame) of shape N_samples x N_labels
        - res_acc (pandas.DataFrame): N_labels x 4, corresponding to 4 accuracy metrics.
    
    """
    # Get label assignments for each sample, and transform to string for further manipulation.
    selected_labels = list_labels[[len(set(l)) for l in  list_labels].index(selected_K)]

    set_selected_labels = [str(c) for c in sorted(set(selected_labels))]

    selected_labels = [str(c) for c in selected_labels]

    # Create list of boolean arrays to select for each cluster samples which belong to it from others.
    list_bool_selected_clusters = [bool_selected.values for c,bool_selected in pd.get_dummies(selected_labels
                                                                                             ).loc[:,set_selected_labels].astype(bool).iteritems()]
    
    res_acc = []

    # # Before anything: get the maximum count of TP/TN/FP/FN over all clusters
    # n_max = merged_kfolds_df.assign(cluster=selected_labels).groupby('cluster').size().max()


    for cluster, tmp in annotated_X.assign(cluster=selected_labels).groupby('cluster'):
        tmp_1 = tmp.loc[tmp.true_class==1,:]
        tmp_0 = tmp.loc[tmp.true_class==0,:]

        acc_all = sklearn.metrics.accuracy_score(tmp.true_class,
                                                 tmp.predicted_class)

        bal_acc_all = balanced_accuracy_score(tmp.true_class,
                                                      tmp.predicted_class,
                                                       adjusted=False)

        acc_0 = sklearn.metrics.accuracy_score(tmp_0.true_class,
                                               tmp_0.predicted_class)

        acc_1 = sklearn.metrics.accuracy_score(tmp_1.true_class,
                                                 tmp_1.predicted_class)

        res_acc.append((cluster,acc_all,bal_acc_all, acc_0,acc_1))

    res_acc = pd.DataFrame(res_acc, columns=['cluster','acc_all', 'bal_acc_all','acc_0','acc_1'])
    
    
    return (selected_labels, set_selected_labels,
            list_bool_selected_clusters, res_acc)


def clustering_analysis_measures(annotated_X, kmeans_k, inertia, df_silhouettes, labels):
    """
    In:
        annotated_X (pandas.DataFrame): should contain the columns:
                                        - true_class
                                        - predicted_class
                                        
        inertia (list): of size len(labels)
        df_silhouettes (pandas.DataFrame): of shape N_evaluated_K x <N_silhouette_calculous>.
                                           This dataframe will be used as a base, expanded with
                                           other measures.
                                           
        labels (list of list): of shape N_evaluated_K, with each element of shape N_samples
        
    out:
        pandas.Dataframe of shape N_evaluated_K x N_measures. N_measures can vary depending
        on the number of times the silhouette was recalulated.
    """
    
    df_clusters_measures = df_silhouettes.copy().assign(inertia=inertia)
    
    # We also add a column measuring the quality of the clusters in terms of TP/FP/FN/TN
    # Note: do not use balanced accuracy here, as we expect clusters to be homogeneous toward one class.
    # Also with regular accuracy, weighting by the number of samples leads to the overall accuracy,
    # which is thus a constant across all K.
    weighted_mean=False

    mean_accuracies = []
    for k, labs in zip(kmeans_k, labels):
        acc = annotated_X.assign(label=labs).groupby('label').apply(
                lambda g: sklearn.metrics.accuracy_score(g.true_class.values, g.predicted_class.values)
            ).values

        #calculate the weighted mean.
        sizes = annotated_X.assign(label=labs).groupby('label').size().values

        if weighted_mean:
            acc_mean = np.average(acc, weights=sizes)
        else:
            acc_mean = np.average(acc)

        mean_accuracies.append(acc_mean)


    df_clusters_measures['mean_accuracy'] = mean_accuracies


    # Also adding the "cluster class accuracy", corresponding to annotating all samples in a cluster with the
    # majority class, and measuring the mean accuracy over all clusters.
    # Majority class here corresponds to the TRUE CLASS with the maximum number of samples.
    # If weighted_mean is set to TRUE, then the calculated accuracy corresponds to the global accuracy.
    weighted_mean=True

    mean_classCluster_accuracies = []
    for k, labs in zip(kmeans_k, labels):
        # Get the majority classes
        labels_majority_classes = label_majority_class_assignment(annotated_X, labs)
        acc = annotated_X.assign(label=labs, majority_class=labels_majority_classes
                                     ).groupby('label').apply(
                                        lambda g: sklearn.metrics.accuracy_score(g.majority_class.values,
                                                                                 g.true_class.values) # I was using 'predicted_class' here before.
                                       ).values

        #calculate the weighted mean.
        sizes = annotated_X.assign(label=labs).groupby('label').size().values

        if weighted_mean:
            acc_mean = np.average(acc, weights=sizes)
        else:
            acc_mean = np.average(acc)

        mean_classCluster_accuracies.append(acc_mean)



    df_clusters_measures['mean_classCluster_accuracy'] = mean_classCluster_accuracies


    
    return df_clusters_measures


def plot_clustering_measures(df_clusters_measures, savefig_file=None, show_plot=True):
    """ Multiple point plots for a set of defined features expected in <df_clusters_measures>
    
    The set of hardcoded features that are plot are:
        ('silhouette','inertia','mean_accuracy','mean_classCluster_accuracy')
        
    In:
        df_clusters_measures (pandas.DataFrame): 
    
    """
    width = 0.8 * df_clusters_measures.shape[0]
    height = 1.9 * df_clusters_measures.shape[1]
    fig = plt.figure(figsize=(width,height))
    
    measures=('silhouette','inertia','mean_accuracy','mean_classCluster_accuracy')
    N_rows = len(measures)
    
    for i, name_measure in enumerate(measures):
        if name_measure not in df_clusters_measures.columns:
            print('{} not in dataframe column, passed.'.format(name_measure))
            continue
            
        if i==0:
            ax = fig.add_subplot(N_rows,1,i+1)
        else:
            ax = fig.add_subplot(N_rows,1,i+1, sharex=fig.axes[0])

        sns.pointplot(
            data=df_clusters_measures.loc[:,name_measure].reset_index().melt(id_vars='index').loc[:,['index','value']].rename(columns={'value':name_measure}),
            x='index',
            y=name_measure,
            ci=0.1,
            ax=ax)

        ax.set_xlabel('')
        
        if 'accuracy' in name_measure:
            ax.set_ylim(ax.get_ylim()[0], 1)
    
    plt.tight_layout()
    
    if savefig_file is not None:
        plt.savefig(savefig_file)
    
    if show_plot:
        plt.show()
        
    else:
        return fig



def get_centroids(X, labels):
    """ Return centroid samples from labels
    
    The method used for aggregation is mean.
    """
    return X.assign(label=labels).groupby('label').aggregate('mean')

    
def get_distance_to_centroids(X, centroids, distance='euclidean'):
    """ Return the distances to all centroid of the sample.
    
    Given N_samples in X, and N_centroids in centroids (each with N_features columns),
    return a N_samples x N_centroids dataframe with euclidean distances between samples
    and centroids.
    
    In:
    
    Return:
    """
    return pd.DataFrame(sklearn.metrics.pairwise.euclidean_distances(X=X, Y=centroids),
                        columns=centroids.index.values,
                        index=X.index.values)
                        
def assign_label_from_centroid_distances(distances_to_centroids):
    """ From a N_samples x N_centroids distance matrix, return an array of assigned labels.
    
    For each sample, the assigned label corresponds to the centroid label with the minimum
    distance.
    
    Labels are assigned from the column names of the <distances_to_centroids> dataframe.
    
    In:
        distances_to_centroid (pandas.DataFrame): N_samples x N_centroids, with label names as columns
        
    Return:
        numpy.array of centroid labels, one per sample.
    """
    return distances_to_centroids.apply(lambda row: row.idxmin(),axis=1).values


def get_list_distances_per_centroid(distances_to_centroids, assigned_labels):
    """ From a N_samples x N_centroids dataframe, returns the list of distances per centroid label.
    
    This is used to evaluate the "outliers" when assigning a new sample to a cluster : if a
    sample is assigned to a cluster but has a distance that is way above the training distribution,
    then this assignment should be considered carefully.
    
    In:
        distances_to_centroids (pandas.DataFrame): N_samples x N_centroids, with label names as columns
        assigned_labels (numpy.array): N_samples x 1 ; labels of the centroids.
        
    Out:
        pandas.DataFrame ; of shape N_samples x 3 ;
        columns are: 'label', 'index_sample', 'distance_to_centroid'
    """
    return distances_to_centroids.assign(label=assigned_labels
                   ).groupby('label').apply(
                    lambda g: g.apply(lambda row: row.min(),axis=1)
                   ).reset_index().set_axis(['label','index_sample','distance_to_centroid'],axis=1,inplace=False)

