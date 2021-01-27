#! /usr/bin/env python
# coding:utf-8

import numpy as np
import pandas as pd
import sklearn

import matplotlib as mpl
import matplotlib.pyplot as plt



# Model comparison
# ================


def compute_roc_and_pr(y_true, y_pred, invert_roc_if_below_random=False):
    """ Compute the ROC and PR curves coordinates.
    
    This function will make use of the sklearn metrics functions to calculate
    two pairs of coordinates : one for the ROC and one for the Precision Recall.
    
    The coordinates are interpolated from the "true" values of TPR, FPR, and precision,
    over a range of 101 values on the x-axis. 
    This allows to compute the ROC and PR curves even when there are no thresholds found.
    
    Args:
        y_true (np.array) : 1-D array of true labels (binary classification).
        y_pred (np.array) : 1-D array of predicted values (continuous, discrete, or binary).
        invert_roc_if_below_random (bool, default=False) : whether to invert the y_true when the ROC AUC is below 0.5.
    
    Returns:
        tuple with :
        - tuple of 2*101 values, ordered as FPR and TPR (x and y for ROC curve)
        - ROC-auc score
        - tuple of 2*101 values, ordered as TPR and precision (x and y for PR curve)
        - PR-auc score
    """
    base_x = np.linspace(0,1,101)
    
    # ROC
    roc_fpr, roc_tpr, roc_thresh = sklearn.metrics.roc_curve(y_true, y_pred)
    # Interpolating for forcing an increasing range of values for the x-axis.
    interp_roc_tpr = np.interp(base_x, roc_fpr, roc_tpr)
    roc_auc_score = sklearn.metrics.auc(base_x, interp_roc_tpr)
    
    if invert_roc_if_below_random:
        if roc_auc_score < 0.5:
            print("Inverting the y_true")
            y_true = 1-y_true
          
            # recalculate the ROC
            roc_fpr, roc_tpr, roc_thresh = sklearn.metrics.roc_curve(y_true, tmp_X[column])
            roc_auc_score = sklearn.metrics.auc(roc_fpr, roc_tpr)
            interp_roc_tpr = np.interp(base_x, roc_fpr, roc_tpr)
            roc_auc_score = sklearn.metrics.auc(base_x, interp_roc_tpr)


    # PR
    pr_prec, pr_rec, pr_thresh = sklearn.metrics.precision_recall_curve(y_true, y_pred)
    if np.isnan(pr_rec).any():
        np.nan_to_num(pr_rec, copy=False)
        
    pr_prec, pr_rec = pr_prec[::-1], pr_rec[::-1]
    interp_pr_prec = np.interp(base_x, pr_rec, pr_prec)
    pr_auc_score = sklearn.metrics.auc(base_x, interp_pr_prec)
    
    return ((base_x, interp_roc_tpr), roc_auc_score,
            (base_x, interp_pr_prec), pr_auc_score)


def compute_PRC_ROC_coordAndAUCS_single(table_scores_nona, y_true, columns_scores):
    """ From a single dataframe, compute the ROC and PRC (and associated AUCs) from each column.
                    
        Args:
            table_scores_nona (pandas.DataFrame): columns as score names, and rows as scored variants.
            y_true (np.array): array of 0,1 values, one per variant.
            columns_scores (list): list of score names to retrieve from the dataframe.
        
        Returns:
            tuple of dictionaries with score names as the keys of each dictionary:
            (models_comp_rocs,
             models_comp_roc_aucs,
             models_comp_prcs,
             models_comp_prc_aucs)
                                     
    """
    if not (len(y_true) == table_scores_nona.shape[0]):
        raise ValueError('y_true and table_scores_nona are not the same dim.')
                    
    models_comp_rocs = {}
    models_comp_roc_aucs = {}
    models_comp_prcs = {}
    models_comp_prc_aucs = {}

    for score in columns_scores:
        ((base_x, interp_roc_tpr),
          roc_auc_score,
          (base_x, interp_pr_prec),
          pr_auc_score) = compute_roc_and_pr(y_true,
                                             list(table_scores_nona.loc[:,score].values))
                
        models_comp_rocs[score] = interp_roc_tpr
        models_comp_roc_aucs[score] = roc_auc_score
        models_comp_prcs[score] = interp_pr_prec
        models_comp_prc_aucs[score] = pr_auc_score
            
    return (models_comp_rocs,
            models_comp_roc_aucs,
            models_comp_prcs,
            models_comp_prc_aucs)


def compute_PRC_ROC_coordAndAUCS_kfolds(kfold_results, scored_df, columns_scores):
    """ From a KfoldResults, compute the ROC and PRC (and associated AUCs) from each column.
        
    Args:
        kfold_results (KfoldResults) : a structure that holds all kfolds related structure (row indices, y_true values, etc.)
        scored_df (pandas.DataFrame) : full dataframe of variants scored by the different methods to compare
        columns_scores: list of score names to retrieve from the dataframe
        
    Returns:
        tuple of dictionaries with score names as the keys of each dictionary:
        (models_comp_rocs, models_comp_roc_aucs,
         models_comp_prcs, models_comp_prc_aucs)
                                             
    """
    models_comp_rocs = {}
    models_comp_roc_aucs = {}
    models_comp_prcs = {}
    models_comp_prc_aucs = {}


    mean_perc_removed = []
    mean_N_used = []
    mean_perc_pos = []

    for i in range(len(kfold_results.test_pred)):
        print(i)
        
        kfold_scores_df = scored_df.iloc[kfold_results.test_indices[i],:].copy()
        kfold_scores_df['FINSURF_HGMD'] = kfold_results.test_pred[i][:,1]
        kfold_scores_df['true_y']= kfold_results.test_y[i]
        kfold_scores_df = kfold_scores_df.loc[:,columns_scores+['FINSURF_HGMD','true_y']]

        # We will remove any variant that is missing at least 1 of the scores.
        # First calculate the number, then drop such variants
        perc_removed = kfold_scores_df.isnull().any(axis=1).sum()/ kfold_scores_df.shape[0]*100
        mean_perc_removed.append(perc_removed)
        print("\tRemoving {}% of {:,} samples".format(perc_removed, kfold_scores_df.shape[0]))
        kfold_scores_df = kfold_scores_df.dropna()

        # Report the percentage of positive samples
        perc_pos = (kfold_scores_df['true_y'].sum() / kfold_scores_df.shape[0]*100)
        print("\tPercentage of positive samples: {}%".format(perc_pos))
        mean_perc_pos.append(perc_pos)

        # Report number of actually-used variants 
        mean_N_used.append(kfold_scores_df.shape[0])

        # And finally : for each score, calculate ROC and PRC values
        for score in columns_scores+['FINSURF_HGMD']:
            ((base_x, interp_roc_tpr),
             roc_auc_score,
             (base_x, interp_pr_prec),
             pr_auc_score) = compute_roc_and_pr(kfold_scores_df.loc[:,'true_y'].values,
                                                kfold_scores_df.loc[:,score].values)

            if i>0:
                models_comp_rocs[score].append(interp_roc_tpr)
                models_comp_roc_aucs[score].append(roc_auc_score)
                models_comp_prcs[score].append(interp_pr_prec)
                models_comp_prc_aucs[score].append(pr_auc_score)
            else:
                models_comp_rocs[score] = [interp_roc_tpr]
                models_comp_roc_aucs[score] = [roc_auc_score]
                models_comp_prcs[score] = [interp_pr_prec]
                models_comp_prc_aucs[score] = [pr_auc_score]

        print("\n")

    # Finale task : calculate average value from the kfolds ROC and PRC values.
    models_comp_rocs = {k:np.nanmean(np.array(v),axis=0)
                        for k, v in models_comp_rocs.items()}
    
    models_comp_roc_aucs = {k:np.mean(v) for k, v in models_comp_roc_aucs.items()}
        
    models_comp_prcs = {k:np.nanmean(np.array(v),axis=0)
                       for k, v in models_comp_prcs.items()}
            
    models_comp_prc_aucs = {k:np.mean(v) for k, v in models_comp_prc_aucs.items()}
                
    print("Mean percentage of samples removed: {}".format(np.mean(mean_perc_removed)))
    print("Mean Number of samples used: {}".format(np.mean(mean_N_used)))
    print("Mean percentage of positives: {}".format(np.mean(mean_perc_pos)))
                            
    return (models_comp_rocs, models_comp_roc_aucs,
            models_comp_prcs, models_comp_prc_aucs)


def plot_comparison_rocsAndPRCs(models_comp_rocs, models_comp_roc_aucs,
                                models_comp_prcs, models_comp_prc_aucs,
                                score_to_color,
                                show_plot=True,
                                savefig_file=None):
    """From dicts of performance-curve values, plot ROC and PRC 
    
    Args:
        models_comp_rocs (dict): per-score array of TPR values
        models_comp_roc_aucs (dict): per-score ROC AUC
        models_comp_prcs (dict): per-score array of precisions values
        models_comp_prc_aucs (dict): per-score PRC AUC  
        score_to_color (dict) : map score name to RGB string

    """
    # IMPORTANT: Check that you have the same vector here and in the
    # <compute_roc_and_pr> function.
    base_x = np.linspace(0,1,101)
    
    fig = plt.figure(figsize=(18,8))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2,sharey=ax1)

    for score in sorted(models_comp_rocs.keys(),
                        key=lambda v: models_comp_roc_aucs[v],reverse=True):
        linewidth=2
        linestyle='-'
        ax1.plot(base_x, models_comp_rocs[score],
                 linewidth=linewidth,
                 linestyle=linestyle,
                 color=score_to_color[score],
                 label="{} - {:.3}".format(score, models_comp_roc_aucs[score])
                )
        
        ax2.plot(base_x, models_comp_prcs[score],
                 linewidth=linewidth,
                 linestyle=linestyle,
                 color=score_to_color[score],
                 label="{} - {:.3}".format(score, models_comp_prc_aucs[score])
                )
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax1.legend(bbox_to_anchor=(1,1))
    ax2.legend(bbox_to_anchor=(1,1))

    ax1.set_ylabel('TPR')
    ax1.set_xlabel('FPR')
    ax2.set_ylabel('Precision')
    ax2.set_xlabel('TPR')


    if savefig_file:
        plt.savefig(savefig_file)
        
    if show_plot:
        plt.show()


def scatterplot_aucs(models_prc_aucs, models_roc_aucs,
                     score_to_color,
                     sort_by='roc',
                     y_true=None,
                     y_true_proportion=None,
                     full_range_roc=False,
                     full_range_prc=False,
                     ax=None,
                     show_plot=True,
                     savefig_file=None):
    """Scatterplot of AUC values for all models in the dicts.
    
    Note : you have to provide either y_true a binary vector of sample labels
    or the <y_true_proportion>, so that a baseline for the AUPRC (equal to
    sum(y_true) / length(y_true)) can be written on the plot.

    Args:
        models_prc_aucs (dict) : map score name to its PRC AUC
        models_roc_aucs (dict) : map score name to its ROC AUC,
        score_to_color (dict) : dictionary mapping score name to color
        y_true (numpy.array, default=None) : array of 0/1 values indicating true class.
        y_true_proportion (float, default=None) : value between 0 and 1 as the proportion of positives in the dataset
        full_range_roc (bool, default=False) : whether to show the full [0-1] AUC range
        full_range_prc (bool, default=False) : whether to show the full [0-1] AUC range
        show_plot (bool, default=True)
        savefig_file (bool, default=True)
    """
    sort_by_dict = {
            'roc':models_roc_aucs,
            'ROC':models_roc_aucs,
            'prc':models_prc_aucs,
            'PRC':models_prc_aucs,
            }

    if not sort_by in sort_by_dict.keys():
        raise ValueError("`sort_by` should be in {}".format(sort_by_dict.keys()))

    if (y_true is None) and (y_true_proportion is None):
        raise ValueError("Missing either y_true or y_true_proportion")
    
    if y_true_proportion:
        if y_true_proportion>1 or y_true_proportion<0:
            raise ValueError("Incorrect value of y_true (should be between 0 and 1)")
    else:
        y_true_proportion = y_true.sum()/ y_true.shape[0]

    if not ax:
        fig = plt.figure(figsize=(12,6.5))
        ax1 = fig.add_subplot(1,1,1)
    else:
        ax1 = ax
    
    if full_range_roc:
        ax1.set_xlim(-0.03, 1.03)
    else:
        minx = min(min(models_roc_aucs.values()),0.5)*0.9
        maxx = 1*1.03
        ax1.set_xlim(minx, maxx)
        
    if full_range_prc:
        ax1.set_ylim(-0.02, 1.02)
    else:
        miny=-0.02
        maxy = max(models_prc_aucs.values())*1.10
        ax1.set_ylim(miny, maxy)
    
    ax1.set_xlabel("ROC AUC")
    ax1.set_ylabel("PRC AUC")

    sorted_models = [it[0] for it in sorted(sort_by_dict[sort_by].items(),
                                            key=lambda item : item[1],
                                            reverse=True)]

    for model in sorted_models:
        ax1.scatter(models_roc_aucs[model],
                    models_prc_aucs[model],
                    color=score_to_color[model],
                    edgecolors="#FFFFFF",
                    s=150,
                    label=f"{model} ({models_roc_aucs[model]:.3},{models_prc_aucs[model]:.3})"
                   )
                        
    # Let's add the horizontal and vertical lines for a random model
    # (0.5 for ROC, proportion of positive for PRC)
    ax1.axvline(0.5,
                color="#444444",
                linestyle="--",
                linewidth=1.2)
    
    ax1.axhline(y_true_proportion,
                color="#444444",
                linestyle="--",
                linewidth=1.2,
                label=("Random: {:.3}, {:.3}".format(0.5, y_true_proportion))
                )

    ax1.legend(bbox_to_anchor=(1,1))
    #ax1.set_aspect("equal")

    
    if savefig_file:
        plt.savefig(savefig_file)
        
    if show_plot:
        plt.show()
