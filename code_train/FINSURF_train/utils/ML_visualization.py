#! /usr/bin/env python
# -*- coding:utf-8 -*-


###############################################################################
# IMPORTS


import numpy as np
import pandas as pd
import sklearn
import sys

import itertools as itt

from FINSURF_train.utils.ML_tests import Clf_scores
from FINSURF_train.utils.ML_tests import evaluate_classifier_preds
from FINSURF_train.utils.ML_tests import named_confusion_matrix
from FINSURF_train.utils.ML_tests import label_majority_class_assignment

from scipy import interp

from IPython.display import display

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
# DEFINITIONS


def viewPydot(pdot):
    plt=Image(pdot.create_png())
    display(plt)


###############################################################################
# PART0: tweaking plots
# =====================


def add_pval_label(ax, label, overlapping=False):
    ymin, ymax = ax.get_ylim()
    connection = ax.annotate("", xy=(0, ymax*1.05), xycoords='data',
                 xytext=(1, ymax*1.05), textcoords='data',
                 arrowprops=dict(arrowstyle="-", ec='#888888',
                 connectionstyle="bar,fraction=0.15",linewidth=2,facecolor='#000000'))

    if overlapping:
        # NOTE: a bit messy ; ideally we would need to get the y position of the
        # connection arrow, to be able to align the text to it.
        ax.text(0.5, ymax+abs(ymax-ymin)*0.20,
                 label,
                 backgroundcolor='#FFFFFF',
                 horizontalalignment='center',
                 verticalalignment='center',
                 color='#222222',
                 weight='regular')
    else:
        ax.text(0.5, ymax+abs(ymax-ymin)*0.40,
                 label,
                 horizontalalignment='center',
                 verticalalignment='center',
                 color='#222222',
                 weight='regular')

    ax.set_ylim(ymin, ymax+abs(ymax - ymin)*0.65)



def patch_labelling(ax, patch, label, vertical=True, shift=0, revert=False, textparams={}):
    """ Function to add a label to a patch in a barplot.
    
    In:
        ax (AxesSubplot): ax on which to act
        patch (matplotlib.patches.Rectangle): patch to label
        label (str)
        vertical [(bool, True)]: whether the barplot is vertical (value on y axis) or not
        shift [(int, 0)]: shift of the x/y position for vertical/horizontal plot
        textparams [(dict, {})]: parameters for the label text.
    """
    # Number of points between bar and label. Change to your liking.
    space = 3
    
    if vertical:
        y_value = patch.get_height()
        x_value = patch.get_x() + patch.get_width() / 2 + shift
        
        ha='center'
        rotation=0

        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'
            
        xytext = (0, space)
            
    else:
        x_value = patch.get_width()
        y_value = patch.get_y() + patch.get_height() / 2 + shift
        
        # Horizontal alignment
        ha='center'
        
        # Vertical alignment
        va='center'
        rotation=-90
        
        if x_value <0 or revert:
            # Invert space to place label to the left
            space *=-1
            rotation=90

        xytext = (space, 0)       
        
    # Create annotation
    ax.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=xytext,              # Shift label
        xycoords='data',
        textcoords="offset points", # Interpret `xytext` as offset in points
        ha=ha,                      
        va=va,
        rotation=rotation,
        #bbox = dict(boxstyle="rarrow,pad=0.3", fc="cyan", ec="b", lw=2),
        **textparams)               
        
    return


###############################################################################
# PART1: FEATURE PLOTS
# ====================

def plot_feature_distributions(X, y, class_names, col_plottypes,
                               feature_colors=None,
                               order_classes = None,
                               palette=None,
                               pval_dict=None,
                               title='',
                               title_color='#020202',
                               remove_outliers_perc=0.0,
                               verbose=False,
                               savefig_file=None,
                               show_plot=True):
    """ Plot distributions per column from a dataframe.

    Plot distributions of values from the annotations found in a dataframe,
    separating classes according to <y>, named using the <class_names>.

    Distribution types are set from the <cols_type> dictionary.

    In:
        X (pandas.dataframe): of annotations
        y (numpy.array): with target class as 0,1
        class_names (dict): dictionary mapping y values to class names
        col_plottypes (dict): dictionary mapping each column to plot to a type of
                          data (binary, discrete, or continuous).

    """
    reduced_cols = []
    for c in X.columns:
        if c not in col_plottypes:
            print('{} has no defined col_type ; skipped'.format(col))
            continue
        reduced_cols.append(c)

    X = X.loc[:,reduced_cols]

    Tot = len(X.columns)
    Cols = 3
    Rows = Tot // Cols 
    Rows += Tot % Cols
    Position = range(1,Tot + 1)

    width = 22
    height = 4 * Rows

    if order_classes is None:
        if isinstance(class_names, dict):
            order_classes = list(class_names.values())
        else: 
            order_classes = list(class_names)

    if palette is None: palette = 'Set2'

    fig = plt.figure(figsize=(width,height))
    datas = []

    for i, col in enumerate(X.columns):
        if verbose:
            print("{} : {}".format(i, col))

        plot_type = col_plottypes[col]
        
        pos = Position[i]
        ax = fig.add_subplot(Rows,Cols,pos)

        # Create the temporary dataframe for the plot.
        data = pd.concat([pd.Series(X.loc[:,col].values,name=col),
                          pd.Series([class_names[v] for v in y], name='class'),
                         ], axis=1)
        data.index = X.index

        datas.append(data)

        # Remove outliers for plot.
        if remove_outliers_perc>0 and plot_type!='binary':
            is_within_quantil = np.logical_and(data[col]>data[col].quantile(remove_outliers_per),
                                               data[col]<data[col].quantile(1-remove_outliers_per))
            reduced_data = data.loc[is_within_quantil, :]
        else:
            reduced_data = data

        if plot_type=='continuous':
            ylab = 'score'
            sns.violinplot(x='class', y = 'value', 
                           data=reduced_data.melt(id_vars='class'),
                           order=order_classes,
                           orient='v',
                           split=False,
                           showfliers=False,
                           palette=palette,
                           cut=0,
                           ax=ax)

        elif plot_type=='discrete':
            ylab = 'score'
            sns.boxplot(x='class', y = 'value', data=reduced_data.melt(id_vars='class'),
                        order=order_classes, orient='v',
                        showfliers=False, palette=palette,
                        ax=ax)
            
            
        elif plot_type=='binary':
            ylab = 'proportion'
            sns.barplot(x='class', y = 'value', data=reduced_data.melt(id_vars='class'),
                        orient='v', order=order_classes, ci=False, 
                        edgecolor='.2',
                        linewidth=2.0,
                        palette=palette,
                        ax=ax)
            ax.set_ylim(0, ax.get_ylim()[1])

        else:
            print(("Error for column '{}': col type '{}' not "
                   "recognized").format(col, plot_type))

        ax.set_xlabel('')
        ax.set_ylabel(ylab)
        if feature_colors:
            color = feature_colors.get(col,'#444444')
        else:
            color = '#444444'

        ax.set_title(col, color=color)
    
        # Add pval symbol if pval_dict is provided
        if pval_dict:
            stars = pval_dict.get(col,'')
            if '*' in stars:
                # Significant test ; add the annotation.
                add_pval_label(ax,stars)


    # Add legend box to the last plot.
    fig.axes[-1].legend(bbox_to_anchor=(1,1))

    if savefig_file:
        plt.savefig(savefig_file)

    if show_plot:
        plt.show()

    return datas, fig


def plot_predictors_from_features(X, y,
                                  title=None,
                                  title_color='#BBBBBB',
                                  rename_cols=None,
                                  savefig_file=None,
                                  show_plot=True):
    """ Plot for each column in X the ROC-curve and PREC-REC curves.

    CAUTION: for classification task only.

    In:
        X (pandas.DataFrame): dataframe of samples described by features
        y (numpy.array): array of classes.
        title ([str])
        title_color ([str,default='#BBBBBB')
        rename_cols ([list, default=None]): if provided, rename the columns
                                            according to the values in the
                                            list.
        savefig_file ([str, default=None]): if provided, save the plot to the
                                            provided filepath.
        
        show_plot ([bool, default=True])

    """
    Tot = len(X.columns) * 2  # two plots per feature
    Cols = 4
    Rows = Tot // Cols
    Rows += Tot % Cols
    Position = range(1, Tot + 1)

    height = 4 * Rows
    width = 25 


    all_auc_scores = []
    all_eval_scores = []

    fig = plt.figure(figsize=(width, height))

    for i, c in zip(range(0, len(X.columns)*2, 2), X.columns):

        pos_roc = Position[i]
        pos_pr = Position[i + 1]
        ax1 = fig.add_subplot(Rows, Cols, pos_roc)
        ax2 = fig.add_subplot(Rows, Cols, pos_pr)

        y_preds = X.loc[:, c]
        y_true = y

        # Here we need to set the y_pred to a 0/1 value. We just take the mean
        # value and define all values above as positive.
        tmp_y_pred = (y_preds>y_preds.mean()).astype(int)
        eval_scores = evaluate_classifier_preds(y_true, tmp_y_pred)
        all_eval_scores.append(eval_scores)

        auc_score = sklearn.metrics.roc_auc_score(y_true, y_preds)
        all_auc_scores.append(auc_score)

        # I used to "correct" models that seemed to invert the prediction classes ;
        # But actually if a model is performing worse than expected, this can highlight
        # properties from the training set.
        # if auc_score < 0.5:
        #     y_true = 1 - y_true
        #     auc_score = 1 - auc_score

        roc_fpr, roc_tpr, roc_thresh = sklearn.metrics.roc_curve(y_true, y_preds)
        pr_prec, pr_rec, pr_thresh = sklearn.metrics.precision_recall_curve( y_true, y_preds)
        pr_prec, pr_rec = pr_prec[::-1], pr_rec[::-1]


        if rename_cols is not None:
            name_col = rename_cols[int(i/2)]
        else:
            name_col = c

        ax1.plot( roc_fpr, roc_tpr, color='#ea1717', label="AUC: {:.2f}".format(auc_score))

        ax1.plot([0, 1], [0, 1], linestyle='--', color='#999999')
        ax1.set_ylim(-0.01, 1.01)
        ax1.set_xlim(-0.01, 1.01)
        ax1.set_ylabel("TPR")
        ax1.set_xlabel("FPR")
        ax1.set_title("{}:\nROC curve".format(name_col))
        ax1.legend(loc='lower right')

        ax2.plot(pr_rec, pr_prec, color='#6be500')
        
        
        worst_prec_rec = (y==pd.Series(y).value_counts().idxmax()).sum() / y.shape[0]
        ax2.axhline(1-worst_prec_rec, linestyle='--', color='#999999')
        ax2.set_ylim(-0.01, 1.01)
        ax2.set_xlim(-0.01, 1.01)
        ax2.set_ylabel("Precision")
        ax2.set_xlabel("TPR")
        ax2.set_title("{}:\nPREC-REC curve".format(name_col))

        ax1.set_aspect('equal')
        ax2.set_aspect('equal')

    if title:
        plt.suptitle(title,fontsize=28,color=title_color,
                    x=0.5, y=1.01)

    plt.tight_layout()

    if savefig_file:
        plt.savefig(savefig_file)


    if show_plot:
        plt.show()

    return pd.DataFrame(all_eval_scores, X.columns).assign(auc_score=all_auc_scores)


def heatmap_relative_changes(list_relative_changes, heatmap_columns,
                             list_pvals_df=None,
                             mask_nonsign=False,
                             my_cmap = sns.diverging_palette(250,15,n=21),
                             feature_colors=None,
                             title=None,
                             title_color='#000000',
                             savefig_file=None, show_plot=True
                            ):
    """ Plot a heatmap of significant changes from a list of change values.
    
    In:
        list_relative_changes (list): list of columns to be plotted. Each element should be a pandas.Series of features with value.
        heatmap_columns (list): list of names for the heatmap (name of each element of <list_relative_changes>)
        list_pvals_df [(list)]: list of pval dataframes. Should associate for each feature (in index) a "corrected_pvalue" column.
        mask_nonsign
    """
    
    if mask_nonsign:
        assert list_pvals_df is not None, "Error: <mask_nonsign> is True, but no list of pvals_df provided..."
    
    # color map of the ratios is given by the user. Default to blue => red.
    # Grey / invisible color map for masking non-significant changes
    flatui = ['#FFFFFF', '#333333']
    cmap_mask = mpl.colors.ListedColormap(sns.color_palette(flatui).as_hex())
    # Get colors
    my_cmap_mask = cmap_mask(np.arange(cmap_mask.N))
    # Set alpha (white tiles should be invisible)
    my_cmap_mask[:,-1] = np.linspace(0,0.4, cmap_mask.N)
    # Create new colormap
    my_cmap_mask = mpl.colors.ListedColormap(my_cmap_mask)
    
    # First concat all relative changes arrays into a single dataframe.
    df_relative_changes = pd.concat(list_relative_changes,
                                    axis=1).set_axis(heatmap_columns, axis=1, inplace=False)
    
    
    if mask_nonsign:
        # Same for pvals (using the 'corrected_pval' for each dataframe, and setting the 'feature' column as index)
        df_pvals = pd.concat([pval_df.set_index('feature')['corrected_pval'] for pval_df in list_pvals_df],
                             axis=1).set_axis(heatmap_columns, axis=1, inplace=False)

        # Match the index of the effect tables.
        df_pvals = df_pvals.loc[df_relative_changes.index.values]

        df_pvals = df_pvals>=0.05 # Set True for non-sign pvals.
    
         

    # The matrix will be transposed, so larger than tall.
    size_height = 0.9 * len(list_relative_changes)
    size_width = 0.5 * list_relative_changes[0].shape[0]
    
    min_size = 13
    
    size_side = max(size_height,size_width)
    size_side = max(min_size,size_side)

    fig = plt.figure(figsize=(size_side, size_side))
    ax1 = fig.add_subplot(1,1,1)
    
    sns.heatmap(df_relative_changes.T,
                cmap=my_cmap,
                #vmin = -df_relative_changes.max().max()*0.9,
                #vmax= df_relative_changes.max().max()*0.9,
                robust=True,
                center=0,
                linewidth=1.5,
                linecolor='#FFFFFF',
                cbar_kws={"shrink": .30},
                square='True',
                ax=ax1)

    
    if mask_nonsign:
        sns.heatmap(df_pvals.T, cmap=my_cmap_mask, cbar=False, ax=ax1)
        
    # Show border of the heatmap
    for _, spine in ax1.spines.items():
        spine.set_visible(True)
             

    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, ha='right')
    
    # Add colors to feature names.
    if feature_colors:
        for feature_tick in ax1.get_xticklabels():
            feature = feature_tick.get_text()
            color = feature_colors[feature]
            feature_tick.set_color(color)


    ax1.set_xlabel('')
    
    plt.tight_layout()
    
    if title:
        ax1.set_title(title, fontsize=28, color=title_color, y=1.1)
    
    if savefig_file:
        plt.savefig(savefig_file)
        
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return


###############################################################################
# PART2: CLASSIFIERS PLOTS
# ========================


# Classification quality
# -----------------------------------------------------------------------------


def plot_roc_curve(kf_predictions, color='#cf385b',
                   color_surface='#777777',
                   mean_only=False,
                   ax=None, savefig_file=None, show_plot=True):
    """ Plot ROC curves from a list of k-fold predictions
    
    The input object should be a list of (expected,probabilities) arrays, with
    the 'probabilities' array having for each sample 2 values: one for the
    first class, one for the second.
    
    Display both the different curves from each k-fold, plus a mean curve
    calculated from the predictions in each k-fold.
    """
    if ax is None:
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(1,1,1)
    
    # This list will contain data for creating the mean curve from the
    # k-fold predictions.
    tprs = []
    fprs = []
    inter_tprs = []
    base_fpr = np.linspace(0, 1, 101)

    for kf_res in kf_predictions:
        y_true,y_prob = kf_res[0],kf_res[1]
        fpr, tpr, roc_thresholds = sklearn.metrics.roc_curve(y_true,y_prob[:,1],drop_intermediate=False)

        if not mean_only:
            ax.plot(fpr,tpr,alpha=0.2,color=color)

        tprs.append(tpr)
        fprs.append(fpr)

        # Interpolate values of y for each x in base_fpr, by guessing the function tpr = f(fpr)
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        inter_tprs.append(tpr)

    inter_tprs = np.array(inter_tprs)
    mean_tprs = np.nanmean(inter_tprs,axis=0)
    std = np.nanstd(inter_tprs,axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    auc_mean = sklearn.metrics.auc(base_fpr,mean_tprs)
    
    ax.plot(base_fpr, mean_tprs, linewidth=2, color=color,
             label='Mean ROC curve\n(AUC={:.4})'.format(auc_mean))
    
    ax.fill_between(base_fpr, tprs_lower, tprs_upper, color=color_surface, alpha=0.3)

    # Random classifier
    ax.plot([0, 1], [0, 1],'--',color='#777777')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate / Recall')
    ax.set_title('ROC curve')
    ax.set_xlim(-0.01,1.01)
    ax.set_ylim(-0.01,1.01)    
    ax.legend()

    ax.set_aspect('equal')
    
    plt.tight_layout()

    if savefig_file:
        plt.savefig(savefig_file)

    if show_plot:
        plt.show()

    # Let's return AUC values per kfold as well as the
    # mean AUC from the interpolated data.
    auc_vals = [sklearn.metrics.auc(a,b)
                for a,b in zip(fprs,tprs)
                ]
    auc_vals.append(auc_mean)
    all_auc = pd.Series(auc_vals,
                        index=['kfold_{}'.format(i)
                               for i in
                               range(len(kf_predictions))]+\
                               ['mean_intercept']
                               )
    return all_auc, ax


def plot_precrec_curve(kf_predictions,
                       color='#cf385b',
                       color_surface='#777777',
                       mean_only=False,
                       ax=None, savefig_file=None, show_plot=True):
    """ Plot precision-recall curves from a list of k-fold predictions,
        with expected classes and probabilities for each k-fold.

    The input object should be a list of (expected,probabilities) arrays, with
    the 'probabilities' array having for each sample 2 values (one for the 1st
    class, one for the second).

    Displays both the different curves from each k-fold, plus a mean curve
    calculated from the predictions in each k-fold.
    """
    if ax is None:
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(1,1,1)
    
    # This list will contain data for creating the mean curve
    # from the k-fold predictions.
    random_clf_preds = []
    recs = []
    precs = []
    interp_precs = []
    base_rec = np.linspace(0, 1, 101)

    
    for kf_res in kf_predictions:
        y_true, y_prob = kf_res[0], kf_res[1]
        prec, rec, precrec_thresholds = sklearn.metrics.precision_recall_curve(y_true,y_prob[:,1])
        if np.isnan(rec).any():
            np.nan_to_num(rec, copy=False)

        prec, rec = prec[::-1], rec[::-1]
        if not mean_only:
            ax.plot(rec,prec,alpha=0.2,color=color)

        recs.append(rec)
        precs.append(prec)
        
        # Interpolate values of y for each x in base_rec, by guessing the
        # function rec = f(prec)
        prec = np.interp(base_rec, rec, prec)
        interp_precs.append(prec)
    
        rand_clf = 1-(y_true==pd.Series(y_true).value_counts().idxmax()).sum()/y_true.shape[0]
        random_clf_preds.append(rand_clf)
    

    interp_precs = np.array(interp_precs)
    mean_precs = interp_precs.mean(axis=0)
    std = interp_precs.std(axis=0)
    
    precs_upper = np.minimum(mean_precs + std, 1)
    precs_lower = mean_precs - std
    
    auc_mean = sklearn.metrics.auc(base_rec, mean_precs)
    
    ax.plot(base_rec, mean_precs, color=color,
            label='Mean precision-recall curve\n(AUC={:.4})'.format(auc_mean))
    ax.fill_between(base_rec, precs_lower, precs_upper, color=color_surface,alpha=0.3)
        

    # Add the random clf constant.
    ax.axhline(np.mean(random_clf_preds),linestyle='--',color='#888888')

    ax.set_xlabel('True Positive Rate / Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall curve')
    ax.set_xlim(-0.01,1.01)
    ax.set_ylim(-0.01,1.01)    
    ax.legend()

    ax.set_aspect('equal')
    
    plt.tight_layout()

    if savefig_file:
        plt.savefig(savefig_file)

    if show_plot:
        plt.show()

    # Let's return AUC values per kfold as well as the
    # mean AUC from the interpolated data.
    auc_vals = [sklearn.metrics.auc(a,b)
                for a,b in zip(recs,precs)
                ]
    auc_vals.append(auc_mean)
    all_auc = pd.Series(auc_vals,
                        index=['kfold_{}'.format(i)
                               for i in
                               range(len(kf_predictions))]+\
                               ['mean_intercept']
                               )
    return all_auc


def plot_predproba_distributions(kf_results,
                                   class_names = ['Negatives', 'Positives'],
                                   color_positives='#cf385b',
                                   color_negatives='#43b0d0',
                                   ax=None,
                                   savefig_file=None,
                                   show_plot=True):

    """ Density plots of results from <stratifiedkfold_predict> function.
    
    In:
        kf_results (list): list containing tuples of k-fold prediction results.
                           Each element is a tuple with
                           - the true labels (pandas.Series)
                           - the predictions (2d numpy array)
                           - the index of rows used for the test set
                           - the trained model
                           
        savefig_file ([path_to_file]) : if provided, save the plot to file.
    """
    if ax is None:
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(1,1,1)

    # Accumulate plots from each k-fold - POSITIVES
    positives_scores = [kf_r[1][(kf_r[0]==1),1] for kf_r in kf_results]
    
    for pos_sc in positives_scores:
        sns.kdeplot(pos_sc,
                alpha = 0.15,
                color=color_positives,
                shade=True,
                cut=2,
                ax=ax)

    # Plot the overall density plot - POSITIVES
    sns.kdeplot(list(itt.chain(*positives_scores)),
        shade = False,
        alpha = 0.25,
        color=color_positives,
        cut=2,
        ax=ax)


    # Accumulate plots from each k-fold - NEGATIVES
    negatives_scores = [kf_r[1][(kf_r[0]==0),1] for kf_r in kf_results]
    
    for neg_sc in negatives_scores:
        sns.kdeplot(neg_sc,
                    alpha = 0.15,
                    color=color_negatives,
                    shade=True,
                    cut=2,
                    ax=ax)

    # Plot the overall density plot - NEGATIVES
    sns.kdeplot(list(itt.chain(*negatives_scores)),
                shade = False,
                alpha = 0.25,
                color=color_negatives,
                cut=2,
                ax=ax)

    max_ylim = ax.get_ylim()[1]
    max_ylim *= 1.1
    ax.set_ylim(0,max_ylim)
    ax.set_xlim(-0.08,1.08)

    ax.set_ylabel('Density')
    ax.set_xlabel('Prediction score')

    red_patch = mpl.patches.Patch(color=color_positives,
                                   alpha=0.5,
                                   label=class_names[1])
    
    blue_patch = mpl.patches.Patch(color=color_negatives,
                                   alpha=0.5,
                                   label=class_names[0])
    legend = ax.legend(handles=[red_patch,blue_patch],
                        bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0.)

    plt.tight_layout()
    
    ax.set_title('Density plot of prediction scores')

    if savefig_file:
        plt.savefig(savefig_file)

    if show_plot:
        plt.show()
    else:
        return ax


def separability_plots(kfold_results, class_names=['0','1'],
                       color_positives='#cf385b', color_negatives='#43b0d0',
                       color_surface='#777777',
                       mean_only=False,
                       main_title='',
                       show_plot=True, savefig_file=None):
    """ Joint plot of the separability of samples from kfold results.
    
    kfold_results should be composed of different elements. Each unit contains:
    - y_true
    - y_prob
    - test_indexes
    - classifier
    
    In:
        kfold_results (list)
        color_positives (str)
        color_negatives (str)
        main_title (str)
        show_plot
        savefig_file
    """
    fig = plt.figure(figsize=(21,6))
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)

    _ = plot_roc_curve(kfold_results,
                        color=color_positives,
                        color_surface=color_surface,
                        mean_only=mean_only,
                        ax=ax1,
                        show_plot=False)
    
    _ = plot_precrec_curve(kfold_results,
                           color=color_positives,
                           color_surface=color_surface,
                           mean_only=mean_only,
                            ax=ax2, show_plot=False)
    
    _ = plot_predproba_distributions(kfold_results,
                                       class_names=class_names,
                                        color_positives=color_positives,
                                        color_negatives=color_negatives,
                                        ax=ax3, show_plot=False)

    plt.suptitle(main_title, fontsize=18, fontweight='bold', y=1.10)
    
    if savefig_file:
        plt.savefig(savefig_file)
        
    if show_plot:
        plt.show()
    else:
        plt.close()





def plot_confusion_matrix(cm, classes,
                          cmap=None,
                          color_class=None,
                          title=None,
                          normalize=False,
                          max_count=None,
                          ax=None,
                          savefig_file=None,
                          show_plot=True):
    """ This function prints and plots the confusion matrix.

    The confusion matrix should be generated from sklearn.metrics.confusion_matrix
    Normalization can be applied by setting `normalize=True`.
    
    In:
        cm (np.array): 2d array for the confusion matrix.
        classes (array): names of the classes
        color_class: if not None, should indicate the 'target class' of interest (0, or 1)
        title (str)
        normalize (bool): whether to show percentage or not
        max_count (int, default=None): if provided, the heatmap is scaled according to it.
        ax (pyplot.Axis)
        savefig_file (str, default=None)
        show_plot (bool, default=True)
    """

    cm_cts = cm.copy()
    cm = cm.copy()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=3)
        max_count = 1

    else:
        if max_count is None:
            max_count = cm.max()

    # We introduce the color distinction between *predicted* positives and
    # *predicted* negatives by setting negative values to their opposite sign.
    if color_class is not None:
        cm[:,abs(color_class-1)] = - cm[:,abs(color_class-1)]

        min_count = - max_count 

        if cmap is None:
            cmap = sns.diverging_palette(160, 0, n=51) 


    else:
        annot_mat = cm
        cmap = plt.cm.Blues
        min_count = 0

    if normalize:
        # have both the percentage and count.
        annot_mat = (pd.DataFrame(cm.astype(str))+\
                     '\n(N='+\
                     pd.DataFrame(cm_cts).applymap(
                         lambda v: '{:,}'.format(v))+\
                     ')'
                     ).values 
        fmt=''
    else:
        annot_mat = cm
        fmt=','


    if ax is None:
        print("Creating new figure")
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1,1,1)

    sns.heatmap(cm,
                vmin=min_count,
                vmax=max_count,
                cmap=cmap,
                center=0,
                annot=annot_mat,
                fmt=fmt,
                annot_kws={"size":20},
                xticklabels=classes,
                yticklabels=classes,
                linewidth=1.2,
                square=True,
                cbar=False,
                cbar_kws={'shrink':0.6},
                ax=ax)

    ax.set_yticklabels(ax.get_yticklabels(),rotation=45, ha='right')
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45, ha='right')
    if title is not None:
        ax.set_title(title)

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    plt.grid(False)

    plt.tight_layout()

    if savefig_file is not None:
        plt.savefig(savefig_file)

    if show_plot:
        plt.show()
    else:
        return ax


def plot_classification_evaluation(true_y_list, pred_y_list, multi=True,
                                   target_class=1,
                                   title="", ax=None, show_plot=True, savefig_file=None):
    """ Plot different metrics associated to binary classification results.
    
    Barplot of quality metrics calculated from a list of true labels and predicted labels (one per sample for each).
    
    In:
        - list_true_y (list): list where elements are arrays of labels
        - list_pred_y (list): list where elements are arrays of labels
        - multi (bool): whether there are multiple array (e.g.: kfold results) or not.
        - target_class (int, default=1): whether the class of interest is 1 or 0 (for the f1, recall, and precision scores)
        - title ([str], default="")
        - ax (pyplot.Axe)
        - show_plot (bool, default=True)
        - savefig_file (str, default=None): path to file if you want to save the figure
        
    Out:
        if show_plot==False: return ax
    """
    assert len(true_y_list) == len(pred_y_list)
    assert target_class in (0,1), "Error: target_class should be 0 or 1"
    
    # Define different plot parameters basing on the "multi" options
    if multi:
        ci='sd'
        get_shift=lambda patch: patch.get_width()/4
    else:
        if isinstance(true_y_list[0], int):
            # Likely to mean that the provided list is directly the array of labels.
            # There must be a better test though.
            true_y_list = [true_y_list]
            pred_y_list = [pred_y_list]
        ci=None
        get_shift = lambda patch: 0
    
    if target_class == 0:
        list_true_y = [np.array([0 if v==1 else 1 for v in array]) for array in true_y_list]
        list_pred_y = [np.array([0 if v==1 else 1 for v in array]) for array in pred_y_list]

    else:
        list_true_y = true_y_list
        list_pred_y = pred_y_list
        
        
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(1,1,1)
        
    # Calculate the different metrics
    tmp_df = pd.DataFrame([evaluate_classifier_preds(y_true, y_pred) for (y_true,y_pred) in zip(list_true_y, list_pred_y)])

    sns.barplot(
        data=tmp_df.melt(),
        x='variable',
        y='value',
        color='#BBBBBB',
        ci=ci,
        ax=ax)
    
    ax.set_xlabel('')
    ax.set_ylim(0,1)
    ax.set_title(title, pad=25)
    
    
    # Here: adding as patch labels the feature means.
    mean_features = {f:'{:.3}'.format(v) for f,v in tmp_df.mean().to_dict().items()}
    
    for patch_label, patch in zip(ax.get_xticklabels(),ax.patches):
        patch_label = patch_label.get_text()
        patch_labelling(ax, patch, mean_features[patch_label], vertical=True,
                        shift=get_shift(patch)
                       )
    sns.despine()
    ax.grid(False)
    
    ax.set_xticklabels([t.get_text().replace('_','\n') for t in ax.get_xticklabels()],
                       rotation=45, ha='right')

    if savefig_file:
        plt.savefig(savefig_file)
        
    if show_plot:
        plt.show()
    else:
        return ax
    


def quality_classification_plots(list_y_true, list_y_pred, class_names,
                                 normalize=False,
                                 cmap=None,
                                 color_class=None,
                                 main_title='',
                                 show_plot=True, savefig_file=None):
    """ Single plot with the Confusion matrix and the quality metrics plot.
    
    In:
        list_y_true (list) : list where each element is an array of 0/1 true classes.
        list_y_pred (list) : list where each element is an array of 0/1 predicted classes.
        class_names (list) : list of classe names to show on the plots.
        main_title (str, default='')
        show_plot (bool, default=False)
        savefig_file (str, default=None)
    """
    
    if len(list_y_true)==1:
        multi=False
        title_cm = 'Confusion matrix'
    else:
        multi=True
        title_cm = 'Average confusion matrix'
        
    fig = plt.figure(figsize=(21,6))
    gs = mpl.gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1/3,2/3])

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])


    mean_cm = np.array([sklearn.metrics.confusion_matrix(y_true, y_pred)
                        for (y_true,y_pred) in zip(list_y_true, list_y_pred)]
                      ).mean(axis=0).astype(int)

    _ = plot_confusion_matrix(mean_cm, class_names,
                              cmap=cmap,
                              normalize=normalize,
                              color_class=color_class,
                                title=title_cm,
                                ax=ax1,
                                show_plot=False)

    _ = plot_classification_evaluation(list_y_true,
                                        list_y_pred,
                                        multi=True,
                                        target_class=1,
                                        title='Quality measures',
                                        ax=ax2,
                                        show_plot=False
                                        )

    plt.suptitle(main_title, fontsize=18, fontweight='bold', y=1.10)
    
    if savefig_file:
        plt.savefig(savefig_file)
        
    if show_plot:
        plt.show()
    else:
        return ax



# FEATURE IMPORTANCE
# -----------------------------------------------------------------------------


def plot_feature_importances(feature_importance_df,
                             order_features=None,
                             nmax=None,
                             savefig_file=None,
                             show_plot=True):
    """ Plot the feature importances (raw and corrected) from the dataframe.
    
    The dataframe should be obtained with the function `ML_tests.feature_importances`.
    """
    # Compare feature importances over all k-fold raw VS corrected
    
    # Sort features by their mean corrected importance
    if not order_features:
        sorted_features = feature_importance_df.loc[feature_importance_df.importance_type=='corrected',:
                                               ].groupby('feature')['value'].mean().sort_values(ascending=False).index.values
    else:
        sorted_features = order_features
    
    # And select if nmax is defined.
    if nmax is not None and nmax > 15:
        feature_names = sorted_features[:nmax]
        width = nmax * 0.8
        
    else:
        feature_names = sorted_features
        width = len(feature_names) * 0.8
        
    
    fig = plt.figure(figsize=(width,10))
    ax1 = fig.add_subplot(1,1,1)
    sns.barplot(
        data=feature_importance_df.loc[feature_importance_df.feature.isin(feature_names),:],
        x='feature',
        y='value',
        hue='importance_type',
        order=feature_names,
        ax=ax1)

    ax1.legend(bbox_to_anchor=(1,1))

    ax1.set_xticklabels(ax1.get_xticklabels(),rotation=45,ha='right')
    ax1.set_xlabel('')
    
    plt.show()



###############################################################################
# PART3 : FEATURE CONTRIBUTIONS AND CLUSTERING
# ============================================

def plot_feature_contributions_to_actual_values(df_contribs,
                                                df_values,
                                                feature,
                                                y_pred,
                                                class_labels,
                                                palette=None,
                                                savefig_file=None,
                                                show_plot=True):
    """ Plot the contribution against the actual value for a feature.
    
    It is possible to display along the dot plot a violin plot of the values
    distributions according to the assigned class, to check that the
    contributions make sense wrt a difference in distributions.
    
    In:
        - df_contribs (pandas.DataFrame): N*F dataframe with contribution
                                          scores of features for each sample.
        - df_values (pandas.DataFrame): N*F dataframe with initial values of
                                        features for each sample.
        - feature (str): name of the feature to plot
        - y_pred (np.array): 0/1 array of N assigned class for the samples.
        - class_labels (list)
        - highlight_sample_idx (int): index from df_values of a sample to
          highlight.
    """

    colors = np.repeat('#424bf4', df_values.shape[0])
    size_marker = np.repeat(mpl.rcParams['lines.markersize']**2,
                            df_values.shape[0])
    df_feat = pd.concat([
                pd.Series(df_contribs[feature].values,name='contribution'),
                pd.Series(df_values[feature].values,name='values')],
                axis=1)
    # Dataframe with feature values and predicted classes.
    df_violin = pd.concat([
                    pd.Series(df_values[feature].values,name='feature'),
                    pd.Series([class_labels[i] for i in y_pred],name='predicted class')
                    ], axis=1)

    fig = plt.figure(figsize=(14,8))
    ax0 = plt.subplot2grid(shape=(1,3),loc=(0,0), colspan=2)
    ax1 = plt.subplot2grid(shape=(1,3),loc=(0,2),colspan=1,sharey=ax0)
    axs = (ax0,ax1)
    #fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
    sns.regplot(data=df_feat,y='values',x='contribution',ax=axs[0],
                fit_reg=False,
                scatter_kws={'facecolors':colors,
                             's':size_marker})

    if palette is None:
        palette = 'Set2'
        
    sns.violinplot(data=df_violin, y = 'feature', x = 'predicted class',
                   order=class_labels,
                   palette=palette,
                   ax=axs[1])
                   

    axs[0].axvline(0,linestyle='--',color='#a3a3a3')
    axs[1].set_ylabel('')
    axs[0].set_ylabel('"{}" values'.format(feature))
    axs[0].set_xlabel(('contributions\n'
                       '(negative: directs to the "{}" class ;\n'
                       'positive: directs to the "{}" class)'
                      ).format(class_labels[0],class_labels[1]))

    fig.suptitle(('Contributions "{}" to the classification of {} samples'
                  '\nalong with the distributions from the two '
                  'predicted classes').format(feature,df_contribs.shape[0]),
                 size=20,
                 y=1.05)
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()


def center_cmap(data, robust=True, center=0):
    """ Return a normalizing function from data values.
    
    This function can be used to reassign correctly values from new data
    basing on the max and min values provided in <data>.
    """
    # Inspired from
    # https://stackoverflow.com/questions/42125433/seaborn-heatmap-get-array-of-color-codes-values
    # And from the seaborn heatmap source code (itself relying on the Axes.pcolormesh code from matplotlib) 

    vmin = np.quantile(data, 0.02) if robust else data.min()
    vmax = np.quantile(data, 0.98) if robust else data.max()

    assert vmin <= center and center <= vmax
    most_distant_absval = max(center - vmin, vmax - center)
    rng = 2 * most_distant_absval

    normlize = mpl.colors.Normalize (center - most_distant_absval,  center + most_distant_absval)

    return normlize


def __DEV_check_cmap_is_centered():
    # Here we use the full dataset of feature contributions to create the normalization function for the colormap
    normlize = center_cmap(data=merged_contribs_clusters.values, robust=True, center=0)

    # now split the dataset into 51 bins (corresponding to the initial N colors of the palette)
    _, bins_feat_contribs = np.histogram(merged_contribs_clusters.values,
                                         bins=51
                                        )

    # For assigning colors to bins, I should be taking the mid-point...
    delta = bins_feat_contribs[1] - bins_feat_contribs[0]


    # And create the array of value to plot (which should represent the range of values.)
    data_plot = np.around(list(bins_feat_contribs+delta/2), 3)
    
    rgba_values = tmp_cmap(normlize(data_plot))
    hex_values = [mpl.colors.rgb2hex(v) for v in rgba_values]
    tmp = pd.DataFrame({'color':hex_values,
                  'bin':data_plot,
                  'y':[1 for _ in range(len(data_plot))]
                 })

    fig = plt.figure(figsize=(20,4))
    ax1 = fig.add_subplot(1,1,1)
    sns.barplot(data=bla,
                x='bin',
                y='y',
                palette=bla.color.values,
                ax=ax1)

    ax1.set_xticklabels(ax1.get_xticklabels(),rotation=45, ha='right')
    ax1.set_title('Demo of the colormap application to the full dataset of feature contributions')
    plt.show()




def plot_accuracy_clusters(df_accuracy_per_label, selected_labels, set_selected_labels, show_plot=True, savefig_file=None):
    """ Plot results from the `select_labels_cluster_analysis` function.
    """
    # Accuracy per cluster
    tmp_acc_perlab = df_accuracy_per_label.melt(id_vars=['cluster'])

    size_clusters = pd.Series(selected_labels
                             ).astype(str).value_counts(
                             ).reset_index().set_axis(['cluster','count'],axis=1,inplace=False)

    order = size_clusters.sort_values(by='count',ascending=False)['cluster'].values


    width = 2.8 * len(set_selected_labels)

    min_width = 8
    width = max(width,min_width)
    fig = plt.figure(figsize=(width,8))

    ax2 = fig.add_subplot(2,1,2)
    ax1 = fig.add_subplot(2,1,1, sharex=ax2)


    sns.barplot(data=tmp_acc_perlab,
                x='cluster',
                y='value',
                order=order,
                hue='variable',
                ax=ax1)

    ax1.legend(bbox_to_anchor=(1,1))
    ax1.set_title("Accuracy of classification per cluster")

    sns.barplot(data=size_clusters,
                x='cluster',
                order=order,
                y='count',
                color='#BBBBBB',
                ax=ax2
               )

    ax2.set_title("Number of samples")

    if savefig_file:
        plt.savefig(savefig_file)
        
    if show_plot:
        plt.show()
    else:
        plt.close()



def barplot_relative_changes(cluster_features_df, order_features, color_feat_map, dict_feature_pvals, ax):
    sns.barplot(data=cluster_features_df.reset_index(),
                x='index',
                y='relative_change',
                order=order_features,
                linewidth=1.3,
                edgecolor='#999999',
                ax=ax)
    
    # Color patches:
    for feature_tick, patch in zip(ax.get_xticklabels(), ax.patches):
        feature = feature_tick.get_text()
        patch.set_facecolor(color_feat_map[feature])

    # And now add the pval labels.
    for feature_tick, patch in zip(ax.get_xticklabels(), ax.patches):
        feature = feature_tick.get_text()
        pval_label = dict_feature_pvals.get(feature,'')
        patch_labelling(ax, patch, pval_label,
                                         vertical=True, textparams={'fontweight':'bold',
                                                                    'fontsize':10})
    ax.set_xlabel('')
    ax.set_ylabel('Relative change')
    ax.set_title('Relative change (in selected vs others)',pad=15)

    # Remove all but the left spine.
    ax.grid(False)
    sns.despine(offset=10, trim=True,
                left=False,
                right=True,
                top=True,
                bottom=True,
                ax=ax)
    
    # Remove the labels as they will appear on the second plot.
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45,ha='right')
    ax.axhline(0,linestyle='-',linewidth=2,color='#CCCCCC')

    return ax



def barplot_normalized_differences(cluster_features_hue_df, order_features, dict_feature_pvals, color_feat_map, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
    
    sns.barplot(data=cluster_features_hue_df.loc[cluster_features_hue_df.feature.isin(order_features),:],
                x='feature',
                y='value',
                order=order_features,
                hue='sample_set',
                palette={'in_cluster':'#FFFFFF','others':'#999999'},
                hue_order=['in_cluster','others'],
                linewidth=1.3,
                edgecolor='#999999',
                ax=ax)
                                                 
    vertical=True

    if vertical:
        get_shift = lambda p: p.get_width() / 2
    else:
        get_shift = lambda p: p.get_height() / 2 

    # And now add the pval labels.
    n_labels = len(ax.get_xticklabels())

    # To be noted that the first half part of patches correspond to the 'selected' cluster,
    # while the rest are patches for the 'others'.
    # The patch needs to be above the highest value among both.
    for (patch_clust, patch_others), group_tick in zip(zip(ax.patches[:n_labels],
                                                           ax.patches[n_labels:]),
                                                       ax.get_xticklabels()):
        # Determine which patch to use
        if vertical:
            y_clust, y_others = (patch_clust.get_height(), patch_others.get_height())

        else:
            y_clust, y_others = (patch_clust.get_width(), patch_others.get_width())

        if all([y_clust<=0, y_others<=0]) or all([y_clust>0, y_others>0]):
            patch_idx = np.argmax(np.abs((y_clust, y_others)))
        else:
            patch_idx = np.argmax((y_clust, y_others))
            
        patch = (patch_clust, patch_others)[patch_idx]

        # Shift needs to be negated if patch_idx==1
        shift = get_shift(patch_clust)
        if patch_idx==1:
            shift = - shift
            
        group = group_tick.get_text()
        pval_label = dict_feature_pvals.get(group,'')
        patch_labelling(ax, patch, pval_label,
                         vertical=True, shift=shift,
                         textparams={'fontweight':'bold',
                                     'fontsize':10})
    
    
    # Color patches
    for feature_tick, patch_clust in zip(ax.get_xticklabels(), ax.patches[:n_labels]):
        patch_clust.set_facecolor(color_feat_map[feature_tick.get_text()])


    ax.grid(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45,ha='right')

    # Remove all but the left spine.
    sns.despine(trim=False,
                left=False,
                right=True,
                top=True,
                bottom=True,
                ax=ax)

    ax.set_ylabel('Value')
    ax.set_title('Barplot of normalized feature differences (in cluster vs others)',pad=15)
    ax.set_xlabel('')
    ax.axhline(0,linestyle='-',linewidth=2,color='#CCCCCC')
    
    ax.legend(loc='upper right')
    
    return ax



def get_data_from_cluster(k_clust,
                      list_relative_changes_clusters,
                      list_pvals_df_clusters,
                      long_merged_contribs_clusters,
                      long_merged_features_clusters,
                      X,
                      labels):
    
    # Feature summarized values
    relative_changes_cluster = list_relative_changes_clusters[k_clust]
    dict_feature_pvals = list_pvals_df_clusters[k_clust].set_index('feature')['pval_label'].to_dict()
    
    
    # This table has N_feature x N_cluster contribution values.
    merged_contribs_clusters_selected = long_merged_contribs_clusters.loc[
                                            long_merged_contribs_clusters['sample_set']=='selected',
                                            :].pivot(columns='cluster',
                                                     index='feature',
                                                     values='value')
    
    # Feature contribution means for the cluster of interest.
    feature_contributions_cluster = merged_contribs_clusters_selected.loc[:,str(k_clust)]
    

    # Data for the confusion matrix.
    cluster_X = X.assign(cluster=labels).groupby('cluster').get_group(str(k_clust))
    
    
    cluster_features_df = pd.concat([
                            relative_changes_cluster.rename('relative_change'),
                            feature_contributions_cluster.rename('feature_contribution')
                            ], sort=True, axis=1)
    
    
    # Order of the columns in the bar plot is defined by the absolute feature contribution:
    order_features = feature_contributions_cluster.abs().sort_values(ascending=False).index.values
    

    # Create normalization function for contribution values
    # basing on the full dataset of feature contributions.
    all_contribs_values = merged_contribs_clusters_selected.values.flatten()
    all_contribs_values.sort()
    
    normlize = center_cmap(all_contribs_values,
                           robust=True,
                           center=0)

    cluster_features_hue_df = long_merged_features_clusters.assign(
                                sample_set=long_merged_features_clusters.sample_set.replace({'selected':'in_cluster'})
                               ).loc[long_merged_features_clusters.cluster==str(k_clust),:]

    return cluster_X, cluster_features_df, cluster_features_hue_df, order_features, dict_feature_pvals, normlize



def add_colorbar(ax, cmap, norm, name_classes, color_class):
    #Here we add a color bar for the interpretation of the barplot colors
    sm = plt.cm.ScalarMappable(cmap=cmap)#, norm=norm)
    sm.set_array([])
    colorbar = plt.colorbar(mappable=sm, ax=ax)
    colorbar.set_ticks([0,1])

    tick_names = name_classes if color_class==1 else name_classes[::-1]
    colorbar.ax.set_yticklabels(tick_names)
    
    colorbar.ax.tick_params(color='#FFFFFF', pad=0)
    colorbar.set_label('Feature\nContribution', rotation=0, ha='center',va='center')
    



def plot_summary_cluster_features(k_clust,
                                  list_relative_changes_clusters,
                                  list_pvals_df_clusters,
                                  long_merged_contribs_clusters,
                                  long_merged_features_clusters,
                                  merged_kfolds_df,
                                  labels,
                                  name_classes, # list of class names
                                  color_class=1, # index of the class of interest.
                                  show_majority_class_labelling=True,
                                  color_features=None,
                                  limit_N_features=False, # Can be an int if needed.
                                  rename_cols=None,
                                  show_plot=True,
                                  savefig_file=None                                  
                                 ):
    """ Summary plot of the quality and feature properties of cluster k.
    
    This plot should be used after the creation of the feature contributions, and the clusters. It integrates
    all the information of summarized differences of each cluster against others into a multi-plot figure.
    
    Plots are:
        - confusion matrix showing the counts of samples per class
        - quality of the cluster (through different metrics)
        - barplots showing the differences between the cluster and other samples.
        - confusion matrix and quality of the cluster *after* labelling all samples according to the majority class.
        
        
    
    In:
        - k_clust (int): the index of the cluster to use. Transforming to str should give the name of the cluster
        
        - list_relative_changes_clusters_normalized (list):  list of relative changes for all cluster.
                                                     Should be produced with the normalize=True option
                                                     of `ML_tests.multiple_relative_changes`
                                                     
        - list_pvals_df_clusters (list): list of dataframes of pvalues associated to each features
                                         (obtained with `ML_tests.multiple_relative_changes`)
                                         
        - long_merged_contribs_clusters (pandas.DataFrame): long-format dataframe of
                                                            all mean feature contributions.
        
        - merged_kfolds_df (pandas.DataFrame): dataframe of samples with features, and with annotations
                                               (columns such as 'true_y', 'predicted_y', etc.)
        
        - name_classes (list): list of class names

        - color_class (int): value indicating the target class (should be 0 or 1, default=1)
        
        - color_features (dict, default=None)
        
        - limit_N_features (int, default=None): maximum number of features to show.
        
        - show_plot (bool, default=True)
        
        - savefig_file (str, default=None): path to file to save the figure to.
        
    """
    

    # ==============================================================================================
    
    (cluster_X,
     cluster_features_df,
     cluster_features_hue_df,
     order_features,
     dict_feature_pvals,
     normlize) = get_data_from_cluster(k_clust,
                      list_relative_changes_clusters,
                      list_pvals_df_clusters,
                      long_merged_contribs_clusters,
                      long_merged_features_clusters,
                      merged_kfolds_df,
                      labels)
    
    
    if limit_N_features is not False:
        assert limit_N_features>0 and limit_N_features < len(order_features), "Error for <limit_N_features>"
        order_features = order_features[:limit_N_features]

    # Reorder the table and select features.
    cluster_features_df = cluster_features_df.loc[order_features,:]
    cluster_features_hue_df = cluster_features_hue_df.loc[cluster_features_hue_df.feature.isin(order_features),:]
    
    # ==============================================================================================
    # COLORS
    
    # 2 classes palette for the confusion matrix.
    palette_2cl = sns.diverging_palette(160,0,n=51, as_cmap=False)

    # This one below is used to produce the color bar
    tmp_cmap = mpl.colors.ListedColormap(sns.color_palette(palette_2cl).as_hex())

    # Get the colors associated to the feature contributions of the considered cluster.
    normalized_feature_contribs = normlize(cluster_features_df.feature_contribution.values)

    rgba_values = tmp_cmap(normalized_feature_contribs)
    hex_values = [mpl.colors.rgb2hex(v) for v in rgba_values]
    color_feat_map = pd.Series(hex_values, index=cluster_features_df.index.values
                              ).to_dict()

    def add_colorbar(ax, cmap, norm, name_classes, color_class):
        #Here we add a color bar for the interpretation of the barplot colors
        sm = plt.cm.ScalarMappable(cmap=cmap)#, norm=norm)
        sm.set_array([])
        colorbar = plt.colorbar(mappable=sm, ax=ax)
        colorbar.set_ticks([0,1])

        tick_names = name_classes if color_class==1 else name_classes[::-1]
        colorbar.ax.set_yticklabels(tick_names)
        
        colorbar.ax.tick_params(color='#FFFFFF', pad=0)
        colorbar.set_label('Feature\nContribution', rotation=0, ha='center',va='center')
    
    # ==============================================================================================
    
    if color_class==1:
        color_positives, color_negatives = hex_values[0], hex_values[-1]
        
    elif color_class==0:
        color_positives, color_negatives = hex_values[-1], hex_values[0]
        
        
    # First figure: quality of the cluster.
    
    #NOTE: might need to add a threshold, and use the prediction scores, rather than directly the predicted_class
    # at 0.5 default threshold.

        
    quality_classification_plots([cluster_X.true_class], [cluster_X.predicted_class],
                                 cmap = palette_2cl,
                                 color_class=1,
                                class_names=name_classes,
                                main_title="Evaluation of the content of the cluster {}".format(k_clust))
    

    if show_majority_class_labelling:
        # Now : quality of the cluster if all samples are set to majority class.
        labels_majority_classes = label_majority_class_assignment(cluster_X, labels=[str(k_clust) for _ in range(cluster_X.shape[0])])
        
        quality_classification_plots([cluster_X.true_class], [labels_majority_classes],
                                     cmap = palette_2cl,
                                     color_class=1,
                                    class_names=name_classes,
                                    main_title="Content of the cluster {} with majority class labelling of samples.".format(k_clust))
    
    
    # And now: features contributions and values.
    

    # Relative change colored barplot
    # -------------------------------
    fig = plt.figure(figsize=(10+0.2*cluster_features_df.shape[0],9))
    ax1 = fig.add_subplot(1,1,1)
    ax1 = barplot_relative_changes(cluster_features_df, order_features,
                                   color_feat_map, dict_feature_pvals, ax1)

    add_colorbar(ax1, tmp_cmap, normlize, name_classes, color_class)
    plt.show()
    
    
    # Colored barplot of features normalized values (cluster vs others)
    # -----------------------------------------------------------------
    fig = plt.figure(figsize=(10+0.2*cluster_features_hue_df.shape[0],9))
    ax2 = fig.add_subplot(1,1,1)
    ax2 = barplot_normalized_differences(cluster_features_hue_df,
                                         order_features, dict_feature_pvals, 
                                         color_feat_map,
                                         ax2)
    
    add_colorbar(ax2, tmp_cmap, normlize, name_classes, color_class)
    
    plt.tight_layout()
    plt.show()
