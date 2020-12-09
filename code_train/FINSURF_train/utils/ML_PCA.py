#! /usr/bin/env python
# -*- coding:utf-8 -*-

###############################################################################
# IMPORTS

import pandas as pd
from sklearn.decomposition import PCA


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import mpl_toolkits.mplot3d.axes3d


mpl.rcParams.update({'figure.autolayout':True})

from IPython.display import display

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


def dataframe_rotations(pca,feature_names, PC_names=None):
    """ Return the maximum variance of each feature over the PCs.
    """
    if PC_names is None:
        PC_names = ["PC{}".format(i) for i in range(len(pca.components_))]
    pca_rotations = pd.DataFrame(pca.components_,
                                 index=PC_names,
                                 columns=feature_names) 
    return pca_rotations



def plot_cumulative_variance_pca(pca, nmax=None, show_plot=True, savefig_file=None):
    """ From a PCA object, plot the cumulative variance from each of the PC.

    This can be used to decide how many components to keep while still
    accurately describing your data.

    """
    tmp = pd.DataFrame([pca.explained_variance_ratio_.cumsum(),
                        range(1, len(pca.explained_variance_ratio_)+1)
                        ]).T
    tmp.columns = ['cumulative variance\npercentage','N PC']

    tmp['N PC'] = tmp['N PC'].astype(int)

    if nmax is None:
        nmax = tmp.shape[0]

    fig = plt.figure(figsize=(15,6))
    ax1 = fig.add_subplot(1,1,1)
    sns.pointplot(data=tmp.head(nmax),
                  x=tmp.columns[1],
                  y=tmp.columns[0],
                  ax=ax1)
    ax1.set_title(('Cumulative distribution\nof the percentage of '
                   'explained variance'))
    if show_plot:
        plt.show()
    
    if savefig_file is not None:
        plt.savefig(savefig_file)


def feature_map_factorplot(pca, features, PC_X='PC0', PC_Y='PC1'):
    """ 2D plot of features projected on the requested PCA dimensions.
    
    Classic R pca representation of the feature spaces projected on the 2D selected PCA components.
    
    
    In:
        pca (sklearn PCA object)
        features (array): list of feature names
        PC_X ([str],default:PC0)
        PC_X ([str],default:PC1)
    """
    PC_names = ['PC{}'.format(i) for i in range(len(pca.components_))]
    assert PC_X in PC_names, print("Error: PC_X '{}' not in PC_names".format(PC_X))
    assert PC_X in PC_names, print("Error: PC_X '{}' not in PC_names".format(PC_X))
    
    pca_rotations = dataframe_rotations(pca, features, PC_names)
    PC_variance_ratio = pd.Series(pca.explained_variance_ratio_,index=PC_names)

    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_xlim(-1.1,1.1)
    ax1.set_ylim(-1.1,1.1)

    PC_axes = [PC_X, PC_Y]
    for feature, feature_pc in pca_rotations.loc[PC_axes,:].T.iterrows():
        if any(feature_pc.abs()>0.05):
            arr = ax1.arrow(0,0,feature_pc[PC_axes[0]], feature_pc[PC_axes[1]],
                            width=0.008)
            ax1.add_patch(arr)

            plt.text(feature_pc[PC_axes[0]], feature_pc[PC_axes[1]], feature,ha='center')

    ax1.set_xlabel("{} ({:.1f}% variance)".format(PC_axes[0],100*PC_variance_ratio[PC_axes[0]]))
    ax1.set_ylabel("{} ({:.1f}% variance)".format(PC_axes[1],100*PC_variance_ratio[PC_axes[1]]))

    ax1.axhline(0, color='#AAAAAA', linestyle='--', linewidth=1.5)
    ax1.axvline(0, color='#AAAAAA', linestyle='--', linewidth=1.5)
    circ = mpl.patches.Circle((0, 0), 1, color='#666666', fill=False)
    ax1.add_patch(circ)

    ax1.grid(False)

    plt.show()
    
      
def heatmap_PCA_features_to_PC(pca, features,
                               N_max_PC = None,
                               title=None,
                               ax=None,
                               savefig_file=None,
                               show_plot=True
                               ):
    """ Heatmap of the feature variance ratio associated to PCA components.
    
    In:
        pca (sklearn PCA object)
        features (array)
    """
    PC_names = ['PC{}'.format(i) for i in range(len(pca.components_))]
    
    PC_variance_ratio = pd.Series(pca.explained_variance_ratio_,index=PC_names)
    pca_rotations = dataframe_rotations(pca, features)

    if not N_max_PC:
        N_max_PC = len(pca.components_)

    height=0.8* N_max_PC
    height=max(45, height)
    
    fig = plt.figure(figsize=(height,height))
    ax1 = fig.add_subplot(1,1,1)
    sns.heatmap(pca_rotations.loc[PC_names[:N_max_PC],:], cmap="PiYG",
                cbar_kws={"shrink": .50},
                center=0,
                linewidth=1.2,
                square=True,
                ax=ax1)

    ax1.set_yticklabels(ax1.get_yticklabels(),rotation=0,ha='right')
    ax1.set_xticklabels(ax1.get_xticklabels(),rotation=45,ha='right')

    if title is None:
        title=("Principal axes in feature space, representing\nthe directions "
               "of maximum variance in the data")

    ax1.set_title(title)

    if savefig_file:
        plt.savefig(savefig_file)
    if show_plot:
        plt.show()




def plot_samples_PCA_2d(pca, X, labels=None, PC_X='PC0', PC_Y='PC1',
                        colors_labels=None,
                        prefix_label='',
                        title='',
                        ax=None,
                        show_plot=True,
                        savefig_file=None):
    """ 2D projection of the samples in X (colored by y).
    """
    # PCA plot on 2D Principal Components to show the seprability of samples.
    PC_names = ['PC{}'.format(i) for i in range(len(pca.components_))]
    
    PC_variance_ratio = pd.Series(pca.explained_variance_ratio_,index=PC_names)

    
    pca_transformed_df = pd.DataFrame(pca.transform(X),
                                      columns=PC_names
                                     )
    
    if labels is None:
        labels = [0 for _ in range(pca_transformed_df.shape[0])]
    
    PC_axes = [PC_X, PC_Y]
    pca_df_plot = pca_transformed_df.loc[:,PC_axes].assign(label = labels)
    
    if not colors_labels:
        colors_labels = sns.color_palette("Paired", len(labels))
        colors_labels = dict(zip(set(labels), colors_labels))


    if ax is None:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1,1,1)
    
    # plot for each class
    for (group_name, group) in pca_df_plot.groupby('label'):
        color = colors_labels[group_name]
        ax.scatter(group.loc[:,group.columns[0]],
                    group.loc[:,group.columns[1]],
                    color=color,
                    alpha=0.6,
                    edgecolor='white',
                    label=prefix_label+'{}'.format(group_name))

    ax.set_xlabel("{} ({:.1f}% variance)".format(PC_axes[0],100*PC_variance_ratio[PC_axes[0]]), labelpad=20)
    ax.set_ylabel("{} ({:.1f}% variance)".format(PC_axes[1],100*PC_variance_ratio[PC_axes[1]]), labelpad=20)


    ax.legend(bbox_to_anchor=(1,1))
    ax.set_title(title)

    if savefig_file:
        plt.savefig(savefig_file)
    
    if show_plot:
        plt.show()
    else:
        return ax
    



def plot_samples_PCA_3d(pca, X, labels=None, PC_X='PC0', PC_Y='PC1', PC_Z='PC2', 
                        colors_labels = None,
                        prefix_label='',
                        title='',
                        ax=None,
                        show_plot=True,
                        savefig_file=None):
    """ 3D projection of the samples in X (colored by y).
    """
        
    # PCA plot on 3D Principal Components to show the seprability of samples.
    PC_names = ['PC{}'.format(i) for i in range(len(pca.components_))]
    
    PC_variance_ratio = pd.Series(pca.explained_variance_ratio_,index=PC_names)

    
    pca_transformed_df = pd.DataFrame(pca.transform(X),
                                      columns=PC_names
                                     )
   
    if labels is None:
        labels = [0 for _ in range(pca_transformed_df.shape[0])]


    PC_axes = (PC_X, PC_Y, PC_Z)
    pca_df_plot = pca_transformed_df.loc[:,PC_axes].assign(label=labels)
    
    if not colors_labels:
        colors_labels = sns.color_palette("Paired", len(labels))
        colors_labels = dict(zip(set(labels), colors_labels))
        

    if ax is None:
        fig = plt.figure(figsize=(10,10))
        ax = Axes3D(fig)
        
    else:
        assert isinstance(ax, mpl_toolkits.mplot3d.axes3d.Axes3D), "'ax' is not a Axes3D instance"
    
    # plot for each class
    for group_name, group in pca_df_plot.groupby('label'):
        color = colors_labels[group_name]
        ax.scatter(group.loc[:,group.columns[0]],
                   group.loc[:,group.columns[1]],
                   group.loc[:,group.columns[2]],
                   color=color,
                   alpha=0.6,
                   edgecolor='white',
                   label=prefix_label+'{}'.format(group_name))

    ax.set_xlabel("{} ({:.1f}% variance)".format(PC_axes[0],100*PC_variance_ratio[PC_axes[0]]), labelpad=20)
    ax.set_ylabel("{} ({:.1f}% variance)".format(PC_axes[1],100*PC_variance_ratio[PC_axes[1]]), labelpad=20)
    ax.set_zlabel("{} ({:.1f}% variance)".format(PC_axes[2],100*PC_variance_ratio[PC_axes[1]]), labelpad=20)


    ax.legend(bbox_to_anchor=(0,1))
    
    ax.set_title(title)
    
    if savefig_file:
        plt.savefig(savefig_file)

    if show_plot:
        plt.show()
    else:
        return ax
