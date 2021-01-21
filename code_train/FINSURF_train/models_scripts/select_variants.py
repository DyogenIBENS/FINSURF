#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
The goal of this library / program is to provide with the ability to select
variant on the basis of their genomic location.

The script should be provided with different sets of variants (along with their
assigned class). The target class should be defined. From variants of this
target class, other variants will be selected on the basis of a provided
criterion, which can be distance to the target class, matched proportions of
the target class for a given feature, etc.


Provided sets may be provided as a single file, or as multiple files. The
multiple files will be identifiable through a '{}' element in the filename,
which will be formatted to the different chromosomes in the program.
"""


# NOTE: maybe drop_duplicates() on rowids, to avoid an over-representation of
# some biotypes when considering "big" INDELS. This is not so crucial.


###############################################################################
# IMPORTS

import collections
import datetime
import glob
import gzip
import importlib
import multiprocessing
import os
import sys

import argparse as ap
import itertools as itt
import numpy as np
import pandas as pd


# BedTools and tmpdir
import tempfile
import pybedtools as pbt


# EXTERNAL LIBRARIES
sys.path.insert(0,'../utils/')

import dataframes

###############################################################################
# DEFINITIONS


def memory_aware_loading(filepath, rowid_filter=None,
                         row_filter_method='row_id',
                         usecols=None
                         ):
    """ Try to load only the rows and columns of interest from the filepath, 

    CAUTION: "row_filter" is a "keep" filter, not a "filter-out".

    NOTE: here the function requires that the file contains a header, as I am
    reading only the "row_id" column ... this can be generalized later, in
    which case the skiprows part should be corrected : I need to increase all
    indices by 1 since the 0th line is the header line in the file (the
    0-based indexing in the read dataframe is artificially modified by the
    pd.read_table function).

    In:
        filepath (str): path to the variants to load.
        rowid_filter (list): list of indexes to *keep* ; can be integers (row
                             indices) or str ('{chrom}_{index}' ; second case
                             by default.)
        row_filter_method (str, default='row_id'): which method to apply for
                                                   filtering rows. Any other
                                                   parameter than 'row_id' will
                                                   result in index-based
                                                   filtering. 
        usecols (list, default=None): list of columns to use ;


    Return:
        pandas.DataFrame

    """
    header=True

    if not os.path.exists(filepath):
        print("Warning: '{}' not found".format(filepath))
        return None

    if row_filter_method is not None:
        assert row_filter_method in ['row_id','index'], \
                ('<row_filter> should be a method among (row_id, index)\n'
                 '(detected : "{}")').format(row_filter_method)
                                            
    if not usecols:
        usecols = list(pd.read_table(filepath,nrows=3).columns)

    if rowid_filter:
        row_ids_file = pd.read_table(filepath, usecols=['row_id'])

        if row_filter_method == 'row_id':
            # Then we get the indices of rows we want to keep.
            rownumber_list = row_ids_file.loc[row_ids_file.row_id.isin(rowid_filter),
                                              :].index.values
        else:
            rownumber_list = np.array(rowid_filter).astype(int)

        # The read_table function has the argument 'skiprows'. We will create the
        # list of row ids to remove.
        row_count = row_ids_file.shape[0]

        if header:
            skiplist = set(range(1, row_count+1)) - set(rownumber_list+1)

        else:
            # CAUTION : this should not happen yet... the function would need
            # to have an argument "column names" in addition to a
            # "header=Bool()" argument.
            skiplist = set(range(0, row_count)) - set(rownumber_list)

    else:
        skiplist = []

    df = pd.read_table(filepath, skiprows=skiplist, usecols=usecols)

    if rowid_filter:
        df.index = rownumber_list

    return df



def distance_matching(ref_df, other_df, distance):
    """ Report variants from <other_df> that are close to <ref_df> variants.

    Create a bedtool of regions from (+) samples, merge them, and intersect
    (-) positions with those regions.
    Keep only (-) samples within regions.
    Keep only (+) samples with associated (-).

    Return in addition to sampled variants a table matching for each positive
    the set of associated negatives.
    """

    print(("Match on distance to positives. Max distance required: {:,} bp"
          ).format(distance))

    ref_regions = pd.concat([ref_df.chrom,
                             ref_df.end-1,
                             ref_df.end],
                            axis=1
                            ).set_axis(['chrom','start','end'],axis=1,inplace=False)

    ref_regions = ref_regions.assign(name=ref_df.index.values)

    # Create a BedTools and merge the regions.
    # Original index is merged into a single column.

    ref_regions_bt = pbt.BedTool.from_dataframe(
                        ref_regions.sort_values(by=['chrom','start'])
                        ).slop(b=distance, g=genome_size
                              ).merge(c=4, o='collapse', delim=';')


    # Full intersect: variants not within regions are still reported.
    other_df_bt = pbt.BedTool.from_dataframe(other_df.iloc[:,[0,1,2]])

    intersect_bt = other_df_bt.intersect(ref_regions_bt, wao=True)

    other_intersect_ref_regions = intersect_bt.to_dataframe() 

    other_intersect_ref_regions.index = other_df.index.values

    # Retrieve 'other' elements which are in 'ref' regions.
    other_df_closeby_ref = other_df.loc[(other_intersect_ref_regions.name!='.').values,:]


    # Now create a table of refs and their associated 'other' variants. 
    # This is done by creating a dict mapping the merged-idx from ref-regions
    # to other-idx ; the ref-idx are then spread to their own keys.
    non_empty_other_intersect_ref_regions = other_intersect_ref_regions.reset_index().loc[
                    (other_intersect_ref_regions.name!='.').values,
                    ['index','thickStart']]

    map_merged_refidx_to_otheridx = collections.defaultdict(list)
    _ = {map_merged_refidx_to_otheridx[ref_idx].append(int(oth_idx))
            for oth_idx, ref_idx in non_empty_other_intersect_ref_regions.values}
    
    map_refidx_to_otheridx = {int(k):other_idx_lst
                         for ref_idx, other_idx_lst in map_merged_refidx_to_otheridx.items()
                         for k in ref_idx.split(';')
                             }

    # Now we can keep only the (+) which were associated to one or more (-)
    ref_with_other = ref_df.loc[list(map_refidx_to_otheridx.keys()),:]
    
    sampled_ref_df = ref_with_other
    sampled_other_df = other_df_closeby_ref


    # Remove the temporary bedfiles to free space in the tmp dir.
    if os.path.exists(ref_regions_bt._tmp()):
        os.remove(ref_regions_bt._tmp())

    if os.path.exists(other_df_bt._tmp()):
        os.remove(other_df_bt._tmp())

    if os.path.exists(intersect_bt._tmp()):
        os.remove(intersect_bt._tmp())

    # Return the sampled ref_df, sampled_other_df, and the table mapping ref
    # samples to their matched other samples.
    return sampled_ref_df, sampled_other_df, non_empty_other_intersect_ref_regions 


def downsampling(ref_df, other_df):
    """ Resampling (+) and (-) to have matched numbers.
    This function was created for the "distance match" scenario, as we might be
    short in others because of short distances.  We want comparable sizes.

    We accept a difference of +- 10% of (+)
    Note: this might impact the distances of (-) to (+) (ie a positive which
    had an associated negative might be removed, while the negative is kept.)
    """
    if np.isclose(ref_within_regions_with_other.shape[0],
                  other_df_closeby_ref.shape[0],
                  atol=ref_within_regions_with_other.shape[0]*0.1):

        print("Sizes are close: ref {:,} vs {:,}".format(ref_df, other_df))

        sampled_ref_df = positive_within_regions_with_other
        sampled_other_df = other_df_closeby_ref

    
    else:
        print("Distance match: resampling datasets to match sizes.")
        if other_df_closeby_ref.shape[0] > positive_within_regions_with_other.shape[0]:
            print("\tMore (-) than (+)")
            sampled_other_df = other_df_closeby_ref.sample(positive_within_regions_with_other.shape[0])
            sampled_ref_df = positive_within_regions_with_other

        else:
            print("\tMore (+) than (-)")
            sampled_ref_df = positive_within_regions_with_other.sample(other_df_closeby_positive.shape[0])
            sampled_other_df = other_df_closeby_ref

    return sampled_ref_df, sampled_other_df


def sample_variants(ref_df, other_df, match_type, matchtype_variables):
    """ sample variants from <other_df> according to variants in <ref_df>

    <match_type> will define whether the sampling should be done on the
    proportions of <ref_df> variants in GENCODE biotypes (gencodeMatch), or on
    the distance to these variants.
    """
    if ref_df is None or other_df is None:
        return None, None, None

    # Note: before anything: filter out coding variants.

    if match_type == 'gencodeMatch':
        other_biotypes = other_df['biotypes.best_biotype'
                         ].replace(np.nan,'biotype:intergenic'
                                 ).apply(lambda v:
                                        dataframes.get_field_kv_pairs_list(v,',',':','biotype')
                                        )
    else:
        other_biotypes = None

    matchtype_variables['other_biotypes'] = other_biotypes

    (sampled_ref_df,
     sampled_other_df,
     non_empty_other_intersect_ref_regions) = sample_controls(
                                                ref_df,
                                                other_df,
                                                match_type,
                                                matchtype_variables
                                                )

    # coding filtering is quite long so better to do it after the sampling
    # than before.
    coding_biotypes = ['biotype:start_codon', 'biotype:CDS',
                       'biotype:Selenocysteine',
                       'biotype:exon',
                       'biotype:stop_codon', 'biotype:splice_site']

    ref_coding = sampled_ref_df['biotypes.best_biotype' 
                           ].replace(np.nan,'biotype:intergenic'
                                   ).str.contains('|'.join(coding_biotypes))

    other_coding = sampled_other_df['biotypes.best_biotype' 
                           ].replace(np.nan,'biotype:intergenic'
                                   ).str.contains('|'.join(coding_biotypes))

    sampled_ref_df = sampled_ref_df.loc[~ref_coding.values,:]
    sampled_other_df = sampled_other_df.loc[~other_coding.values,:]


    if match_type == 'distanceMatch':
        # Recreate the map of ref idx to other idx lists, but filtering
        # non-coding variants.

        map_merged_refidx_to_otheridx = collections.defaultdict(list)

        # Here the "other" variants that are coding are filtered out.
        _ = {map_merged_refidx_to_otheridx[ref_idx].append(int(oth_idx))
            for oth_idx, ref_idx in non_empty_other_intersect_ref_regions.loc[~other_coding.values,:].values}
        
        # This can mean that some "ref" variants, which were only associated to
        # coding "other" variants, wont make it in the dict below.
        map_refidx_to_otheridx = {int(k):other_idx_lst
                             for ref_idx, other_idx_lst in map_merged_refidx_to_otheridx.items()
                             for k in ref_idx.split(';')
                                 }

        # Here the 'ref' variants that were coding are removed.
        new_map_refidx_to_otheridx = {k:map_refidx_to_otheridx[k]
                                         for k in sampled_ref_df.index.values
                                         if k in map_refidx_to_otheridx
                                      }

        # But we still need to remove the ref samples which were associated
        # only to coding "other" samples.
        sampled_ref_df = sampled_ref_df.loc[list(new_map_refidx_to_otheridx.keys()),:]

    else:
        new_map_refidx_to_otheridx = None
        
    ref_indices = sampled_ref_df.index.values
    other_indices = sampled_other_df.index.values

    return ref_indices, other_indices, new_map_refidx_to_otheridx


# Main function for sampling.
def sample_controls(ref_df, other_df, match_type, matchtype_variables):
    """
    """
    # if ratio is not None:
    #     assert ratio <=0.5, ("Error: the reference set should always be the least "
    #                          "represented class.")
        
    if match_type == 'noMatch':
        sampled_ref_df = ref_df
        sampled_other_df = other_df

    elif match_type == 'gencodeMatch':
        # CAUTION : here ref biotypes might correspond to the global dataframe
        # biotypes (while the function is called for each chromosome, and the
        # "other_biotypes" is calculated in the function `sample_variants` for
        # the current dataframe.
        # The "ref_biotypes" will be used as our "target values", while the
        # "current_ref_biotypes" will be used for filtering the dataframe.
        current_ref_biotypes = ref_df['biotypes.best_biotype'
                                     ].replace(np.nan,'biotype:intergenic'
                                     ).apply(lambda v:
                                        dataframes.get_field_kv_pairs_list(v,',',':','biotype')
                                        )

        ref_biotypes = matchtype_variables['ref_biotypes']
        #ref_biotype_props = matchtype_variables['ref_biotype_props']
        other_biotypes = matchtype_variables['other_biotypes']

        assert (ref_biotypes is not None), "<ref_biotypes> is required."
        assert (other_biotypes is not None), "<other_biotypes> is required."
        print("Match on GENCODE repartition of ref sample biotypes.")
        
        # Doing a selection
        # First we need to identify categories that may be missing from REF or
        # OTHER ; new proportions for categories will be calculated without
        # these missing categories. Sampling will be based on these new
        # proportions.

        cat_to_exclude = []
        for c in list(ref_biotypes.index.values)+list(other_biotypes.unique()):
            if c not in ref_biotypes.index.values or c not in other_biotypes.unique():
                cat_to_exclude.append(c)

        cat_to_exclude = set(cat_to_exclude)

        # Now calculate the proportions of the classes in each set.
        tmp_ref_biotypes = ref_biotypes.loc[[v for v in ref_biotypes.index.values
                                             if v not in cat_to_exclude]]
        tmp_oth_biotypes = other_biotypes.loc[~other_biotypes.isin(cat_to_exclude)]

        prop_ref = tmp_ref_biotypes.div(tmp_ref_biotypes.sum())
        prop_other = tmp_oth_biotypes.value_counts().div(tmp_oth_biotypes.shape[0])
            
        # And now we will go through each category to sample variants from the
        # OTHER dataframe, according to the proportions in the REF.
        # The order of categories is defined by the lowest to greatest ratio of
        # prop_OTHER / prop_REF.

        ratio_props = prop_other.div(prop_ref)
        
        sampled_ref = []
        sampled_other = []
        for i, (cat, ratio_prop_cat) in enumerate(ratio_props.sort_values().items()):
            if i == 0:
                if ratio_prop_cat < 1:
                    # This means that the category is under-represented in the
                    # OTHER dataframe ; we retrieve all samples from that
                    # category, which will define the total number of samples
                    # to sample.
                    sampled_ref.append(ref_df.loc[current_ref_biotypes==cat,:])
                    sampled_other.append(other_df.loc[other_biotypes==cat,:])
                else:
                    N_to_sample = prop_ref[cat] * prop_other.shape[0]

                    sampled_ref.append(ref_df.loc[current_ref_biotypes==cat,:])
                    sampled_other.append(other_df.loc[other_biotypes==cat,:].sample(N_to_sample))

                N_other_to_sample = int(other_df.loc[other_biotypes==cat,:].shape[0] / prop_ref[cat])

            else:
                sampled_ref.append(ref_df.loc[current_ref_biotypes==cat,:])

                N_to_sample = int(N_other_to_sample * prop_ref[cat])
                sampled_other.append(other_df.loc[other_biotypes==cat,:].sample(N_to_sample))


        sampled_ref_df = pd.concat(sampled_ref)
        sampled_other_df = pd.concat(sampled_other)

        # sampled_other_df = pd.concat(selected_other)
        # sampled_ref_df = ref_df

        # --------------------------------------------------------------------------
        # OLD PART

        # OLD PART BELLOW : I was wrongly taking into account the differences
        # in the frequencies between REF and OTHER.

        # sorted_ref_biotypes_props = sorted(ref_biotype_props.items(), key=lambda v: v[1])

        # for biotype, prop in sorted_ref_biotypes_props:
        #     sub_other_df = other_df.loc[(other_biotypes==biotype).values,:]

        #     # Here the number of 'other' to be sampled is defined as the
        #     # total size of the 'other' set, multiplied by the proportion of
        #     # samples that are of specific biotype in the 'ref' samples set.
        #     count = int(prop * other_df.shape[0])

        #     if sub_other_df.shape[0] > count:
        #         selected_other.append(sub_other_df.sample(count))

        #     else:
        #         selected_other.append(sub_other_df)

        # sampled_other_df = pd.concat(selected_other)
        # sampled_ref_df = ref_df

        # Now, try to correct sampled proportions
        # Check which proportions might be corrected :
        # - those that show strong difference.
        # - and which difference arises from a higher "OTH" proportion
        # (which can be corrected)
        # for i in range(10):
        #     sampled_other_biotypes = sampled_other_df['biotypes.best_biotype'].apply(
        #                     lambda v: dataframes.get_field_kv_pairs_list(v,',',':','biotype')
        #                     )
        #     props_df = pd.concat([sampled_other_biotypes.value_counts().div(sampled_other_df.shape[0]).rename('OTH'),
        #                           ref_biotype_props.rename('REF')],
        #                         axis=1, sort=True).dropna()

        #     props_df = pd.concat([
        #                     props_df,
        #                     props_df.apply(
        #                         lambda row: (~np.isclose(row['OTH'],row['REF'], rtol=1e-2, atol=1e-1)) &\
        #                                     (row['OTH'] > row['REF'])
        #                                     , axis=1).rename('props_to_correct')
        #                     ],axis=1)
        #     
        #     if not any(props_df['props_to_correct']):
        #         break

        #     selected_other = []
        #     for biotype, row in props_df.iterrows():
        #         if row['props_to_correct'] == False:
        #             selected_other.append(sampled_other_df.loc[sampled_other_biotypes==biotype,:])
        #     
        #         else:
        #             frac = row['REF'] / row['OTH']
        #             selected_other.append(sampled_other_df.loc[sampled_other_biotypes==biotype,:
        #                                                       ].sample(frac=frac))

        #     sampled_other_df = pd.concat(selected_other)
        # 
        # END OF OLD PART
        # --------------------------------------------------------------------------


        non_empty_other_intersect_ref_regions = None


    elif match_type == 'cytobandMatch':
        cytoband_ref = matchtype_variables['cytoband_ref']

        selected_other = []
        for chrom, cyto in cytoband_ref.items():
            tmp = other_df.loc[other_df['chrom']==chrom]
            selected_other.append(tmp.loc[tmp['cytoband'].isin(cyto),:])

        sampled_ref_df = ref_df
        sampled_other_df = pd.concat(selected_other)
        non_empty_other_intersect_ref_regions = None


    elif match_type == 'distanceMatch':
        (sampled_ref_df,
        sampled_other_df,
        non_empty_other_intersect_ref_regions)  = distance_matching(ref_df, other_df,
                                                                    matchtype_variables['distance'])


    return (sampled_ref_df, sampled_other_df, non_empty_other_intersect_ref_regions)


def create_parser():
    parser = ap.ArgumentParser(epilog=__doc__,
                               formatter_class=ap.RawDescriptionHelpFormatter)


    parser.add_argument('--numvar_paths',
                        help=('path(s) to the different sets of variants.\n'
                              'There should be at least two paths. The paths '
                              'can either be a "direct" path, or an '
                              '"unformatted" path, containing "{}". \n'
                              'These unformatted paths will be filled with '
                              'the chromosomes formatted as "chr1","chr2",'
                              'etc.\n'
                              'These files should contain the columns:\n'
                              '\tchrom,start,end,row_id,biotypes.best_biotype'
                              ),
                        required=True,
                        nargs='+',
                       )

    parser.add_argument('--names',
                        help=(''),
                        required=True,
                        nargs='+'
                       )

    parser.add_argument('--rowid_filters_paths',
                        help=('paths to files containing row_ids to keep '
                              'for each <numvar_path>\n'
                              'These can be empty strings: "", but still '
                              'should be provided.'
                              ),
                        nargs='+'
                       )


    parser.add_argument('--filter_types',
                        help=('filtering types to apply for the '
                              '<rowid_filters_paths> (one for each)\n'
                              'It should be "row_id" or "index", and '
                              ' can be empty strings: "", for files '
                              'which do not require filtering.'
                             ),
                        nargs='+'
                       )

    parser.add_argument('--ref',
                        help=('Indicate which name among <names> is the '
                              'reference set. These variants will be used as '
                              'reference to match variants from other sets.'
                              ),
                        required=True,
                        type=str
                       )

    parser.add_argument('-m',
                        '--match_type',
                        help='',
                        choices=['noMatch','gencodeMatch','cytobandMatch','distanceMatch'],
                        required=True
                       )

    parser.add_argument('-d',
                        '--distance',
                        help=('Max distance of ClinVar controls to eQTLs'
                              '(default=1000bp)'),
                        type=int,
                        required=False,
                        default=1000
                       )


    parser.add_argument('-o',
                        '--outputdir',
                        help='',
                        required=True
                       )

    parser.add_argument('--ncores',
                        help='',
                        required=True,
                        type=int
                       )

    parser.add_argument('--temp_path_dir',
                        help='path to directory where to write temp files.',
                        required=True,
                        type=str
                       )
    return parser


def validate_args(args):
    assert len(args.numvar_paths) == len(args.names), \
            ("Error: number of numvar paths and number of names differ.")

    assert len(args.numvar_paths) == len(args.rowid_filters_paths), \
            ("Error: number of numvar paths and number of rowid_filters paths differ.")

    assert args.ref in args.names, ("Error: <ref> name is not found among names")

    assert args.distance>0, "Provided distance is null or negative."


    for numvar_path, filter_path, filter_type in zip(args.numvar_paths,
                                                     args.rowid_filters_paths,
                                                     args.filter_types):
        if filter_path is not "":
            if filter_type is None:
                raise ValueError("'filter_type' undefined for {} ('row_id' or 'index')".format(filter_path))

            if '{}' in numvar_path:
                assert '{}' in filter_path, ("Error: one of the <rowids_filters_path> "
                                             "associated to an unformatted <numvar_paths> "
                                             "does not contain '{}' characters.")
            else:
                assert '{}' not in filter_path, ("Error: one of the <rowids_filters_path> "
                                                 "contains '{}' characters but is not "
                                                 "associated to an unformatted <numvar_paths> ")

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
        print("Created the outputdirectory :\n{}".format(args.outputdir))


def read_indices(fp):
    return [int(ind.strip('\n')) for ind in open(fp,'r').readlines()]

def write_indices(fp,indices):
    """ Caution: this function DOES NOT append to a file.
    """
    with open(fp,'w') as pf:
        for idx in indices:
            pf.write(str(idx))
            pf.write('\n')


def export_dataset_model(X, y, class_names, split_by_chrom, outputdir_model):
    """ Exporting X and y dataframes. Should be called from a notebook for demo.

    Caution: the y array should be composed of int, 0-based classes (0, 1, 2, ...) ; 
             class_names should be ordered as the y classes (ie class_names[0] will be
             associated to the '0' class in the y array.) 
    In:
        X (pandas.DataFrame)
        y (numpy.array)
        class_names (list)
        split_by_chrom (list): for each class in class_names, split the indices file by chromosome if True.
        outputdir_model (str)
        
    Out:
            
    """
    if not os.path.exists(outputdir_model):
        os.makedirs(outputdir_model)
        
    else:
        if len(os.listdir(outputdir_model))>0:
            print(("Error: '{}' already exists, and is not empty. Please remove"
                   " the directory or its content.").format(outputdir_model))
            return
               
    outputdir_dataset = outputdir_model + '/datasets/'  # Will contain the dataframe and the y array
    # Will contain the selected indices refering to the original numeric features dataframe.
    outputdir_dataset_indices = outputdir_dataset + '/datasets/{}_numeric_df_indices/' 
    os.makedirs(outputdir_dataset)

    created_files = []
    for (class_i, class_name), split_by_chrom in zip(enumerate(class_names), split_by_chrom):
        if not os.path.exists((outputdir_dataset_indices.format(class_name))):
            os.makedirs((outputdir_dataset_indices.format(class_name)))

        if split_by_chrom:
            for chrom in X.chrom.unique():
                class_indices = X.loc[(X.chrom==chrom).values & (y==class_i), :].index.values
                outputfile = outputdir_dataset_indices.format(class_name)+'/{}_indices.txt'.format(chrom)
                write_indices(outputfile, class_indices)
                created_files.append(outputfile)

        else:
            class_indices = X.loc[(y==class_i),:].index.values
            outputfile = outputdir_dataset_indices.format(class_name)+'/indices.txt'.format(class_name)
            write_indices(outputfile, class_indices)
            created_files.append(outputfile)

    # Now export the dataframes.
    X.to_csv(outputdir_dataset+'/X.tsv.gz',header=True,index=False,sep="\t",compression="gzip")
    write_indices(outputdir_dataset+'/y.tsv',y)
    
    created_files.append(outputdir_dataset+'/X.tsv.gz')
    created_files.append(outputdir_dataset+'/y.tsv')
    
    for f in created_files:
        assert os.path.isfile(f), "Error: '{}' was not created...".format(f)
        print("\tCreated file: '{}'".format(f))
    
    return


def load_model_dataset(model_datasets_directory):
    """ Just read the X and y tables.
    """
    assert(os.path.exists(model_datasets_directory+'/X.tsv.gz')), '"X.tsv.gz" not found.'
    assert(os.path.exists(model_datasets_directory+'/y.tsv')), '"y.tsv" not found.'
    
    X = pd.read_table(model_datasets_directory+'/X.tsv.gz')
    y = read_indices(model_datasets_directory+'/y.tsv')
    
    return X, y


def export_indices(output_file, sampled_indices):
    if os.path.isfile(output_file):
        # Need to merge with the existing file.
        previous_sampled_indices = [int(l.strip('\n'))
                                    for l in open(output_file,'r').readlines()]

        sampled_indices = previous_sampled_indices+sampled_indices
        sampled_indices = sorted(list(set(sampled_indices)))

    else:
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))


    with open(output_file,'w') as pf:
        for ind in sampled_indices:
            pf.write(str(ind))
            pf.write('\n')

    print("Wrote into file '{}'".format(output_file))
    return 

coding_biotypes = ['biotype:start_codon', 'biotype:CDS',
                   'biotype:Selenocysteine',
                   'biotype:exon',
                   'biotype:stop_codon', 'biotype:splice_site']

chroms = ['chr{}'.format(i) for i in range(1,23)] + ['chrX','chrY']
# NOTE HARDCODED : BAD
genome_size = "/users/ldog/moyon/Thesis/RegulationData/hg19/hg19.chrom.sizes"

def main():
    # So the first argument will be a list of sets, corresponding to path to
    # files. These files can be either normal files, or files with '{}' which
    # will be formatted with "chr1..22XY".
    #
    # The second argument should be a list of names associated to each files.
    # There can be special actions associated to each name, such as eQTLs.
    # These special actions can require specific arguments : for instance, the
    # eQTLs might be selected for their associated target genes. So it requires
    # both a path to the metadata file of eQTLs, as well as a path to a list of
    # gene names of interest.

    # The returned elements of this program should be the row_ids associated to
    # each files provided, for each name provided. So there should be an
    # outputdir, where for each name a directory is created, and then for each
    # file in the set (either a single file, or one file per chromosome.)

    # A "ref" parameter should indicate which <name> is the one that is the
    # reference, after wich the filters (distance match, or gencode match)
    # should be performed.

    # Filters can be provided as a list. Problem : if one is provided, then
    # there should be a filter for each file. Empty strings can be used for
    # situations where no filter should be applied. Also if the original path
    # to variants was unformatted, the filter path should be unformatted too.
        
    parser = create_parser()
    args = parser.parse_args()

    validate_args(args)

    # Bedtools related actions.
    tmp_dir = args.temp_dir_path
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    tmp_dir = tempfile.TemporaryDirectory(dir=tmp_dir).name
    os.makedirs(tmp_dir)
    print("Temporary bedfiles are in '{}'".format(tmp_dir))
    pbt.set_tempdir(tmp_dir)


    # FIRST PART : gather filepaths, load filters for dataframes.
    # ===========================================================

    # Get the sets of filepaths and filter paths.
    variant_sets = {}
    set_filters = {}
    set_filter_types = {}

    for name, numvar_path, rowid_filter, filter_type in zip(args.names,
                                               args.numvar_paths,
                                               args.rowid_filters_paths,
                                               args.filter_types
                                               ):
        if '{}' in numvar_path:
            variant_sets[name] = [numvar_path.format(chrom) for chrom in chroms]

            fp_found = []
            for fp in variant_sets[name]:
                fp_found.append(os.path.isfile(fp))
                if not os.path.isfile(fp):
                    print("Warning: '{}' not found.".format(fp))
                
            assert any(fp_found), "None of the files for '{}' set were found.".format(name)

            if any([os.path.isfile(rowid_filter.format(chrom)) for chrom in chroms]):
                lf = []
                for chrom in chroms:
                    if os.path.isfile(rowid_filter.format(chrom)):
                        lf.append([ri.strip('\n')
                                  for ri in open(rowid_filter.format(chrom),'r').readlines()])
                    else:
                        lf.append([])

                set_filters[name] = lf

            else:
                print("Warning: no rowid_filters_path for set '{}'".format(name))
                set_filters[name] = [None for _ in chroms]

            if filter_type != "":
                set_filter_types[name] = [filter_type for _ in chroms]
            else:
                set_filter_types[name] = [None for _ in chroms]

        else:
            assert os.path.isfile(numvar_path), ("Error: '{}' not found.".format(numvar_path))
            variant_sets[name] = [numvar_path]

            if os.path.isfile(rowid_filter):
                set_filters[name] = [[ri.strip('\n') for ri in open(rowid_filter,'r').readlines()] ]
            else:
                print("Warning: no rowid_filters_path for set '{}'".format(name))
                set_filters[name] = [None]

            if filter_type != "":
                set_filter_types[name] = [filter_type]
            else:
                set_filter_types[name] = [None]
    

    # SECOND PART : PREPARE ARGUMENTS FOR SELECTION MODEL
    # ===================================================


    usecols = ['chrom','start','end','row_id','biotypes.best_biotype', 'cytoband']

    if args.match_type == 'gencodeMatch':
        # We need to get the full biotypes composition of the reference dataset
        # before going for the sampling of others sets.
        # Indeed, we do not want to sample the specific biotype composition of
        # each chromosome, but an overall genomic composition.
        ref_biotype_counts = pd.Series()

        for fp, row_filter, filter_type in zip(variant_sets[args.ref],
                                               set_filters[args.ref],
                                               set_filter_types[args.ref]
                                               ):

            tmp = memory_aware_loading(fp, row_filter,
                                       row_filter_method=filter_type,
                                       usecols=usecols)
            
            tmp_counts = tmp['biotypes.best_biotype'
                            ].replace(np.nan,'biotype:intergenic'
                                    ).apply(lambda v: dataframes.get_field_kv_pairs_list(v,',',':','biotype')
                                            ).value_counts()

            ref_biotype_counts = ref_biotype_counts.add(tmp_counts, fill_value=0)

        ref_biotypes = ref_biotype_counts
        #ref_biotype_props = ref_biotype_counts / ref_biotype_counts.sum()

        
    else:
        ref_biotypes = None
        #ref_biotype_props = None


    if args.match_type == 'cytobandMatch':
        # Here cytobands are specific to chromosomes ; a cytoband name can be
        # associated to multiple chromosomes. So, it will be important to
        # maintain this information while performing the selection.
        cytobands_ref = {}
        for fp, row_filter, filter_type in zip(variant_sets[args.ref],
                                  set_filters[args.ref],
                                  set_filter_types[args.ref]
                                  ):
            tmp = memory_aware_loading(fp, row_filter,
                                       row_filter_method=filter_type,
                                       usecols=usecols)

            # Now get the cytobands per chrom.
            cyto = tmp.groupby('chrom')['cytoband'].unique().to_dict()

            # Theoretically, if manipulating multiple files, there's only one
            # chrom here.
            for chrom, cyto_list in cyto.items():
                cytobands_ref[chrom] = cytobands_ref.get(chrom,[]) + list(cyto_list)

    else:
        cytobands_ref = None


    matchtype_variables = {
                        'cytoband_ref' : cytobands_ref,
                        'ref_biotypes':ref_biotypes,
                        #'ref_biotype_props' : ref_biotype_props,
                        'distance' : args.distance,
                      }
     

    # THIRD PART : LOAD FILES, AND APPLY SELECTION MODEL
    # ==================================================

    # Note: I don't care much of the order of the chromosomes when dealing with
    # a single-file resource : I will report the row indices of the dataframe.
    # So say I go through each chromosome in the numerical order + X Y, if
    # chr10 is before chr2 in my file, then accumulated row indices will be
    # out-of-order ; I just have to sort them before exporting.
        
    if len(variant_sets[args.ref]) == 1:
        full_ref_df = memory_aware_loading(variant_sets[args.ref][0],
                                           set_filters[args.ref][0],
                                           set_filter_types[args.ref][0],
                                           usecols=usecols)
        chrom_ref = []
        for chrom in chroms:
            try:
                chrom_ref.append(full_ref_df.groupby('chrom').get_group(chrom))

            except KeyError:
                print(("Warning: chromosome {chrom} not found in ref dataset."
                      ).format(chrom=chrom))
                chrom_ref.append(None)

        ref_loaded = True
        
    else:
        chrom_ref = variant_sets[args.ref]
        ref_loaded = False



    # For each set in the non-ref list, match variants to the ref set.
    for name in set(args.names) - set([args.ref]):
        if len(variant_sets[name]) == 1:
            full_other_df = memory_aware_loading(variant_sets[name][0],
                                                 set_filters[name][0], # NOTE : used to be args.ref here...
                                                 set_filter_types[name][0],
                                                 usecols=usecols)

            chrom_other = []
            grouped_full_other_df = full_other_df.groupby('chrom')
            for chrom in chroms:
                try:
                    chrom_other.append(grouped_full_other_df.get_group(chrom))

                except KeyError:
                    print(("Warning: chromosome {chrom} not found in ref dataset."
                          ).format(chrom=chrom))
                    chrom_other.append(None)

            other_loaded = True
        
        else:
            chrom_other = variant_sets[name]
            other_loaded = False

        
        # Create the list of arguments to provide to the Pool.
        if ref_loaded:
            if other_loaded:
                # Nothing to load, give directly the dataframes.
                list_args = ((chrom_ref[i],
                              chrom_other[i],
                              args.match_type,
                              matchtype_variables
                              )
                            for i, chrom in enumerate(chroms))

            else:
                # Need to load "other_df" 
                list_args = ((chrom_ref[i],
                              memory_aware_loading(chrom_other[i],
                                                  set_filters[name][i],
                                                  set_filter_types[name][i],
                                                  usecols=usecols),
                              args.match_type,
                              matchtype_variables
                              )
                            for i, chrom in enumerate(chroms))

        else:
            if other_loaded:
                # Need to load "ref df" 
                list_args = ((memory_aware_loading(chrom_ref[i],
                                                  set_filters[args.ref][i],
                                                  set_filter_types[args.ref][i],
                                                  usecols=usecols),
                              chrom_other[i],
                              args.match_type,
                              matchtype_variables
                              )
                            for i, chrom in enumerate(chroms))

            else:
                # Need to load both dataframes.
                list_args = ((memory_aware_loading(chrom_ref[i],
                                                  set_filters[args.ref][i],
                                                  set_filter_types[args.ref][i],
                                                  usecols=usecols),
                              memory_aware_loading(chrom_other[i],
                                                  set_filters[name][i],
                                                  set_filter_types[name][i],
                                                  usecols=usecols),
                              args.match_type,
                              matchtype_variables
                              )
                            for i, chrom in enumerate(chroms))


        pool = multiprocessing.Pool(args.ncores)
        # Now we can apply the sampling method.
        sampling_res = pool.starmap(sample_variants, (args_smp for args_smp in list_args))

        # Now : exporting.
        # Merge the results if the provided files were split by chromosome.
        for i, (name_set, loaded) in enumerate(zip(*[(args.ref, name),
                                                    (ref_loaded,
                                                        other_loaded)])):
                                                        
            outputdir_set = args.outputdir+'/'+name_set

            if loaded:
                sampled_indices = []
                for sampled_r in sampling_res:
                    if sampled_r[i] is not None:
                        sampled_indices.append(sampled_r[i])

                sampled_indices = sorted(list(itt.chain(*sampled_indices)))

                # sampled_indices = sorted(list(itt.chain(*[sampled_r[i]
                #                        for sampled_r in sampling_res])))
                # Write to a single file, named after the original filename.
                output_file = outputdir_set + '/' + \
                                os.path.basename(variant_sets[name_set][0]).split('.tsv.gz')[0]+'.txt'
                export_indices(output_file, sampled_indices)

            else:
                # Here: this means that the set was split into 1 file per chrom ;
                # we will produce a file for each chrom.
                for idx_chrom, (chrom , fp) in enumerate(zip(chroms, variant_sets[name_set])):
                    #NOTE: what happens if the file for one chrom does not exist? 
                    sampled_indices = sampling_res[idx_chrom][i]

                    if sampled_indices is None:
                        # NOTE: this is what happens : do not export anything.
                        continue

                    output_file = outputdir_set + '/' + \
                                  os.path.basename(variant_sets[name_set][idx_chrom]).split('.tsv.gz')[0]+'.txt'
                    export_indices(output_file, sampled_indices)

        # We also export the table matching ref indices to other indices in the
        # context of distanceMatching
        if args.match_type=='distanceMatch':
            tables_dist_assoc = []
            for chrom, sampling_r in zip(chroms, sampling_res):
                if sampling_r[2] is None:
                    continue
                table_refidx_to_othidx = pd.DataFrame(
                                            [(str(k),';'.join([str(ov) for ov in v]))
                                             for k,v in sampling_r[2].items()
                                            ])
                table_refidx_to_othidx.columns = [args.ref, name]
                table_refidx_to_othidx['chrom'] = chrom
                tables_dist_assoc.append(table_refidx_to_othidx)
            
            tables_dist_assoc = pd.concat(tables_dist_assoc)
            fp_table = args.outputdir + \
                 '/distanceMatch_idx-{}_associated_idx-{}.tsv.gz'.format(args.ref,name)

            tables_dist_assoc.to_csv(fp_table,header=True,index=False,sep="\t",
                                    compression='gzip')
            

            
        pool.close()
        pool.join()

    pbt.cleanup(remove_all=True)
    os.rmdir(tmp_dir)
    return 0


if __name__ == "__main__":
    return_code = main()
    sys.exit(return_code)



