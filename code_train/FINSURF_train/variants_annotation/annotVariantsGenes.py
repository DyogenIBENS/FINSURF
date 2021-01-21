#! /usr/bin/env python
# -*- coding:utf-8 -*-

""" Annotate variants with genes information from GENCODE and FINSURF regulatory regions.
"""


###############################################################################
# IMPORTS


# CONFIG FILE

import configobj

import collections
import datetime
import functools
import os
import sys
import gzip

import argparse as ap
import itertools as itt
import numpy as np
import pandas as pd

import multiprocessing

import tempfile


# NOTE: this will need to be included in the package.
sys.path.insert(0, "../utils/")
from dataframes import pair_and_select, convert_keyvalue_pairs_to_str, transform_to_dict

import annotToNumeric


###############################################################################
# DEFINITIONS


def read_gene_set(genes_file, mapped_rowid, row_ids):
    genes = []
    n_genes = 0
    with open(genes_file,'rb') as pf:
        for l in pf.readlines():
            l = l.decode('utf8').strip('\n')
            if l.startswith('#'):
                continue

            else:
                n_genes+=1
                genes.append(l.split(';'))

    if mapped_rowid:
        if not row_ids.unique().shape[0] == n_genes:
            raise ValueError(("Error : <mapped_rowid> is True, but {:,} genes "
                              "were detected for {:,} unique row_id."
                             ).format(n_genes, row_ids.unique().shape[0])
                            )

        if n_genes>1:
            genes = [set(v) for v in genes]

        else:
            # Special case : a single variation is considered.
            genes = set(genes[0])

    else:
        if n_genes>1:
            genes = set(itt.chain.from_iterable(genes))

        else:
            # Special case : a single variation is considered.
            genes = set(genes[0])

    return genes

def create_args_parser():
    parser = ap.ArgumentParser(epilog=__doc__,
                               formatter_class=ap.RawDescriptionHelpFormatter)

    parser.add_argument("-i",
                        "--input_annotated",
                        type=str,
                        help="Input of annotated positions.",
                        required=True)

    parser.add_argument("-o",
                        "--output",
                        type=str,
                        help="",
                        required=True)


    parser.add_argument("-g",
                        "--genes_file",
                        type=str,
                        help="",
                        required=True)

    parser.add_argument("-m",
                        "--mapped_rowid",
                        type=str,
                        choices=['False','True'],
                        help=("If set to true, genes in <genes_file> are "
                              "assigned to each row_id in <input_annotated>.\n"
                              "Otherwise, they are considered in bulk (each "
                              "variant will be tested against the full set.)"),
                        required=True)

    parser.add_argument('-cfg_gene',
                        '--configfile_gene',
                        type=str,
                        required=True
                       )

    parser.add_argument('-cfg_ann',
                        '--configfile_ann',
                        type=str,
                        required=True
                       )

    parser.add_argument("-c",
                        "--n_cores",
                        type=int,
                        help="",
                        required=False,
                        default=4)

    parser.add_argument("-cs",
                        "--chunksize",
                        type=int,
                        help="",
                        required=False,
                        default=50000)
    return parser

def assign_class_any_gene(row):
    """ assign genic / enhancer location label from row of gene sets.

    For a given variant with its associated gene sets, indicate whether the
    variant is in gene, enhancer, both, or intergenic.
    """

    has_targs = len(functools.reduce(set.union,row.filter(regex='targets')))>0
    in_gene = len(row['biotypes.multi.names'])>0
    
    if in_gene:
        if has_targs:
            return "in_gene_and_enhancer"
        
        else:
            return "in_gene"
    
    else:
        if has_targs:
            return "in_enhancer"
        else:
            return "intergenic"


# This function bellow is more precise and should be applied when the sets of
# genes have been intersected with a set of interest.
def assign_class_targetgene(row):
    """ Assign precise relationship label of association to genes.

    For a variant with multiple sets of filterered genes,
    indicate which kind of association links it to a gene.

    Here should be provided a row with sets of genes of interest. Any non-empty
    set will be used to characterize the row / variant : is it in a gene ; is
    it in an enhancer ; is it in both ; is it the same target gene in both sets
    or not ; etc.

    Labels considered are:
        - bestBiotype
        - bestBiotype+enhancer_selfTarget : meaning that the gene of interest
                                            in which the variant is is also an
                                            enhancer target (there might be
                                            other targets of interest !! but
                                            they are supplanted by this one.)
        - bestBiotype+enhancer_otherTarget : meaning that among enh targs,
                                             there's another gene from the gene
                                             set. BUT THE GENE IN WHICH THE
                                             VARIANT IS IS NOT AMONG THEM.
        - notBestBiotype
        - notbestBiotype+enhancer_selfTarget
        - notbestBiotype+enhancer_otherTarget
        - enhancer_target : one of the target genes is a gene of interest,
                            irrespective of the variant location (within
                            a gene that is not of interest, or outside of any
                            gene)
        - closest_gene : same reasoning.

    In:
        array of gene sets.

    Return label (string).

    """
    if len(functools.reduce(set.union,row.values))==0:
        return 'not_associated'
    
    enh_set = functools.reduce(set.union, row.filter(regex='targets').values)
    
    if len(row['biotypes.best_biotype'])>0:
        label = 'bestBiotype'
        # Need to check enhancers :
        # Can be self targeting OR targeting other gene of interest.
        
        if len(enh_set)==0:
            return label
        
        else:
            if len(enh_set & row['biotypes.best_biotype']) > 0:
                return 'bestBiotype+enhancer_selfTarget'
            else:
                # The enh_set is non-empty, so there are other targets of
                # interest.
                return 'bestBiotype+enhancer_otherTarget'
        
        
    elif len(row['biotypes.multi.names'])>0:
        label = 'notBestBiotype'
        # Need to check enhancers :
        # Can be self targetting OR targetting other gene.
        
        if len(enh_set)==0:
            return label
        
        else:
            if len(enh_set & row['biotypes.multi.names']) > 0:
                # Note : there might be multiple cases of selfTargeting here!
                return 'notBestBiotype+enhancer_selfTarget'
            else:
                return 'notBestBiotype+enhancer_otherTarget'
        
    
    else:
        # Need to check enhancers
        if len(enh_set)>0:
            return 'enhancer_target'
        else:
            closest_genes = functools.reduce(set.union, row.filter(regex='closest').values)
            if len(closest_genes) > 0:
                return 'closest_gene'

            else:
                return 'ERROR'


def ratio_shared_targets(row):
    if len(functools.reduce(set.union,row.values))==0:
        return 0

    try:
        gene_counts = collections.Counter(
                        itt.chain.from_iterable(row.apply(tuple)))

        count_counts = collections.Counter(gene_counts.values())

        total_count = 0
        shared_count = 0
        for k in count_counts:
            if k==1:
                # Number of genes that are only associated by a single source
                total_count+=count_counts[k]
            else:
                # Number of genes that are associated by k>=2 sources (in which
                # case share count is incremented by the number of such genes)
                total_count+=count_counts[k]
                shared_count+=count_counts[k]

        return shared_count/total_count

    except Exception as e:
        print(e)
        return 0


# Main and arguments
# ------------------

# Only consider assembled autosomes and sexual chromosomes.
chroms_chr = ['chr'+str(i) for i in range(1, 23)]+['chrX','chrY']
chroms = [chrom.strip('chr') for chrom in chroms_chr] 



def main():
    parser = create_args_parser()
    args = parser.parse_args()

    if os.path.exists(args.output):
        raise ValueError('File "{}" already exists'.format(args.output))

    mapped_rowid = eval(args.mapped_rowid)
    args.output = args.output if args.output.endswith('.gz') else args.output+'.gz'

    print("Starting script: {}".format(datetime.datetime.now()))


    print("- Reading config files: {}".format(datetime.datetime.now()))

    config_ann = configobj.ConfigObj(args.configfile_ann)

    # LOAD BEST BIOTYPE METADATA
    path_gen_biot = config_ann['GENERAL']['datadir'] + config_ann['GENERAL']['biotype_order']
    gencode_order_biotypes = pd.read_table(path_gen_biot,header=0)
    gencode_dict_order_biotypes = gencode_order_biotypes.set_index('biotype')['importance'].to_dict()


    # Now get the operations to apply on the annotations.
    full_list_operations, new_names_metadata = annotToNumeric.process_config_yaml(
                                                    args.configfile_gene,
                                                    gencode_dict_order_biotypes)

    new_names_metdata = annotToNumeric.complete_new_names_metadata(new_names_metadata, config_ann)

    print("- Reading input file: {}".format(datetime.datetime.now()))

    annot_table = pd.read_table(args.input_annotated)

    print("- Reading genes file: {}".format(datetime.datetime.now()))
    genes = read_gene_set(args.genes_file, mapped_rowid, annot_table['row_id'])


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # FIRST PART : GET GENE SETS

    # Here : the operation aims at getting for all variants the list of gene names
    # to which they are associated
    # through GENCODE ; closest gene ; enhancer sets.
    print("- FIRST PART : get genes sets: {}".format(datetime.datetime.now()))

    full_list_ops_w_values = annotToNumeric.complete_list_ops_chunk(
                                                annot_table,
                                                full_list_operations,
                                                new_names_metadata)

    # 2) process the operations (getting the gene sets per annotation)
    pool = multiprocessing.Pool(12)

    gene_num_annots = pool.map(annotToNumeric.apply_operation, full_list_ops_w_values)
    gene_num_annots = pd.concat(gene_num_annots,axis=1)
                                
    pool.close()
    pool.join()
                                
    # 3) Remove the 'blank strings' sets.
    gene_num_annots = gene_num_annots.applymap(lambda v: set(v.split(';')) - set(' '))


    # Here we prepare the sub-table of enhancer interactions only.
    tmp = gene_num_annots.filter(regex='targets')

    to_remove = []
    for c in tmp.columns:
        if 'genehancer' in c or 'genhancer' in c:
            to_remove.append(c)

    tmp_nogen = tmp.drop(to_remove, axis=1)


    # From here we can already assign a label in terms of location in gene or
    # enhancer.
    label_location = gene_num_annots.apply(assign_class_any_gene, axis=1)
    label_location = label_location.rename('genomic_location')

    label_location_nogen = gene_num_annots.drop(to_remove,axis=1
                                ).apply(assign_class_any_gene, axis=1)

    label_location_nogen = label_location_nogen.rename('genomic_location.noGenehancer')

        
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    print("- SECOND PART : intersect with input genes: {}".format(datetime.datetime.now()))

    if mapped_rowid:
        # Caution : need to spread the gene set per row id to all rows
        # associated to this row_id
        table_genes = pd.Series(genes).rename('gene'
                        ).to_frame().assign(row_id=annot_table['row_id'].unique())

        genes = pd.merge(annot_table.loc[:,['row_id']],
                         table_genes,left_on='row_id',right_on='row_id',
                         how='left')['gene'].values

        rows_sets = zip(gene_num_annots.iterrows(), genes)

    else:
        rows_sets = zip(gene_num_annots.iterrows(), itt.repeat(genes))


    subset_gene_num_annots = []
    for (i, row), gene_set in rows_sets:
        subset_gene_num_annots.append(row.apply(lambda v: v & gene_set))
    #
    subset_gene_num_annots = pd.concat(subset_gene_num_annots, axis=1).T


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    print("- THIRD PART : label target genes : {}".format(datetime.datetime.now()))

    label_genesets = subset_gene_num_annots.apply(assign_class_targetgene, axis=1)
    label_genesets = label_genesets.rename('targets_associations')

    label_genesets_nogen = subset_gene_num_annots.drop(to_remove, axis=1
                            ).apply(assign_class_targetgene, axis=1)

    label_genesets_nogen = label_genesets_nogen.rename('targets_associations.noGenehancer')


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    print("- FOURTH PART : get scores : {}".format(datetime.datetime.now()))

    # We collect 2 scores :
    # - ALL enhancer sets
    # - enhancer sets without GENEHANCERS

    shared_target_ratio = tmp.apply(ratio_shared_targets, axis=1
                                   ).rename('ratio_shared_targets')
    shared_target_ratio_nogen = tmp_nogen.apply(ratio_shared_targets, axis=1
                                ).rename('ratio_shared_targets.noGenehancer')


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    print("- FINALE PART : collect and output : {}".format(datetime.datetime.now()))

    # Now agregate all info into a table.

    full_df_output = pd.concat([
                        gene_num_annots.applymap(tuple).applymap(';'.join),
                        label_location,
                        label_location_nogen,
                        shared_target_ratio,
                        shared_target_ratio_nogen,
                        label_genesets,
                        label_genesets_nogen
                        ], axis=1)
                

    print("- Exporting... : {}".format(datetime.datetime.now()))
    full_df_output.to_csv(args.output, header=True,
                          index=False,
                          sep='\t',
                          compression='gzip')

    return 0







###############################################################################
# MAIN

if __name__ == "__main__":
    return_code = main()
    sys.exit(return_code)
