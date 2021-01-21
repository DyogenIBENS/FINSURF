#! /usr/bin/env python
# coding:utf-8

""" Script dedicated to annotation of variants from bed-like table, using scores from alternative
methods.
"""


###############################################################################
# IMPORTS

import os
import sys
import collections
import yaml
import datetime
import tempfile

import argparse as ap
import itertools as itt
import numpy as np
import pandas as pd


sys.path.insert(0, "../variants_annotation/")
import annotVariants

import pybedtools as pbt




###############################################################################
# DEFINITIONS

def process_config(configfile):
    with open(configfile,'r') as pf:
        resources = yaml.load(pf)

    full_processing_params = collections.OrderedDict()

    for score_name, resource_info in resources.items():
        if resource_info['filetype'] == 'bigWig':
            func = annotVariants.get_bigwig_annotations
            annot_params_tuple = (func,
                                  resource_info['path'],
                                  resource_info['chr_prefixed']
                                  )
        

        elif resource_info['filetype'] in ['bed','tsv','Bed']:
            if resource_info['altspec'] == True:
                func = annotVariants.get_bedfile_annotations_refalt

            else:
                func = annotVariants.get_bedfile_annotations

            kv_sep=':'
            field_sep=','
            annot_params_tuple = (func,
                                  resource_info['path'],
                                  resource_info['chr_prefixed'],
                                  resource_info['columns'],
                                  resource_info['usecols'],
                                  field_sep,
                                  kv_sep
                                 )
        else:
            raise ValueError("filetype not recognized for annot '{}'".format(score_name))
            
        full_processing_params[resource_info['name']] = annot_params_tuple

    return full_processing_params


def create_parser():
    parser = ap.ArgumentParser(epilog=__doc__,
                               formatter_class=ap.RawDescriptionHelpFormatter)

    parser.add_argument('-i',
                        '--input',
                        type=str,
                        required=True,
                        help=''
                        )

    parser.add_argument('-c',
                        '--configfile',
                        type=str,
                        required=True,
                        help=''
                        )

    parser.add_argument("-o",
                        "--output",
                        type=str,
                        required=True,
                        help="")

    parser.add_argument('-nc',
                        '--n_cores',
                        type=int,
                        required=True,
                        help=''
                        )

    parser.add_argument('-cs',
                        '--temp_dir_path',
                        type=str,
                        required=True,
                        help='Path to directory where to write temporary files.'
                        )

    parser.add_argument('-t',
                        '--chunksize',
                        type=int,
                        required=False,
                        default=50000,
                        help=''
                        )
    return parser


###############################################################################
# MAIN

def main():
    parser = create_parser()

    args = parser.parse_args()

    # temp directory
    base_tmp_dir = args.temp_dir_path

    if not os.path.exists(base_tmp_dir): os.makedirs(base_tmp_dir)
    # This will create a randomly named directory, isolated from any other
    # directory within the base dir.
    tmp_dir = tempfile.mkdtemp(dir=base_tmp_dir)+'/'
    if not os.path.exists(tmp_dir): os.makedirs(tmp_dir)
    pbt.set_tempdir(tmp_dir)
    pbt.cleanup(remove_all=True)


    # Those are the columns I need to retrieve in the table to annotate.
    expected_columns_df = ['chrom','start','end','id','ref','alt','row_id']

    df = pd.read_table(args.input, nrows=3)
    missing_cols = []
    for c in expected_columns_df:
        if c not in df.columns.values:
            missing_cols.append(c)

    if len(missing_cols)>0:
        raise ValueError("There are missing columns in the table to annotate:\n"
                         "{}".format(missing_cols))


    df = pd.read_table(args.input, usecols=expected_columns_df)

    # We are going to go through this dataframe by chunk, and annotate them
    # with the different scores defined in the config file.

    full_processing_params = process_config(args.configfile)



    # Main function that annotates variants.
    #annotated_variants_chunks = []
    chunksize = args.chunksize
    chunkidx = list(range(0,len(df),chunksize))

    print("- Annotation of chunks")
    print(("(Nb: {} ; total number of postion to annotate: {})"
                  ).format(len(chunkidx), len(df)))

    print("(writting to {})".format(args.output))

    for nb,i in enumerate(chunkidx):
        print("\tchunk {}/{}".format(nb,len(chunkidx)))
        chunk = df[i:i+chunksize]
        header = True if nb==0 else False

        annotations_chunk = annotVariants.annotate_regions(
                                             chunk,
                                             full_processing_params,
                                             args.n_cores)

        annotated_variants = pd.concat([chunk, annotations_chunk],axis=1)
        print("\tStarting writting the chunk to file: {}".format(datetime.datetime.now()))
        
        annotated_variants.to_csv(args.output,
                                sep="\t",
                                header=header,
                                index=False,
                                mode='a',
                                compression='gzip'
                                )
    
        print("\tDone writting the chunk to file: {}".format(datetime.datetime.now()))


    print("All done: {}".format(datetime.datetime.now()))
    pbt.cleanup(remove_all=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())



