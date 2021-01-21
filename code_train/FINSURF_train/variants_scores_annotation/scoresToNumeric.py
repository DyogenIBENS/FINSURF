#! /usr/bin/env python
# coding:utf-8

""" Script to convert annotations of variants obtained from scoring methods into numeric values that
can be used for representation, clustering, etc.
"""


###############################################################################
# IMPORTS

import collections
import datetime
import os
import sys

import argparse as ap
import itertools as itt
import multiprocessing as mp
import numpy as np
import pandas as pd

sys.path.insert(0, "../variants_annotation/")
import annotToNumeric


###############################################################################
# DEFINITIONS


def argparser():
    parser = ap.ArgumentParser()

    parser.add_argument("-i",
                        "--inputfile",
                        type=str,
                        required=True)

    parser.add_argument("-o",
                        "--outputfile",
                        type=str,
                        required=True)

    parser.add_argument("-cfg_num",
                        "--configfile_num",
                        type=str,
                        required=True,
                        )

    parser.add_argument("-cs",
                        "--chunksize",
                        type=int,
                        required=False,
                        default=50000
                        )

    parser.add_argument("-nc",
                        "--ncores",
                        type=int,
                        required=False,
                        default=1
                        )

    return parser

def main():
    parser = argparser()
    args = parser.parse_args()

    print("\n")
    if os.path.exists(args.outputfile):
        print(("ERROR: {} already exists ; please remove it before execution."
               ).format(args.outputfile))
        sys.exit(1)

    (full_list_operations,
     new_names_metadata) = annotToNumeric.process_config_yaml(
                                    args.configfile_num,
                                    None)

    reader = pd.read_table(args.inputfile,
                           chunksize=args.chunksize)

    compression = 'gzip' if args.outputfile.endswith('.gz') else None

    pool = mp.Pool(args.ncores) 

    print("STARTING NUMERIZATION ; {}\n".format(datetime.datetime.now()))
    for i,chunk in enumerate(reader):
        print("\tchunk: {} ; {}".format(i, datetime.datetime.now()))

        full_list_ops_w_values = annotToNumeric.complete_list_ops_chunk(
                                                    chunk,
                                                    full_list_operations,
                                                    new_names_metadata)

        numeric_annotations = pool.map(annotToNumeric.apply_operation,full_list_ops_w_values)

        numeric_annotations = pd.concat(numeric_annotations, axis=1)

        assert numeric_annotations.shape[0] == chunk.shape[0], \
                ("ERROR: N_rows of current num and original "
                "chunk differ: {} vs {}").format(numeric_annotations.shape[0],
                                                 chunk.shape[0])

        output_header = True if i == 0 else False

        if i>0:
            sample_outputfile = pd.read_table(args.outputfile,nrows=3)
            shape_written_file = sample_outputfile.shape[1]
            shape_towrite_df = numeric_annotations.shape[1]
            assert shape_written_file == shape_towrite_df, \
                    ("ERROR: N_cols of written file and current num "
                    "chunk differ: {} vs {}").format(shape_written_file, shape_towrite_df) 
            
        numeric_annotations.to_csv(args.outputfile,
                                   mode='a',
                                   header=output_header,
                                   compression=compression,
                                   sep="\t",index=False)

        print("Done for the chunk. {}\n".format(datetime.datetime.now()))


    print("ALL DONE ; {}\n".format(datetime.datetime.now()))
    pool.close()
    pool.join()


    return 0


if __name__ == "__main__":
    sys.exit(main())



