#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
"""

import configobj
import datetime
import glob
import os
import sys

import argparse as ap
import itertools as itt
import pandas as pd

from functools import partial

from sklearn.externals import joblib

sys.path.insert(0, "../utils/")
from dataframes import *

import finsurf_treeinterpreter_nopool as finsurf_interpreter


# DEFINITIONS



def argparser():
    parser = ap.ArgumentParser()

    parser.add_argument("-in",
                        "--input_numfile",
                        type=str,
                        required=True)

    parser.add_argument("-ig",
                        "--input_genfile",
                        type=str,
                        required=True)

    parser.add_argument("-o",
                        "--outputfile",
                        type=str,
                        required=True)

    parser.add_argument("-pm",
                        "--path_model",
                        type=str,
                        required=True,
                        )

    parser.add_argument("-pc",
                        "--path_columns_file",
                        type=str,
                        required=True,
                        )
    
    parser.add_argument('--prediction_mode',
                        type=str,
                        required=True,
                        choices=['scoring','feature_contribs'],
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



# MAIN

def main(): 
    parser = argparser()
    args = parser.parse_args()

    print("\n")
    if os.path.exists(args.outputfile):
        print(("ERROR: {} already exists ; please remove it before execution."
               ).format(args.outputfile))
        sys.exit(1)

    # Load model
    model = joblib.load(args.path_model)

    # Load columns
    columns_model = pd.read_csv(args.path_columns_file,
                                  header=None,
                                  sep="\t",
                                  ).iloc[:,0].values

    # Load head of numeric file + head genes file ; check all columns are
    # available for the model.

    input_num = pd.read_csv(args.input_numfile,
                            header=0,
                            index_col=None,
                            sep="\t",
                            nrows=5
                           )
    input_gen = pd.read_csv(args.input_genfile,
                            header=0,
                            index_col=None,
                            sep="\t",
                            nrows=5
                           )

    # Check that the numeric file and genes file are of same size (? might be
    # extremely time consuming...)


    # Now process the numeric file and genes files

    reader_num = pd.read_csv(args.input_numfile,
                            header=0,
                            index_col=None,
                            sep="\t",
                            chunksize=args.chunksize
                           )
    reader_gen = pd.read_csv(args.input_genfile,
                            header=0,
                            index_col=None,
                            sep="\t",
                            chunksize=args.chunksize
                           )


    # Iterate over chunks from both files.
    # Build the dataframe of numeric values to be processed by the model.
    # Check if the prediction should be score only or feature contribs.
    # Write iteratively in the output file.
    # Note : if feature contribs, there needs to be a conversion into "class 1
    # only", and then a re-indexing of variants.

    for i, (chunk_num,chunk_gen) in enumerate(zip(reader_num, reader_gen)):
        if i==0:
            header=True

        else:
            header=False

        # Build the temporary X dataframe will all numeric values for the
        # model.
        tmp_num_X = chunk_num.copy()
        tmp_num_X['ratio_shared_targets'] = chunk_gen['ratio_shared_targets'].values
        tmp_num_X = tmp_num_X.loc[:,columns_model]

        if args.prediction_mode == 'scoring':
            # Get the predictions
            preds = pd.DataFrame(model.predict_proba(tmp_num_X))
            tmp_res = preds 

        else:
            # build the Interpreter object.
            model_interpreter = finsurf_interpreter.Interpreter(model=model,
                                                                X = tmp_num_X)
            # Calculate the feature contributions for each variant (for both
            # classes)
            model_interpreter.predict()

            # Now reshape the calculated feature contributions into a
            # dataframe.
            tmp_feature_contrib_df = []
            for i, (mat, idx) in enumerate(zip(model_interpreter.finale_FC_mat,
                                               tmp_num_X.index.values)):
                var_FC_df = pd.DataFrame(mat.T,columns=['pred','bias']+list(columns_model)
                                        ).reset_index().rename(columns={'index':'class'}
                                                              ).assign(sample_idx=idx)

                tmp_feature_contrib_df.append(var_FC_df)

            tmp_feature_contrib_df = pd.concat(tmp_feature_contrib_df)
            tmp_feature_contrib_df_CL1 = tmp_feature_contrib_df.loc[tmp_feature_contrib_df['class']==1,:]#.set_index(sample_idx).sort_index()
            tmp_res = tmp_feature_contrib_df_CL1
        
        # Export.
        tmp_res.to_csv(args.outputfile,
                       header=header,
                       sep="\t",
                       compression="gzip",
                       mode="a"
                      )
        

    return 0 

if __name__ == "__main__":
    sys.exit(main())

