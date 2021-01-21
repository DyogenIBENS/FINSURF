#! /usr/bin/env python
# coding:utf-8

""" Convert raw annotations of variants into numeric values 
that can be used for visualization and model training.
"""


###############################################################################
# IMPORTS

import collections
import configobj
import datetime
import glob
import os
import sys
import yaml

import argparse as ap
import itertools as itt
import multiprocessing as mp
import numpy as np
import pandas as pd

from functools import partial

sys.path.insert(0, "../utils/")
from dataframes import *


###############################################################################
# DEFINITIONS

# -----------------------------------------------------------------------------
# Transform operations


def transversion_ordinal(value):
    if value=='transition':
        return 0
    elif value=='transversion':
        return 1
    else:
        return 2
    

def has_element(value):
    """ Return 1 if annotation is a non-empty string, else 0.
    """
    if isinstance(value,str):
        if len(value)>1:
            res = 1
        else:
            # This might happen if the null values were replaced by ''.
            res= 0
    else:
        res= 0
    return res

def get_field(value, field, field_sep=',',kv_sep=':'):
    """ Transform value to dictionnary, and get the requested field value.
    """
    if isinstance(value,str):
        res=transform_to_dict(value, field_sep=field_sep, kv_sep=kv_sep)[field]
    else:
        res=None

    return res

def count_multi_hit(value, hit_separator=';'):
    """ Count the number of <hit_separator> in string <value> (0 by default.)
    """
    if isinstance(value,str) and value!='':
        return value.count(hit_separator)+1

    else:
        return 0


def mean_value_array(value, sep=None):
    """ Average from string-formated array of values separated by <sep>

    If <value> is a string, it will be split by <sep> and the elements
    will be cast to float (default=0), from which the average is returned.

    Else if <value> is an iterable, then its elements are considered to be strings ;
    each element is split by <sep> and all values are aggregated in a single list,
    from which the average is returned.
    """
    if not sep:
        sep=";"

    if isinstance(value, collections.Iterable) and not isinstance(value, str):
        acc_values = list(itt.chain(*[v.split(sep) for v in value]))
        mean_val = np.mean([try_float(v, return_val=0) for v in acc_values])
        return mean_val 
                
    elif isinstance(value, str):
        return np.mean([try_float(v) for v in value.split(sep)])

    return np.nan


def join_unique(value, sep=None):
    """ Return a string of unique values separated by <sep>

    If <value> is a string, it will be split by <sep> and the elements
    will be transformed to a set, formatted back into a string (joined by sep).
    
    Else if <value> is an iterable, then its elements are considered to be strings ;
    each element is split by <sep> and all values are aggregated in a single list,
    from which unique elements are joint back with <sep>
    """
    if not sep:
        sep=";"

    if isinstance(value, collections.Iterable) and not isinstance(value, str):
        unique_values = set(itt.chain(*[v.split(sep) for v in value]))
        return sep.join(unique_values)
                
    elif isinstance(value, str):
        return sep.join(set(value.split(sep)))

    else:
        return None


def count_unique(value, sep=None):
    if not sep:
        sep=";"
    return count_multi_hit(join_unique(value, sep=sep), hit_separator=sep)
        

def accumulate_field(value, field, hit_separator, field_sep=',',kv_sep=':'):
    """ Accumulate annotation from <field> across multiple hits in value.

    From a string value with multiple hits separated by <hit_separator>,
    retrieve the <field> associated annotation, and accumulate it into an
    array.

    For a non-empty string value, there should be N_hit_separator + 1 values in
    the constructed array.

    """
    if isinstance(value, str):
        acc_values = [get_field_kv_pairs_list(v,field_sep,kv_sep,field)
                      for v in value.split(hit_separator)]
        return acc_values
                
    else:
        return None


def get_best_hit(kv_pairs_list, hit_separator, method,
                 gencode_dict_order_biotypes=None, field_sep=',', kv_sep=':'):
    """ Return annotation with best hit criterion for interesected region.

    In cases of region-intersection annotations, multiple regions can be
    reported. In such cases, one way to deal with these annotations is to
    choose the best one, relying on some criterion.
    
    Criterion for selection:
    - score: the annotations should have a 'score' field, from which a numeric
             value can be retrieved and used for ordering.
    - first: just take the first annoation from the list of annotations.
    - biotype: specific for gencode annotations. Returns the most impactful
               biotype.

    
    In:
        kv_pairs_list (str): single str containing multiple kv_pairs groups,
                             from which one will be selected basing on the
                             method requested.

        kv_pairs_sep (str): separator of the different hits to split on

        method_best_hit (str): on which field the kv_pairs should be selected.

        field_sep (str, default=','): separator of fields within a kv-pair group.
        kv_sep (str, default=':'): separator between k and v within a kv-pair group.


    Out:
        str: best kv_pairs group
        OR
        initial kv_pairs_list (usually a numpy.nan value)
    """
    if isinstance(kv_pairs_list,str):
        # split annotations
        all_kv_pairs = kv_pairs_list.split(hit_separator)
        # And keep the best hit
        if method=='score' or method=='end' or method=='pos':
            scores = [try_float(get_field_kv_pairs_list(kv_pairs,
                                                        field_sep,
                                                        kv_sep,
                                                        method),
                                return_val=np.nan)
                      for kv_pairs in all_kv_pairs]

            best_annot = all_kv_pairs[np.argmax(scores)]
            res = best_annot

        elif method=='first':
            res = all_kv_pairs[0]

        elif method=='biotype':
            assert get_field_kv_pairs_list is not None
            biotypes_list = [get_field_kv_pairs_list(kv_pairs,
                                                     field_sep,
                                                     kv_sep,
                                                     'biotype')
                             for kv_pairs in all_kv_pairs]

            # Here the biotypes are ordered by "importance" (this should have
            # been defined during the GENCODE file processing). The highest the
            # worst => if a biotype was not defined it's assigned a high score.
            biotypes_scores = [gencode_dict_order_biotypes.get(b,10000)
                               for b in biotypes_list]
            res = all_kv_pairs[np.argmin(biotypes_scores)]

        elif "field_" in method:
            # Similar to "score" method, but the field can be given by the
            # user.
            # Note that the associated value should be numeric!!
            field = method.split('field_')[1]
            values = [try_float(get_field_kv_pairs_list(kv_pairs,
                                                        field_sep,
                                                        kv_sep,
                                                        field),
                                return_val=np.nan)
                      for kv_pairs in all_kv_pairs]

            best_annot = all_kv_pairs[np.argmax(values)]
            res = best_annot

    else:
        # No processing possible ; return the value as is.
        res = kv_pairs_list

    return res

# NOTE: ugly but necessary for my apply_composite_operations...
def copy_val(v):
    return v

# -----------------------------------------------------------------------------
# Higher order operations
# ======================
#
# Those functions are applied on an array of values rather than single values.

def explode_keyvalues(name, values, replace_na, expected_cols,
                      field_sep=',', kv_sep=':'):
    """ Create a dataframe from a list of "k<kv_sep>v<field_sep>k<kv_sep>v".

    Note that created columns that are mostly composed of null values might
    arise from empty hit situations. This is solved by requesting a set of
    expected columns : any column not matching this set is removed.

    Column names are prefixed with the <name>.
    """
    exploded = [transform_to_dict(v,field_sep,kv_sep) for v in values]
    exploded = pd.DataFrame(exploded).astype(float) # Keep as float for NaN.
    # Now remove unexpected columns and fill with replace_na (should be 0)
    exploded = exploded.loc[:,expected_cols].replace(np.nan,replace_na)
    # The columns should be named after the keys detected in the operation.
    exploded = exploded.add_prefix(name+'.')
    return exploded

def copy_values(name, values, replace_na=None):
    """ Re-instanciate a pandas.Series from <values>, named with <name>
    """
    res = pd.Series(values,name=name).replace(np.nan, replace_na)
    return res

def apply_composite_operations(name, values, replace_na, list_op, list_args):
    """ Apply on <values> the sequence of operations in <list_op> with their <list_args>
    """
    res = values
    # Really not optimal, as I will go through the list multiple times...
    try:
        for i, op in enumerate(list_op):
            res = [op(v, **(list_args[i])) for v in res]

        res = pd.Series(res,name=name).replace(np.nan,replace_na)
        return res

    except Exception as e:
        print("ERROR during processing of annotation '{}'".format(name))
        print(sys.exc_info())
        sys.exit(1)


def apply_operation(args):
    """ Generic function to apply a function with required arguments.
    """
    (func,
     colname,
     values,
     replace_na) = args[:4]

    ops_args = []
    for el in args[4:]:
        if el is not None:
            ops_args.append(el)

    if len(ops_args)==0:
        ops_args = [{}]

    if func is apply_composite_operations:
        return func(colname, values, replace_na, *ops_args)
    else:
        return func(colname, values, replace_na, **(ops_args[0]))

# -----------------------------------------------------------------------------
# Args and config files + output config file.

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

    parser.add_argument("-cfg_ann",
                        "--configfile_ann",
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

def process_op_name(op_name):
    map_optname_to_func = {'countMultiHit':count_multi_hit,
                           'getField':get_field,
                           'hasElem':has_element,
                           'keepAs':copy_val,
                           'getBestHit':get_best_hit,
                           'accumulateField':accumulate_field,
                           'joinUnique':join_unique,
                           'countUnique':count_unique,
                           'meanValue':mean_value_array,
                           'transOrd':transversion_ordinal
                          }

    return map_optname_to_func[op_name]


def process_config_yaml(configfile, gencode_dict_order_biotypes):
    """ Read processing options of annotations from the yaml config file.

    The yaml file should associate to EACH column of the annotation file a set
    of operations to apply; these operations are completed with options,
    defined in the yaml as well.

    A list of function / options / column names will be returned, as well as a
    second metadata dictionary, compiling the associations between "old column
    names" and "newly generated column names".

    In:
        configfile (str): path to the yaml config file.
        gencode_dict_order_biotypes (dict): a dictionary matching biotypes to a
                                            score.

    Out:
        full_list_operations: list with elements such as:
                                (function for operations processing,
                                 new column name,
                                 list of functions to apply sequencially,
                                 list of arguments for each function)
        new_names_metadata:
    """
    conf = yaml.load(open(configfile))

    # Keys should be columns in the annotation dataframe ; associated values
    # contain the operations that need to be parsed.

    # The script requires the Python version to be ≥3.6, so that the ordered
    # dict is parsed with ordered keys.

    new_names_metadata = collections.OrderedDict()

    # Here the format is (new_colname, old_colname,
    #                     operation_list,  operation_args_list)
    full_list_operations = []

    for old_colname, params in conf.items():
        process_ops = params['process']
        if not process_ops:
            # These cases will not be added to the metadata, but are still
            # copied.
            format_op = (copy_values, old_colname, None)
            full_list_operations.append(format_op)
            continue

        for newcol, newcol_params in process_ops.items():
            try:
                # By default, store the column in the metadata table.
                use_col = newcol_params.get('use_col',True) 
                replace_na = newcol_params.get('replace_na',None)
                ops = newcol_params['operation']
                if 'args' in newcol_params:
                    ops_args = newcol_params['args']
                else:
                    ops_args = {}

            except KeyError as e:
                print(("KeyError while processing '{}' parameters"
                      ).format(newcol))
                print(sys.exc_info())
                sys.exit(1)
                

            if isinstance(ops, str):
                ops, ops_args = np.array([ops]), [ops_args]
            else:
                ops, ops_args = np.array(ops), ops_args

            # Add the new column metadata.
            # Special cases : getDummies, explodeKeyValues
            # which produces **multiple** columns.
            if 'getDummies' in ops:
                # IMPORTANT: for now, no combination of operations with such
                # operation. (you cannot pipe a "getField" op to a
                # "explodeKeyValues" op.
                try:
                    id_op = np.where(ops=='getDummies')[0][0]
                    path = ops_args[id_op]['file_all_names']
                    dummies = [l.strip('\n') for l in open(path,'r').readlines()]
                    # Get the column names that should be generated.
                    newcols = [newcol+'.'+dum for dum in dummies]
                    # And add them to the full dict of metadata.
                    for nc in newcols:
                        new_names_metadata[nc] = dict(
                                                    old_name=old_colname,
                                                    distype=newcol_params['distype']
                                                    )
                    

                except Exception as e:
                    error_msg = ("ERROR during processing of '{}' with "
                                 "'explodeKeyValues' operation "
                                 "for column '{}'").format(old_colname, newcol)
                    print(error_msg)
                    print(sys.exc_info())
                    sys.exit(1)

            elif 'explodeKeyValues' in ops:
                # IMPORTANT: for now, no combination of operations with such
                # operation. (you cannot pipe a "getField" op to a
                # "explodeKeyValues" op.
                try:
                    id_op = np.where(ops=='explodeKeyValues')[0][0]
                    expected_cols = ops_args[id_op]['expected_cols']
                    # Get the column names that should be generated.
                    newcols = [newcol+'.'+expc for expc in expected_cols]
                    # And add them to the full dict of metadata.
                    for nc in newcols:
                        new_names_metadata[nc] = dict(
                                                    old_name=old_colname,
                                                    distype=newcol_params['distype']
                                                    )
                    # And store the operation. 
                    format_op = (explode_keyvalues, newcol, replace_na, ops_args[0])
                    full_list_operations.append(format_op)
                    

                except Exception as e:
                    error_msg = ("ERROR during processing of '{}' with "
                                 "'explodeKeyValues' operation "
                                 "for column '{}'").format(old_colname, newcol)
                    print(error_msg)
                    print(sys.exc_info())
                    sys.exit(1)

            elif 'getBestHit' in ops:
                # I need to add the gencode dictionary of biotypes here.
                try:
                    id_op = np.where(ops=='getBestHit')[0][0]
                    ops_args[id_op].update({'gencode_dict_order_biotypes':gencode_dict_order_biotypes})

                    new_names_metadata[newcol] = dict(
                                                old_name=old_colname,
                                                distype=newcol_params['distype']
                                                )
                    # And store the operation. 
                    format_op = (apply_composite_operations, newcol,
                                 replace_na,
                                 [process_op_name(op) for op in ops],
                                 ops_args)
                    full_list_operations.append(format_op)
                    

                except Exception as e:
                    error_msg = ("ERROR during processing of '{}' with "
                                 "'getBestHit' operation "
                                 "for column '{}'").format(old_colname, newcol)
                    print(error_msg)
                    print(sys.exc_info())
                    sys.exit(1)

            else:
                # No particular operation to do ; the expected result is 1D,
                # and the new column name is directly the one in the dict.
                new_names_metadata[newcol] = dict(old_name=old_colname,
                                                  distype=newcol_params['distype'])

                # Add the operation to the dict.
                format_op = (apply_composite_operations,
                             newcol,
                             replace_na,
                             [process_op_name(op) for op in ops],
                             ops_args)

                full_list_operations.append(format_op)


    return full_list_operations, new_names_metadata



# -----------------------------------------------------------------------------
# Others


def complete_list_ops_chunk(chunk, full_list_operations, new_names_metadata):
    """ Reconstruct a list of operations, with associated data values.
    """
    full_list_ops_w_values = []
    for ops_args in full_list_operations:
        func = ops_args[0]
        colname = ops_args[1] 
        replace_na = ops_args[2]

        if colname in new_names_metadata:
            old_colname = new_names_metadata[colname]['old_name']
        else:
            old_colname = colname

        if old_colname not in chunk.columns:
            print(("WARNING: column name {} in numeric config file, "
                   " but not found in dataframe. Ignored.").format(colname))
            continue

        ops_args_w_vals = [func, colname, chunk[old_colname].values, replace_na]
        for el in ops_args[3:]:
            ops_args_w_vals.append(el)

                               
        full_list_ops_w_values.append(tuple(ops_args_w_vals))

    return full_list_ops_w_values
    
def numerize_annotations(chunk, full_list_operations, new_names_metadata, ncores):
    """ Numerize annotations from chunk according to the full_list_operations.

    <full_list_operations> should associate columns from chunk to function,
    with options, in order to transform the annotation to a numeric version of
    it. <new_names_metadata> is used to map new column names to old ones.

    The first step is to retrieve for each operation the values it will operate
    on. Then, a multiprocessing pool will go through the operations and apply
    them.

    In:
        chunk (pandas.DataFrame) : annotations.
        full_list_operations (list):
        new_names_metadata (dict):
        ncores (int)

    Return:
        pandas.DataFrame: of shape N_el x M_newcols

    """

    return numeric_annotations

def complete_new_names_metadata(new_names_metadata, conf_ann):
    """ Complete the dict structure with "color" and "category" information.
    """
    default = {'color':'#BBBBBB','category':'other'}
    # Get colors and categories.
    old_names_metadata = {}
    for category, cat_params in conf_ann.items():
        if category=='GENERAL': continue

        columns_cat = []
        for resource in cat_params.sections:
            columns_cat.append(cat_params[resource]['names'])
        
        columns_cat = list(itt.chain(*columns_cat))
        for old_name in columns_cat:
            old_names_metadata[old_name] = {'color': cat_params['color'],
                                            'category': category}

    # Complete the new_names_metadata_dict.
    for newcol, newcol_metadata in new_names_metadata.items():
        # Here I split on '.' as multiple operations might have been applied
        # during the annotations ; in which case the name was completed with
        # '.'<operation>
        old_name = newcol_metadata['old_name'].split('.')[0]
        if old_name in old_names_metadata:
            newcol_metadata.update(old_names_metadata[old_name])
        else:
            newcol_metadata.update(default)


    return new_names_metadata


# -----------------------------------------------------------------------------
# MAIN

def main():
    # This is required to have the ordered dict from the yaml config file.
    if sys.version_info[0] < 3 or sys.version_info[1] < 6:
        print("This script requires Python version 3.6 or later.")
        sys.exit(1)
    
    parser = argparser()
    args = parser.parse_args()

    print("\n")
    if os.path.exists(args.outputfile):
        print(("ERROR: {} already exists ; please remove it before execution."
               ).format(args.outputfile))
        sys.exit(1)

    # Here: parse the annotation config file to retrieve 
    # - biotypes order
    # - column names (old) to categories, as well as colors.
    conf_ann = configobj.ConfigObj(args.configfile_ann)

    # Get the gencode ordered biotypes for best_hit retrieval.
    path_gen_biot = conf_ann['GENERAL']['datadir']+\
                    conf_ann['GENERAL']['biotype_order']
    gencode_order_biotypes = pd.read_table(path_gen_biot, header=0)

    gencode_dict_order_biotypes = gencode_order_biotypes.set_index('biotype'
                                                )['importance'].to_dict()
        
    # Get the operations to apply to annotations 
    (full_list_operations,
     new_names_metadata) = process_config_yaml(args.configfile_num, gencode_dict_order_biotypes)
    

    # Complete the metadata on new columns with colors and categories.
    new_names_metadata = complete_new_names_metadata(new_names_metadata, conf_ann)

    # And now process the annotations dataframe.
    reader = pd.read_table(args.inputfile,
                           chunksize=args.chunksize)

    compression = 'gzip' if args.outputfile.endswith('.gz') else None

    pool = mp.Pool(args.ncores) 

    print("STARTING NUMERIZATION ; {}\n".format(datetime.datetime.now()))
    for i,chunk in enumerate(reader):
        print("\tchunk: {} ; {}".format(i, datetime.datetime.now()))

        full_list_ops_w_values = complete_list_ops_chunk(chunk,
                                                    full_list_operations,
                                                    new_names_metadata)

        numeric_annotations = pool.map(apply_operation,full_list_ops_w_values)

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

    # Now write the metadata into file.
    dirpath = os.path.dirname(args.outputfile)
    filename = args.outputfile.split('/')[-1].split('.tsv')[0]
    filepath_config = dirpath+'/'+filename+'_colMetadata.yaml'
    yaml.dump(dict(new_names_metadata), open(filepath_config,'w'))
    
    print("Created files:\n\t{}\n\t{}\n\n".format(args.outputfile,filepath_config))
    return 0


if __name__ == "__main__":
    sys.exit(main())



