#! /usr/bin/env python
# -*- coding:utf-8 -*-


###############################################################################
# IMPORTS

import collections
import os
import re
import sys

import numpy as np
import pandas as pd
import functools

from IPython.display import display

os.environ['QT_QPA_PLATFORM']='offscreen'


def display_df(df):
    with pd.option_context("display.max_columns",
                            100, "display.max_rows",
                            100, 'display.max_colwidth', -1):
        display(df)


def silent_try_convert(var,requested_type,return_val):
    """ Cast to requested_type if possible, otherwise silently return return_value.

    In:
        var: value to be cast into the requested type
        requested_type: type to use for casting
        return_val: value to return if the cast does not work. If set to None: return <var> as is
    """
    try:
        return requested_type(var)
    except (ValueError, TypeError) as e:
        if return_val is not None:
            return return_val
        else:
            return var

try_int = functools.partial(silent_try_convert,requested_type=int)
try_float = functools.partial(silent_try_convert,requested_type=float)

try_int_else_zero = functools.partial(silent_try_convert,requested_type=int,return_val=0)
try_float_else_zero = functools.partial(silent_try_convert,requested_type=int,return_val=0)


def split_to_bins(X, y_proba_pred, start=0, end=1, step=0.2):
    """ Copy a dataframe and associated to each row a bin_step.

    bin steps are defined from start, end, and step values. Then the samples
    are assigned to each bin by looking at the y_proba_pred vector.

    A dataframe with a new column is returned.

    In:
        X: pandas.DataFrame with input sample, to be copied.
        y_proba_pred: N*1 vector of probabilities, used for assignment to bins.
        start (int)
        end (int)
        step (float)

    Out:
        pandas.DataFrame copied from X with one more column with bin names.
    """
    # Separating the test set into bins basing on the prediction score.
    bins = np.arange(start,end,step)
    bin_names = ["[{};{}[".format(s,e) for s,e in zip(bins,np.arange(start+step,end+step,step))]

    preds_assignment_to_bins = np.digitize(y_proba_pred,bins=bins)

    bin_names_assignment = [bin_names[i-1] for i in preds_assignment_to_bins]
    bin_names_assignment[:4]

    X_binned = X.copy()
    X_binned['predicted_bin'] = pd.Categorical(bin_names_assignment, bin_names)

    return X_binned


def drop_multiple_columns(df, regex_col_list):
    """ Drop columns using the list of regex and return a new dataframe.
    """
    for regex_col in regex_col_list:
        try:
            df = df.drop(df.head().filter(regex=regex_col).columns,axis=1)
        except:
            pass
    return df


def explode_df_from_multivalue_columns(df, lst_cols, fill_value=''):
    """ Explode single-cell with list of values into multiple rows.

    From the provided list of columns, cells that contain list of values are
    used to explore these lists into multiple rows, copying other fields in the
    row.
    """
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list): lst_cols = [lst_cols]

    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].apply(len)

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({col:np.repeat(df[col].values,
                             df[lst_cols[0]].str.len())
                             for col in idx_cols
                            }).assign(**{col:np.concatenate(df[col].values)
                                        for col in lst_cols
                                        }).loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({col:np.repeat(df[col].values,
                             df[lst_cols[0]].str.len())
                             for col in idx_cols
                             }).assign(**{col:np.concatenate(df[col].values)
                                         for col in lst_cols
                                         }).append(
                                         df.loc[lens==0, idx_cols]
                                         ).fillna(fill_value).loc[:, df.columns]


def pair_and_select(values, colnames, select_cols):
    """ Map colnames to values, and select elements with a bool vector.
    """
    return np.array(list(zip(colnames,values)))[np.array(select_cols)]


def convert_keyvalue_pairs_to_str(array_kv_pairs, field_sep=',', kv_sep=':'):
    """ Convert an array of colname:value pairs to a single string.

    By default: kv separator is ':', field separator is ','.
    """
    return field_sep.join(['{}{}{}'.format(kv[0],kv_sep,kv[1])
                           for kv in array_kv_pairs])


def transform_to_dict(kv_pairs, field_sep, kv_sep,
                      ordered=True, no_fail=False, replace_chars={}):
    """ Create a dictionary from a string of key-value pairs.

    In:
        kv_pairs (str): string to convert to dictionary
        field_sep (str): separator between kv pairs
        kv_sep (str): separator between key, and value
        ordered (bool, default=True): whether to return a dict() or collections.OrderedDict()
        no_fail (bool, default=False): if True, silently return the value as is if conversion fails.
        replace_char (dict, default={}): dictionary used to replace any character in the kv_pairs string to a new character.

    Out:
        dictionary mapping keys to values.
    """
    try:
        dict_struct = collections.OrderedDict if ordered else dict
        # Here: create a list of tuple with (key,value) read from a string split
        # according to provided separators.
        list_kvs = [re.split(kv_sep,
                             kv.translate(str.maketrans(replace_chars)).strip(),
                             maxsplit=1)
                    for kv in kv_pairs.split(field_sep) if kv!=['']]
        # It might happen that some keys are missing values ;
        # we add an empty string to these.
        list_kvs = [kv+[''] if len(kv)==1 else kv for kv in list_kvs]
        return dict_struct(list_kvs)

    except Exception:
        if no_fail:
            return kv_pairs
        else:
            print(sys.exc_info())
            print(kv_pairs)
            raise Exception


def get_field_kv_pairs_list(kv_pairs, field_sep, kv_sep, field, default_value=None, dict_kwargs={}):
    """ Get value associated to field from dict-like str. Empty string if fail.

    In:
        kv_pairs (str): string to convert to a dictionary
        field_sep (str): separator between kv pairs
        kv_sep (str): separator between key, and value
        field (str): field to get from the dictionary
        default_value (default=None): value to return if key not found in dict.
        dict_kwargs: dict with arguments to be passed to transform_to_dict

    Out:
        if all went well:
            str associated to the requested field
        else:
            numpy.nan
    """
    try:
        value_as_dict = transform_to_dict(kv_pairs, field_sep, kv_sep, **dict_kwargs)
        d = value_as_dict.get(field,default_value)
        return d

    except:
        if default_value is not None:
            return default_value
        else:
            #print(str(sys.exc_info()[0])+"; value: {}".format(kv_pairs))
            return kv_pairs
