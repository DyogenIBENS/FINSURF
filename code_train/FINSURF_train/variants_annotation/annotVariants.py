#! /usr/bin/env python
# coding:utf-8

""" Annotate variants from a VCF file with all annotations from configured files.
"""


###############################################################################
# IMPORTS


# CONFIG FILE

import configobj

import collections
import datetime
import errno
import os
import re
import sys
import subprocess
import vcf
import gzip

import argparse as ap
import itertools as itt
import numpy as np
import pandas as pd

import pyBigWig as pbw
import pybedtools as pbt
import tabix
import pysam

from collections import OrderedDict, namedtuple

from multiprocessing import Pool

import tempfile


sys.path.insert(0, "../utils/")
from dataframes import pair_and_select, convert_keyvalue_pairs_to_str, transform_to_dict




###############################################################################
# DEFINITIONS


# bed-format related functions
# ----------------------------

def tabix_list_chrom(bed_file_path):
    # This function might be necessary to face cases where the chromosome in
    # the bw file is encoded without the "chr" string. In which case variants
    # must be converted ('chr1' => '1')
    process = subprocess.Popen(['tabix','-l',bed_file_path],stdout=subprocess.PIPE) 
    return [chrom.strip() for chrom in process.stdout]


def are_chrom_chr_prefixed(chrom_list):
    """ Given a list of chrom names, check if they are 'chr' prefixed or not.
    """
    if any(pd.Series(chrom_list).isin(chroms_chr)):
        return True
    return False


def convert_bedtools_intersection(intersection, column_names, usecols,
                                  field_sep=',',kv_sep=':'):
    selected_pairs = pair_and_select(intersection,
                                     column_names,
                                     usecols)

    return convert_keyvalue_pairs_to_str(selected_pairs, field_sep=field_sep, kv_sep=kv_sep)


def bedtools_closest_features(df_regions, bedfile, prefix_chr, colnames, usecols):
    """ Return a DF with upstream and downstream elements for chrom positions.

    The input dataframe should contain chromosome positions, that are
    associated to their closest downstream and upstream elements from a given
    bedfile.

    The closest elements are reported as string containing key-value pairs that
    are established from the list of colnames (filtered with usecols).

    Two queries are performed with BedTools on the bedfile to return the
    upstream element and the downstream element.

    """
    df_regs = df_regions.copy()
    bt_annots = pbt.BedTool(bedfile)
    
    # Check that the chromosomes have 'chr' or not:
    if prefix_chr:
        if not df_regs.iloc[0,0].startswith('chr'):
            df_regs.iloc[:,0] = 'chr' + df_regs.iloc[:,0].astype(str)
    else:
        if df_regs.iloc[0,0].startswith('chr'):
            df_regs.iloc[:,0] = df_regs.iloc[:,0].astype(str).str.strip('chr')


    df_for_bt = pd.concat([pd.Series(df_regs.iloc[:,0].values, name='chrom'),
                           pd.Series(df_regs.iloc[:,1].values, name='start'),
                           pd.Series(df_regs.iloc[:,2].values, name='end')
                            ],
                        axis=1).sort_values(by=['chrom','start','end'])


    original_index = df_for_bt.index.values
    bt_variants = pbt.BedTool.from_dataframe(df_for_bt)

    res_upstream = bt_variants.closest(bt_annots,D='a',id=True, io=True,
                                       t='first'
                                 ).to_dataframe(
                                         )
    res_upstream.index = original_index
    res_upstream.sort_index(inplace=True)

    res_downstream = bt_variants.closest(bt_annots,D='a',iu=True, io=True,
                                         t='first'
                                   ).to_dataframe(
                                           )
    res_downstream.index = original_index
    res_downstream.sort_index(inplace=True)

    result_df = pd.concat([
                    pd.Series(
                        res_upstream.apply(
                            lambda row: convert_bedtools_intersection(
                                    row[3:],
                                    colnames+['distance'],
                                    usecols+[True]), axis=1),
                        name='upstream'),
                    pd.Series(
                        res_downstream.apply(
                            lambda row: convert_bedtools_intersection(
                                    row[3:],
                                    colnames+['distance'],
                                    usecols+[True]), axis=1),
                        name='downstream')
                    ], axis=1)

    os.remove(bt_variants._tmp())
    os.remove(bt_annots._tmp())

    # Now I should really have a dataframe with nrows = n_variants
    return result_df




def get_bedfile_annotations_refalt(df_regions, bedfile_path, prefix_chr, colnames,
                            selectcols, field_sep=',', kv_sep=':', multisep='|'):
    """ Intersect regions with bedfile regions,and select columns.
    
    The first step rely on Tabix to query, for each row in the dataframe of
    region, the possible hits from the bedfile. For each position, 0, 1, or
    multiple hits can occur. Each hit value is associated to the column names,
    and a selection is performed.

    Finally for a given position, a series of hit is reported as a string, each
    hit separated by a '|', and for each hit the pairs of colname:value are
    reported, separated by ','.
    """
    if not 'ref' in df_regions.columns.values:
        raise ValueError("This function requires the 'ref' and 'alt' information in the columns.")
    if not 'alt' in df_regions.columns.values:
        raise ValueError("This function requires the 'ref' and 'alt' information in the columns.")

    df_regs = df_regions.copy()
    df_regs.index = df_regs.index.rename('chunk_idx')
    # Check that the chromosomes have 'chr' or not:
    if prefix_chr:
        if not df_regs.iloc[0,0].startswith('chr'):
            df_regs.iloc[:,0] = 'chr' + df_regs.iloc[:,0].astype(str)
    else:
        if df_regs.iloc[0,0].startswith('chr'):
            df_regs.iloc[:,0] = df_regs.iloc[:,0].astype(str).str.strip('chr')


    intersection = bedfile_intersect(df_regs, bedfile_path)

    
    # Now build a dataframe with the results ; empty results are droppped for
    # now, and will be brought back by a reindex.
    intersection_full = []
    for i, multi_intersect in zip(df_regs.index.values,intersection):
        if multi_intersect is not None:
            for multi_i in multi_intersect:
                intersection_full.append(multi_i+[i])
        # TODO : There should not be any ELSE case, but it might be safer to handle
        # them?

    intersection_full = pd.DataFrame(intersection_full,
                                     columns=colnames+['chunk_idx'])
    
    if intersection_full.empty:
        # Here is handled a situation where no hit are found.
        return pd.Series(intersection)

    # Take the opportunity to rename 'pos' to 'end' if the file was a 1-based
    # bed-line file, with a single column for positions.
    if 'pos' in intersection_full.columns.values:
        intersection_full = intersection_full.rename(columns={'pos':'end'}) 


    # Now create two index : one for the chunk, and one for the results.
    # The results will be queried with the first index.
    # Missing values will be set to None.

    # NOTE: missing values here might arise either by missing annotation of the
    # position, OR because of INDELs ... There might be a need for a more
    # elegant way to deal with these.
    build_index = lambda row: '_'.join([str(v)
                                     for v in (row['chunk_idx'],
                                               row['chrom'],
                                               row['end'],
                                               row['ref'],
                                               row['alt'])
                                        ])

    df_regs_idx = df_regs.reset_index().apply(build_index, axis=1)

    intersection_idx = intersection_full.apply(build_index,axis=1)

    df_regs['var_idx'] = df_regs_idx.values
    intersection_full['var_idx'] = intersection_idx.values
    grouped_intersections = intersection_full.groupby('chunk_idx')

    df_regs['vartype'] = df_regs.apply(lambda row: get_vartype(row['ref'], row['alt']),
                                       axis=1)

    processed_res = []
    for i, row in df_regs.iterrows():
        try:
            intersection_row = grouped_intersections.get_group(i)
        except KeyError:
            # This means there were no hit for this position.
            processed_res.append(None)
            continue

        vt = row['vartype']

        if vt=='SNV':
            # Get the associated score.
            try:
                res = intersection_row.set_index('var_idx').loc[row['var_idx']]
                # And format it.
                res = pair_and_select(res,colnames, selectcols)
                res_str = convert_keyvalue_pairs_to_str(res,
                                                field_sep=field_sep,
                                                kv_sep=kv_sep)
            except KeyError:
                res_str = None

        else:
            # We will retrieve all scores. This will need to be handled during
            # the conversion to numeric.
            res = [pair_and_select(r,colnames,selectcols)
                   for j, r in intersection_row.iterrows()]
            res_str = multisep.join(set([convert_keyvalue_pairs_to_str(r,
                                                                       field_sep=field_sep,
                                                                       kv_sep=kv_sep)
                                        for r in res]))

        processed_res.append(res_str)


    return pd.Series(processed_res)


def get_bedfile_annotations(df_regions, bedfile_path, prefix_chr, colnames,
                            selectcols, field_sep=',', kv_sep=':', multisep='|'):
    """ Intersect regions with bedfile regions,and select columns.
    
    The first step rely on Tabix to query, for each row in the dataframe of
    region, the possible hits from the bedfile. For each position, 0, 1, or
    multiple hits can occur. Each hit value is associated to the column names,
    and a selection is performed.

    Finally for a given position, a series of hit is reported as a string, each
    hit separated by a '|', and for each hit the pairs of colname:value are
    reported, separated by ','.
    """
    df_regs = df_regions.copy()
    # Check that the chromosomes have 'chr' or not:
    if prefix_chr:
        if not df_regs.iloc[0,0].startswith('chr'):
            df_regs.iloc[:,0] = 'chr' + df_regs.iloc[:,0].astype(str)
    else:
        if df_regs.iloc[0,0].startswith('chr'):
            df_regs.iloc[:,0] = df_regs.iloc[:,0].astype(str).str.strip('chr')


    intersection = bedfile_intersect(df_regs, bedfile_path)

    # Deal with colnames and selected columns. In the end I want a string such
    # as: colname:value
    processed_res = []
    for i,res in enumerate(intersection):
        if res is None: processed_res.append(res)
        else:
            # Pair values with colnames, and select which columns to keep
            # Now accepting multiple hits, separated by '|' (which is <multisep>)
            res = [pair_and_select(r,colnames,selectcols) for r in res]
            res_str = multisep.join(set([convert_keyvalue_pairs_to_str(r,
                                                                       field_sep=field_sep,
                                                                       kv_sep=kv_sep)
                                        for r in res]))

            processed_res.append(res_str)
            
    return pd.Series(processed_res)


def bedfile_intersect(df_regions, bed_file_path):
    """ Annotate positions in dataframe with bedfile content (tabix indexed).

    CAUTION: only works with tabix indexed bedfiles.
    Using the 'tabix' library, queries are made for each interval in the
    dataframe.

    IMPORTANT NOTE: tabix is inclusive in the manner it queries the file.
    It means that a position represented by:
        chr1    15  16
    will be associated to both the intervals:
        chr1    10  15
        chr1    16  20

    For each query, results are evaluated so that if a query did not match a
    region, it is associated to 'None'.

    Args:
        df_regions(dataframe): dataframe with regions to annotate.
        bed_file_path(str): path to bedfile ('.tbi' file should be in the same
                            directory)
    Returns:
        list of annotations. eg [["chr","start","end","state"]] for 1 region.

    """
    # Query with tabix library.
    bed_tabix = tabix.open(bed_file_path)
    list_chrom_tbx = tabix_list_chrom(bed_file_path)
    try:
        list_chrom_tbx = [v.decode('utf8') for v in list_chrom_tbx]
    except:
        list_chrom_tbx = list_chrom_tbx

    all_results = []
    for index,row in df_regions.iloc[:,range(0,3)].iterrows():
        if row.chrom in list_chrom_tbx:
            query_res = bed_tabix.query(row.iloc[0],
                                        row.iloc[1]+1, # here +1 is necessary because tabix considers END included with "query" command.
                                        row.iloc[2])
            res = list(query_res)
            # Need to evaluate whether query returned something or not.
            if res:
                # Multi-hits are taken into account!
                all_results.append(res)
            else:
                all_results.append(None)
        else:
            all_results.append(None)

    return all_results
    

# bigwig related functions
# ------------------------

def get_bigwig_annotations(df_regions, bigwig_file_path, prefix_chr):
    """ Annotate positions in a dataframe with value from bigwig file.

    Dataframe should contain the "chrom","start", and "end" information. For
    each line, the bigwig file is queried, and the value is stored in a list.

    In term of speed, it barely matters to query the file at once or make a
    query for each region.

    Please check how missing values are handled by default in the bigwig file.
    Chromosome naming schemes ("chrN vs "N") should be handled.

    Args:
        df_regions(dataframe): columns are "chrom","start","end"
        bigwig_file_path(str): path to bigwig file to use.

    Returns:
        list: retrieved values from the bigwig file.
    """

    bwFile = pbw.open(bigwig_file_path)
    values = []
    

    for name,region in df_regions.iloc[:,range(0,3)].iterrows():
        reg = region.copy()
        if reg['chrom'].startswith('chr'):
            if not prefix_chr:
                reg["chrom"] = reg["chrom"].strip("chr")
        else:
            if prefix_chr:
                reg["chrom"] = 'chr'+reg["chrom"]
                
        # It might be that the chromosome is unavailable for annotations.
        # In this case, the script should not fail. NaN values are returned for
        # such positions.
        if reg['chrom'] not in bwFile.chroms().keys():
            values.append(np.NaN)
            continue

        try:
            query_res = bwFile.values(*region)
            values.append(query_res[0])

        except RuntimeError as e:
            bwFile.close()
            raise type(e)(sys.exc_info()+" {}:{}-{}".format(*reg))

    bwFile.close()
    # This was a mistake: the missing values should be taken care of during
    # conversion to numeric, not here.
    #values = pd.Series(values).replace(np.nan,0)
    values = list(itt.chain(values))
    return pd.Series(values)

def get_bigwig_annotations_mean(df_regions, bigwig_file_path, prefix_chr, mean_size=10):
    """ Annotate positions with mean value from values in bigwig file.

    Dataframe should contain the "chrom","start", and "end" information. For
    each line, the bigwig file is queried, and the value is stored in a list.

    In term of speed, it barely matters to query the file at once or make a
    query for each region.

    Please check how missing values are handled by default in the bigwig file.
    Chromosome naming schemes ("chrN vs "N") should be handled.

    Args:
        df_regions(dataframe): columns are "chrom","start","end"
        bigwig_file_path(str): path to bigwig file to use.
        mean_size(int): number of bases to consider (default=5)

    Returns:
        list: retrieved values from the bigwig file.
    """
    chromsizes = pbt.chromsizes('hg19')

    bwFile = pbw.open(bigwig_file_path)
    values = []

    for name,region in df_regions.iloc[:,range(0,3)].iterrows():
        reg = region.copy()
        chromsize_max = chromsizes[region.chrom][1]

        if reg['chrom'].startswith('chr'):
            if not prefix_chr:
                reg["chrom"] = reg["chrom"].strip("chr")
        else:
            if prefix_chr:
                reg["chrom"] = 'chr'+reg["chrom"]
                
        # It might be that the chromosome is unavailable for annotations.
        # In this case, the script should not fail. NaN values are returned for
        # such positions.
        if reg['chrom'] not in bwFile.chroms().keys():
            values.append(np.NaN)
            continue

        try:
            start = reg.start-int(np.floor(mean_size/2))
            start = start if start>=0 else 0
            end = reg.end+int(np.ceil(mean_size/2))
            end = end if end < chromsize_max else chromsize_max

            query_res = bwFile.values(reg[0],start,end)
            values.append(np.nanmean(query_res))

        except RuntimeError as e:
            print("CAUGHT AN ERROR")
            bwFile.close()
            raise type(e)(sys.exc_info()+" {}:{}-{}".format(*reg))

    bwFile.close()
    # This was a mistake: the missing values should be taken care of during
    # conversion to numeric, not here.
    #values = pd.Series(values).replace(np.nan,0)

    values = list(itt.chain(values))
    return pd.Series(values)



# others
# ------

def get_vartype(ref,alt):
    if len(ref)==1:
        if len(alt)==1:
            return "SNV"
        else:
            return "INS"
    else:
        if len(alt)==1:
            return "DEL"
        else:
            return "INDEL"

def get_vartrans(ref,alt):
    if len(ref)==1 and len(alt)==1:
        # First check that we work with known nucleotides.
        if not ref in ('A','C','G','T'):
            return 'unknown'
        if not alt in ('A','C','G','T'):
            return 'unknown'

        if ref=='A':
            if alt=='G':
                return 'transition'
            else:
                return 'transversion'

        elif ref=='C':
            if alt=='T':
                return 'transition'
            else:
                return 'transversion'

        elif ref=='G':
            if alt=='A':
                return 'transition'
            else:
                return 'transversion'

        elif ref=='T':
            if alt=='A':
                return 'transition'
            else:
                return 'transversion'
        else:
            # unknown bases
            # Should be handled above.
            return 'unknown'
    
    else:
        return 'not_SNV'


def expand_regions(regions, start):
    """Return dataframe with one row per base in a set of regions.

    Given a list of regions elements (defined by chromosome, start, and end
    positions, plus the "ref" and "alt" alleles), this function expands each
    region and store each 1-base interval in a dataframe.

    This dataframe thus contain the columns: "chrom", "start", "end", "ref",
    and "alt".

    Args:
        regions(list): a list of "Record" elements, with "chrom", "start",
                       "end", "ref", and "alt" fields.

        start(int): idx of the first row from the original dataset (when
                    working with chunks.

    Returns:
        DataFrame: 5 columns, 1-base intervals on each row.

    """

    columns_varinfo=["chrom","start","end","id",
                     "ref","alt","row_id","vartype","vartrans"]

    list_pos = []
    for row_id, reg in enumerate(regions, start):
        for position in range(reg.start, reg.end):
            list_pos.append([reg.chrom,
                             position,
                             position+1,
                             reg.id,
                             reg.ref,
                             reg.alt,
                             reg.chrom+'_'+str(row_id), # <chrom>_<index>
                             reg.vartype,
                             reg.vartrans])
    #
    expanded_df = pd.DataFrame(list_pos,
                               columns=columns_varinfo)
    return expanded_df


def get_seqcontext_bed(df, dict_pysam, chrom_sizes):
    """ Returns a dict {'seqcontext':pd.Series} with 3bp genomic sequences.

    For each [chrom, start, end] in df, fetch the associated sequence (no
    mask). Only the triplet composed of the base, its left-flank base, and its
    right-flank base are returned per position.

    If a position is not found in the regular chromosomes, "None" is returned.

    In:
        df (pd.DataFrame): dataframe with at least the 3 first columns being
                           'chrom','start','end'
        dict_pysam (dict): dictionary mapping chromosomes to fasta files as
                           pysam.FastaFile objects
    """

    all_results = []
    for chrom, chunk in df.iloc[:,[0,1,2]].groupby('chrom'):
        try:
            pysam_fasta = dict_pysam[chrom]
        except KeyError:
            print("Chromosome not found: {} ; Seq context assigned to None".format(chrom))
            res =pd.Series([None for _ in range(chunk.shape[0])], index=chunk.index.values)
            all_results.append(res)
            continue

        if str(chrom).startswith('chr'):
            bt_df = pbt.BedTool.from_dataframe(chunk.assign(chrom=chunk.loc[:,'chrom'].str.strip('chr')
                                              ).iloc[:,[0,1,2]])
        else:
            bt_df = pbt.BedTool.from_dataframe(chunk.iloc[:,[0,1,2]])
        # Now extend the position to a three bases window.
        bed_df = bt_df.slop(b=1,g=chrom_sizes).to_dataframe()
        bed_df.iloc[:,0] = bed_df.iloc[:,0].astype(str)
        # And now retrieve the fasta sequence.
        res = []
        for v in bed_df.values:
            res.append(pysam_fasta.fetch(*v))


        res = pd.Series(res,index=chunk.index.values)
        all_results.append(res)

    # Now merge all results, and get them into a dict structure
    return pd.concat(all_results).sort_index().values


def format_variant_info(df_regions):
    """ Copy an input variant df and add a UCSC link to each.
    """
    ucsc_website = "https://genome.ucsc.edu/cgi-bin/hgTracks?"\
                   "db=hg19&position={0}%3A{1}-{2}"

    ucsc_links = [ucsc_website.format(r.chrom,r.start,r.end)
                   for n,r in df_regions.iterrows()] 

    # Here each variant is identified by its index. Multiple table will group
    # related information together.
    variants_info = df_regions.copy()
    #variants_info["id"] = variants_info.index 
    variants_info["ucsc_gb"] = ucsc_links

    # Here convert the dataframe to an ordered dictionary: each variant is a
    # dictionary with columns as keys.
    #variants_info = variants_info.apply(OrderedDict, axis=1)

    return variants_info


def apply_annot_function(name, df_regions, args):
    """ Return a pandas DataFrame or pandas Series.
    """
    print(name)
    try:
        res = args[0](df_regions, *args[1:])
        if len(res.shape) > 1:
            res.set_axis([name+'.'+str(c) for c in res.columns],
                         axis=1,
                         inplace=True)
        else:
            res.rename(name, inplace=True)

        return res
    except Exception:
        print("ERROR for annotation '{}'".format(name))
        print(sys.exc_info())


def annotate_regions(df_regions, full_processing_params, n_cores):
    """ Main function for the annotation of regions.

    """

    # Now each args set is associated to a node to be processed.
    pool = Pool(n_cores)
    all_annotations = pool.starmap(apply_annot_function,
                           ((name, df_regions, funargs)
                           for name, funargs in full_processing_params.items())
                            )
    pool.close()
    pool.join()

    # Annotations are returned as {column_name:annotations}. First unflat the
    # list into a single dictionary, then store into the dataframe the
    # annotations, with the correct order.
    # unflat_all_annotations = {}
    # [unflat_all_annotations.update(el) for el in all_annotations]

    # for args in list_args:
    #     column_name = args[0]
    #     variants_annot[column_name] = unflat_all_annotations[column_name]

    all_annotations = pd.concat(all_annotations, axis=1)
    all_annotations.index = df_regions.index.values

    return all_annotations 

# ORDER OF COLUMNS BETWEEN VCF AND BED NOT CONSERVED: SCORE IN VCF IS AFTER
# ALT, SCORE IN BED IS BEFORE. HAVE TO CHECK IF IT'S CONSISTENT.
# ALSO, BED DOES NOT CONSERVE GENOTYPES OF PATIENTS
def create_record_vcf(line,Record):
    chrom, end, varid = line.split("\t")[:3]
    #try:
    #    end = int(end)
    #except ValueError:
    #    continue # likely: header without a '#'
    chrom = str(chrom)
    end = int(end)

    variant_refalt_info = line.split("\t")[3:5]

    vartype = get_vartype(*variant_refalt_info)
    vartrans = get_vartrans(*variant_refalt_info)

    start = end -1
    if vartype == 'DEL':
        # We want to score the totality of the region deleted.
        end = start + len(variant_refalt_info[0])+1
    elif vartype == 'INS':
        # We want to score both the position start and end position
        # between which the insertion happened.
        end = start + 2
    elif vartype == 'INDEL':
        # We are going to score it as a deletion...
        end = start + len(variant_refalt_info[0])+1


    if chrom[:3] != "chr":
        chrom = "chr"+chrom

    variant_pos_info = [chrom,start,end,varid]

    # Reference and Alt values
    variant_info = variant_pos_info + variant_refalt_info + [vartype, vartrans]
    myvariant = Record(*variant_info)
    return myvariant

def create_record_bed(line,Record):
    variant_pos_info = line.split("\t")[:3]

    variant_pos_info[1:3] = map(try_int, variant_pos_info[1:3])

    # Reference and Alt values
    try:
        variant_refalt_info = line.split("\t")[5:7]
    except:
        variant_refalt_info = ['.','.']

    if variant_pos_info[0][:3] != "chr":
        variant_pos_info[0] = "chr"+variant_pos_info[0]
    
    try:
        variant_id = line.split("\t")[3]
    except:
        variant_id = '.'

    variant_info = variant_pos_info + [variant_id] + variant_refalt_info + ['.']
    myvariant = Record(*variant_info)
    return myvariant


def read_variants_from_bed(bed_file_path, Record):
    """ Return list of Reccord NamedTuple from the provided bedfile.

    A 'Record' named tuple is defined as a ("chrom","start","end","ref",
    "alt") tuple.

    Args:
        bed_file_path(str): path to bedfile to parse

    Returns:
        list of Record named tuple.
    """
    variant_positions = []

    ext = bed_file_path.split('.')[-1]
    if ext=='gz':
        open_file = gzip.open
    else:
        open_file = open


    with open_file(bed_file_path, "r") as pf:
        for line in pf:
            line = line.strip("\n")
            if line[0] == "#": continue
            my_variant = create_record_bed(line, Record)
            variant_positions.append(myvariant)

    return variant_positions


def read_variants_from_vcf(vcf_file_path, Record):
    """ Return list of Reccord NamedTuple from the provided vcffile.

    A 'Record' named tuple is defined as a ("chrom","start","end","id","ref",
    "alt") tuple.

    Args:
        vcf_file_path(str): path to vcffile to parse

    Returns:
        list of Record named tuple.
    """
    variant_positions = []

    ext = vcf_file_path.split('.')[-1]
    if ext=='gz':
        open_file = gzip.open
    else:
        open_file = open

    #with vcf.Reader(filename=vcf_file_path) as pf:
    with open_file(vcf_file_path,"r") as pf:
        for line in pf:
            try:
                line = line.decode("utf8")
            except:
                line = line

            if line[0] == "#":
                continue

            myvariant = create_record_vcf(line,Record)
            variant_positions.append(myvariant)

    return variant_positions


def create_args_parser():
    parser = ap.ArgumentParser(epilog=__doc__,
                               formatter_class=ap.RawDescriptionHelpFormatter)

    parser.add_argument("-i",
                        "--input",
                        type=str,
                        help="",
                        required=True)

    parser.add_argument("-o",
                        "--output",
                        type=str,
                        help="",
                        required=True)


    parser.add_argument("-it",
                        "--input_type",
                        type=str,
                        choices=['bed','vcf'],
                        help="",
                        required=True)


    parser.add_argument("-cfg",
                        "--config_file",
                        type=str,
                        help="Config file with all defined resources.",
                        required=True
                       )

    parser.add_argument('-gs',
                        '--get_sequence_context',
                        type=str,
                        choices=['False','True'],
                        default='False'
                        )

    parser.add_argument("-c",
                        "--n_cores",
                        type=int,
                        help="",
                        required=False,
                        default=4)

    parser.add_argument("-t",
                        "--temp_dir_path",
                        type=str,
                        help="Path to directory where to write temporary files.",
                        required=True
                        )

    parser.add_argument("-cs",
                        "--chunksize",
                        type=int,
                        help="",
                        required=False,
                        default=50000)
    return parser


def check_resource_config(resource_dict, name, resource_dir):
    """ Check that all parameters are present in a resource config dict.

    For a given resource dictionary with file paths, names, filetypes, etc.,
    check that the expected structure is respected, in terms of parameters
    content.

    For instance, assert that the number of provided files and parameters
    (name, type) is the same.

    If bedfiles are provided, assert that the column names and selections are
    defined.
    """
    counts = dict(n_files = len(resource_dict['files']),
                  n_names = len(resource_dict['names']),
                  n_ft = len(resource_dict['filetypes']),
                  n_chr = len(resource_dict['chrPrefix']))

    assert len(set(counts.values()))==1,\
            ("Error: different counts in parameters for {}:\n{}"
            ).format(name, '\n'.join('\t{}: {}'.format(k,v)
                                     for k,v in counts.items()))

    for f in resource_dict['files']:
        fp = resource_dir+'/'+f
        if not os.path.isfile(fp):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),fp)

    if 'options' in resource_dict:
        for k in resource_dict['options'].keys():
            if k != 'all':
                assert k in resource_dict['names'],\
                        ('Option name "{}" has no match in provided '
                         'list for resource "{}"').format(k, name)

    if 'Bed' in resource_dict['filetypes']:
        assert 'colnames' in resource_dict.sections,\
                '"colnames" missing in {}'.format(name)

        assert 'keepcols' in resource_dict.sections,\
                '"keepcols" missing in {}'.format(name)

        # Check that column names are available for the correct files.
        bool_bednames = np.array(resource_dict['filetypes'])=='Bed'
        bednames = np.array(resource_dict['names'])[bool_bednames]

        for colcat in ['colnames', 'keepcols']:
            assert all([bedname in resource_dict[colcat]
                        for bedname in bednames]),\
                    ('One or more bed files have no defined "{}" in {}'
                    ).format(colcat, name)
        

def process_list_bool(array):
    map_bool = {k:True for k in ('True','true','1',1)}
    map_bool.update({k:False for k in ('False','false','0',0)})


    return [map_bool[v] for v in array]

        
def process_config(configfile):
    """ Associate functions to all resources in config
    """
    # This is the full list of parameters that will be processed during the
    # annotation.
    full_processing_params = collections.OrderedDict()
    key_to_remove = []

    conf = configobj.ConfigObj(configfile)

    # conf should be in the form of a dictionary.
    datadir = conf['GENERAL']['datadir']
    ordercat = conf['GENERAL']['ordercols']

    # This will be used for each resource.
    params_keys = ('files','names','filetypes', 'chrPrefix')

    # Those are the main general categories, such as CONSERVATION.
    for category in ordercat:
        cat_config = conf[category]

        # Complete the dirpath.
        cat_dir = datadir + cat_config['dir']

        # Those are the subgroups of annotations, such as PHYLOP.
        for resource in cat_config.sections:
            resource_config = cat_config[resource]
            resource_dir = cat_dir + resource_config['dir']

            check_resource_config(resource_config, resource, resource_dir)

            # Now store in the full_processig all the names, files, and
            # options.
            list_params = [resource_config[k] for k in params_keys[:-1]]
            list_params.append(process_list_bool(resource_config[params_keys[-1]]))
            paired_params = zip(*list_params) 

            for fpath, name, filetype, prefix_chr in paired_params:
                fpath = resource_dir + '/' + fpath

                # Identify the function that will be used for annotation.
                # Add the parameter tuple to the full processing list.
                if filetype == 'BigWig':
                    func = get_bigwig_annotations
                    annot_params_tuple = (func, fpath, prefix_chr)
                    full_processing_params[name] = annot_params_tuple

                elif filetype == 'Bed':
                    func = get_bedfile_annotations
                    cols = resource_config['colnames'][name]
                    keepcols = process_list_bool(resource_config['keepcols'][name])

                    kv_sep = resource_config.get('keyvalue_sep',':')
                    field_sep = resource_config.get('field_sep',',')

                    # NOTE: positional arguments is not optimal ; see annotToNum
                    annot_params_tuple = (func, fpath, prefix_chr,
                                          cols,
                                          keepcols,
                                          field_sep,
                                          kv_sep
                                         )
                    full_processing_params[name] = annot_params_tuple

                if 'options' in resource_config:
                    options = resource_config['options']

                    # First process file-specific options, such as those which
                    # remove annotation (for instance: gencode genes should not
                    # be intersected, so they have the option "noIntersect",
                    # which will revert the "full_processing_params.append"
                    # operation above.
                    if (name in options) or ('all' in options):
                        for opt in options.get(name,[])+options.get('all',[]):
                            if opt == 'noIntersect':
                                # Remove default operation in processing_params
                                key_to_remove.append(name)
                                continue

                            elif opt.startswith('mean'):
                                # An option like mean_<N> corresponds to a
                                # bigwig mean annotation.
                                if '_' in opt:
                                    size = int(opt.split('_')[1])
                                else:
                                    size = 10

                                func = get_bigwig_annotations_mean
                                newname = name+'.mean.{}'.format(size)
                                annot_params_tuple = (func,
                                                      fpath,
                                                      prefix_chr,
                                                      size)

                            elif opt == 'getClosest':
                                func = bedtools_closest_features
                                newname = name+'.closest'
                                annot_params_tuple = (func,
                                                      fpath,
                                                      prefix_chr,
                                                      cols,
                                                      keepcols)

                            elif opt.startswith('getMultiHit'):
                                if '_' in opt:
                                    multisep = opt.split('_')[1]
                                else:
                                    multisep = '|'

                                func = get_bedfile_annotations
                                newname = name+'.multi'
                                annot_params_tuple = (func,
                                                      fpath,
                                                      prefix_chr,
                                                      cols,
                                                      keepcols,
                                                      field_sep,
                                                      kv_sep,
                                                      multisep)

                            full_processing_params[newname] = annot_params_tuple

    for k in key_to_remove:
        del full_processing_params[k]

    return full_processing_params


# Main and arguments
# ------------------

# Only consider assembled autosomes and sexual chromosomes.
chroms_chr = ['chr'+str(i) for i in range(1, 23)]+['chrX','chrY']
chroms = [chrom.strip('chr') for chrom in chroms_chr] 

Record = collections.namedtuple("Record",
            ["chrom","start","end","id","ref","alt","vartype", "vartrans"])


# Those might be exported in the config file...
fasta_path = "/kingdoms/dyogen/workspace5/RegulationData/hg19/genome_fasta/Homo_sapiens.GRCh37.75.dna.chromosome.{}.fa.gz"

chromsizes = '/kingdoms/dyogen/workspace5/RegulationData/hg19/hg19.chrom.sizes'



def main():
    parser = create_args_parser()
    args = parser.parse_args()

    # First : set-up temporary directory for bedtools operations.
    base_tmp_dir = args.temp_dir_path

    if not os.path.exists(base_tmp_dir): os.makedirs(base_tmp_dir)
    # This will create a randomly named directory, isolated from any other
    # directory within the base dir.
    tmp_dir = tempfile.mkdtemp(dir=base_tmp_dir)+'/'
    if not os.path.exists(tmp_dir): os.makedirs(tmp_dir)
    pbt.set_tempdir(tmp_dir)
    pbt.cleanup(remove_all=True)




    if eval(args.get_sequence_context) == True:
        dict_pysam = {chrom: pysam.FastaFile(fasta_path.format(str(chrom).strip('chr')))
                      for chrom in chroms_chr}


    args.output = args.output if args.output.endswith('.gz') else args.output+'.gz'
    print("Starting script: {}".format(datetime.datetime.now()))
    print("- Reading input file: {}".format(datetime.datetime.now()))

    file_type = args.input_type
    if file_type == "vcf":
        variants_regions = read_variants_from_vcf(args.input, Record)

    else:
        variants_regions = read_variants_from_bed(args.input, Record)

    if os.path.exists(args.output):
        os.remove(args.output)

    # Read the configuration file, and get all parameters for annotation.
    full_processing_params = process_config(args.config_file)


    # Main function that annotates variants.
    #annotated_variants_chunks = []
    chunksize = args.chunksize
    chunkidx = list(range(0,len(variants_regions),chunksize))

    print("- Annotation of chunks")
    print(("(Nb: {} ; total number of postion to annotate: {})"
          ).format(len(chunkidx), len(variants_regions)))

    print("(writting to {})".format(args.output))
    
    for nb,i in enumerate(chunkidx):
        print("\tchunk {}/{}".format(nb,len(chunkidx)))
        chunk_regions = variants_regions[i:i+chunksize]
        header = True if nb==0 else False

        # First step: create a dataframe with regions expanded to 1-base
        # intervales.
        print("\tExpanding regions: {}".format(datetime.datetime.now()))

        df_regions = expand_regions(chunk_regions, start=i)


        # retrieve variant info: pos, ref, alt, etc.
        print("\tGet annotations: {}".format(datetime.datetime.now()))
        variants_annot = format_variant_info(df_regions)

        
        if eval(args.get_sequence_context) == True:
            variants_annot['seqcontext'] = get_seqcontext_bed(df_regions,
                                                              dict_pysam,
                                                              chromsizes)

        # Get the annotations associated to positions.
        annotations_chunk = annotate_regions(df_regions, full_processing_params,
                                             args.n_cores)

        print("\tAnnotation finished: {}".format(datetime.datetime.now()))

        # Merge with the dataframe of basic annotations (variant type, etc.)
        annotated_variants_chunk = pd.concat([variants_annot,
                                              annotations_chunk]
                                            ,axis=1)

        print("\tMerged table: {}".format(datetime.datetime.now()))
        # We remove lines where the variants falls in undefined regions.
        regular_chrom_bool = annotated_variants_chunk.chrom.isin(chroms+chroms_chr)
        if sum(~regular_chrom_bool)>0:
            print(("\tNote: {:,} variants are excluded because outside of "
                   " assembled autosomes, or X, or Y").format(sum(~regular_chrom_bool)))

        # NOTE: NOT removed on the 20190211 ; if removed, this impact the
        # annotation conversion process (most of my annotations are defined
        # only for regular chroms)
        annotated_variants_chunk = annotated_variants_chunk.loc[
                                          regular_chrom_bool, :]

        if args.get_sequence_context is True:
            # Remove also variants with 'N' in the seq context, when the option
            # is true.
            seq_context_has_N = annotated_variants_chunk.seqcontext.str.contains('N')
            if sum(seq_context_has_N.replace(np.nan,True))>1:

                print(("\tNote: {:,} variants are excluded because their "
                       "sequence context contained masked nucleotides."
                       ).format(sum(seq_context_has_N.replace(np.nan,True))))


        print("\tStarting writting the chunk to file: {}".format(datetime.datetime.now()))
        annotated_variants_chunk.to_csv(args.output,sep="\t",
                                        header=header,index=False,
                                        mode="a",compression='gzip')

        print("\tDone writting the chunk to file: {}".format(datetime.datetime.now()))
        
    print("All done: {}".format(datetime.datetime.now()))
    pbt.cleanup(remove_all=True)
    return 0


###############################################################################
# MAIN

if __name__ == "__main__":
    return_code = main()
    pbt.cleanup(remove_all=True)
    os.rmdir(tmp_dir)
    sys.exit(return_code)
