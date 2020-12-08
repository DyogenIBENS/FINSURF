#!/usr/bin/env python

import argparse
import os
import re
import sys
import subprocess

import pandas as pd

import tabix

import utils

from collections import OrderedDict, namedtuple

chroms_chr = ['chr'+str(i) for i in range(1, 23)]+['chrX','chrY']
chroms = [chrom.strip('chr') for chrom in chroms_chr] 

Record = namedtuple("Record",
            ["chrom","start","end","id","ref","alt","vartype", "vartrans"])

colnames = ["chrom", "pos", "start", "end","id","ref","alt","local_var_id","vartype","vartrans","ucsc_link","el_id","genes"]
chunksize = 5000

def tabix_list_chrom(bed_file_path):
    # This function might be necessary to face cases where the chromosome in
    # the bw file is encoded without the "chr" string. In which case variants
    # must be converted ('chr1' => '1')
    process = subprocess.Popen(['tabix','-l',bed_file_path],stdout=subprocess.PIPE) 
    return [chrom.strip() for chrom in process.stdout]

def bedfile_intersect_index(df_regions, bed_file_path):
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
    
    all_results = dict()
    df_range = len(df_regions.columns)
    for index,row in df_regions.iloc[:,range(0,df_range)].iterrows():
        variant_key = "@".join(str(c) for c in row)
        if variant_key not in all_results:
            all_results[variant_key] = []
        if row.chrom in list_chrom_tbx:
            query_str = str(row.iloc[0]) + ":" + str(int(row.iloc[2])+1) + "-" + str(row.iloc[3])
            query_res = bed_tabix.querys(query_str)

            res = list(query_res)
            # Need to evaluate whether query returned something or not.
            if res:
                # Multi-hits are taken into account!
                all_results[variant_key].append(res)
            else:
                all_results[variant_key].append(None)
        else:
            all_results[variant_key].append(None)

    return all_results



def build_reader(input_file, chunksize):
    """ Read a few lines of the input file to build a reader. 

    Two things are checked:
    - whether the first line starts with a "#", indicating a potential header
    - or whether the first line first field is "chrom", indicating again a
      potential header.
    """
    lines = "" 
    try:
        if bool(re.search(".gz$", input_file)):
            lines = pd.read_csv(input_file, compression='gzip', nrows=3, header=None, sep="\t", error_bad_lines=False)
        else:
            lines = pd.read_csv(input_file, nrows=3, sep="\t", header=None)
    except Exception as e:
        return "Error: " + str(e) + "Please see sample for the right file format."
    if lines.shape[1]<5:
        msg = ("Error: in reading your VCF input file: Only {} fields detected, using TAB ; the expected "
               "number should be at least 5."
               ).format(lines.shape[1])

        #raise ValueError(msg)
        return msg
        
    first_cell = lines.iloc[0,0]

    if first_cell.startswith("#") or not (first_cell in chroms_chr or first_cell in chroms):
        # Likely detected the header ; we skip it.
        reader = pd.read_csv(input_file,
                             skiprows=1,
                             chunksize=chunksize,
                             header=None,
                             usecols=list(range(0,5)),
                             names=['chrom','pos','id','ref','alt'],
                             sep="\t")

    else:
        reader = pd.read_csv(input_file,
                             chunksize=chunksize,
                             header=None,
                             usecols=list(range(0,5)),
                             names=['chrom','pos','id','ref','alt'],
                             sep="\t")

    return reader

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

    columns_varinfo=["chrom","pos","start","end","id",
                     "ref","alt","row_id","vartype","vartrans"]

    list_pos = []
    for row_id, reg in enumerate(regions, start):
        try:
            for position in range(reg.start, reg.end):
            
                list_pos.append([reg.chrom,
                    reg.start+1,
                             position,
                             position+1,
                             reg.id,
                             reg.ref,
                             reg.alt,
                             #reg.chrom+'_'+str(row_id), # <chrom>_<index>
                             str(row_id), # NOTE: modified here for SQL indexing.
                             reg.vartype,
                             reg.vartrans])
        except Exception as e:
            return "Error"+ e +": at line " + str(row_id + 1) + ": \n" + str(reg)      
    #
    expanded_df = pd.DataFrame(list_pos,
                               columns=columns_varinfo)
    return expanded_df

def create_record_vcf(row, Record):
    """ Process a row read from a VCF into a Record with additional fields.

    The variant will be assigned START and END values, basing on the POS field
    (which should be the second one) and on the detected variant type (SNV,
    DEL, INS):
    - SNV : start = pos -1 ; end = pos
    - INS : we include the position AFTER the END, as the insertion might be
      disrupting a binding site which important site is located right after.
      (so end = POS +1)
    - DEL : here POS locates the first base that was not removed ; so start =
      POS -1, and end is based on this start position + length of deletion +1.
    - INDEL : processed as deletions as deletions

    In addition, the "chrom" field will be prefixed with a "chr" string if not
    present.
    """
    try:
        chrom, end, varid = list(row.iloc[:3].values)
        chrom = str(chrom)
        end = int(end)
   
        variant_refalt_info = list(row[3:5].values)

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
    except Exception as e:
        return "Error: Problem in your VCF input file at line: " + str(row) + str(e)
    return myvariant

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
        if not ref in ('A','C','G','T','a','c','g','t'):
            return 'unknown'
        if not alt in ('A','C','G','T','a','c','g','t'):
            return 'unknown'
        
        ref = ref.upper()
        alt = alt.upper()

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

def format_variant_info(df_regions):
    """ Copy an input variant df and add a UCSC link to each.
    """
    ucsc_website = "https://genome.ucsc.edu/cgi-bin/hgTracks?"\
                   "db=hg19&position={0}%3A{1}-{2}"

    ucsc_links = [ucsc_website.format(r.chrom,r.start - 100,r.end + 100)
                   for n,r in df_regions.iterrows()] 

    # Here each variant is identified by its index. Multiple table will group
    # related information together.
    variants_info = df_regions.copy()
    variants_info["ucsc_gb"] = ucsc_links

    return variants_info

def run_intersect(score, regulatory, vcf, chunksize, output):
    reader = build_reader(vcf, chunksize=chunksize)
    
    # Check if there's error in reading vcf file
    if type(reader) is str:
        return reader
    
    flatten = lambda l: [item for sublist in l for item in sublist]
    
    result_file = utils.make_tmp_file('result','txt',output)	
    f = open(result_file,"w")	
    f.write("#" + "\t".join(utils.header) + "\n")
    all_results = []
    for i, chunk in enumerate(reader):
        chunk_regions = chunk.apply(lambda row: create_record_vcf(row, Record), axis=1).values # format the variants to the namedTuple structure.
        df_regions = expand_regions(chunk_regions, start=i)
        
        # Check if there's error in VCF file
        if type(df_regions) is str:
            return df_regions
            
        df_regions = format_variant_info(df_regions)
        df_regs = df_regions.copy()
        if not df_regs.iloc[0,0].startswith('chr'): # Check that the chromosomes have 'chr' or not
            df_regs.iloc[:,0] = 'chr' + df_regs.iloc[:,0].astype(str)
		
		### intersect with regulatory gene file
        intersection = bedfile_intersect_index(df_regs, regulatory)
        processed_res = []
        for i,res in enumerate(intersection):
            if intersection[res] == [None]:  continue 
            else:
                variant = res.split("@")
                for y in flatten(intersection[res]):
                    results = variant + y[3:]
                    processed_res.append(results)

            ### intersect with score file
            if len(processed_res) > 0: 
                df_reg2 = pd.DataFrame(OrderedDict(zip(colnames, zip(*processed_res))))
                intersection2 = bedfile_intersect_index(df_reg2, score)
                for i,res in enumerate(intersection2):
                    if intersection2[res] == [None]: continue 
                    else:
                        variant = res.split("@")
                        for y in flatten(intersection2[res]):
                            if variant[9] == "transition":
                                results = [variant[0], variant[1], variant[3], y[5]]
                            else:
                                results = [variant[0], variant[1], variant[3], y[-1]]
                            results += variant[4:7] + variant[8:]
                            if results not in all_results:
                                all_results.append(results)
	## remove temporary file
    #os.remove(vcf)

    all_results = sorted(all_results, key = lambda x: x[3], reverse=True)
    for i in all_results:
        f.write("\t".join(i) + "\n")
    f.close()
    return result_file

def argparser():
    parser = argparse.ArgumentParser(epilog=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-i",
                        "--input",
                        type=str,
                        help="Path to variant file (gzipped or not). vcf format",
                        required=True)
    parser.add_argument("-ig",
                        "--inputgene",
                        type=str,
                        help="Path to gene input file (to filter result)",
                        required=False)

    parser.add_argument("-s",
                        "--score",
                        type=str,
                        help="Path to score file.",
                        required=True)


    parser.add_argument("-g",
                        "--gene",
                        type=str,
                        help="Path to regulatory gene file.",
                        required=True)

    parser.add_argument("-cs",
                        "--chunksize",
                        type=int,
                        help="Number of variants to read as a block from input file.",
                        required=False,
                        default=5000)
    parser.add_argument("-od",
                        "--output_dir",
                        type=str,
                        help="output directory",
                        required=False,
                        default="./res")
    return parser

def main():
	parser = argparser()
	args = parser.parse_args()
	result_file = run_intersect(args.score, args.gene, args.input, args.chunksize, args.output_dir)

	return_str = result_file.strip()
	if args.inputgene and args.inputgene != '':
		return_str += ":" + args.inputgene.strip()
	print(return_str)
	return 0

if __name__ == "__main__":
    return_code = main()
    sys.exit(return_code)

