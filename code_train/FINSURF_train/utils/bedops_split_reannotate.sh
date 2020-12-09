#! /bin/bash
set -o nounset
set -o errexit

usage() {
    echo "Usage: ./$(basename $0) <input_bed> <output_bed> <name_delim>" ;
    exit 1 ;
}


if [[ ! $# -eq 3 ]] ; then
    usage ;
fi

# This should be the bed file to split regions from.
inputfile=$1 ;
outputfile=$2 ;
multidelim=$3 ;


bedops --partition $inputfile | \
    awk -F "\t" '{ if ($2<$3) {print $0} }' | \
    bedtools sort -i - | \
    bedmap --echo \
           --echo-map-id \
           --delim "\t" \
           --multidelim $multidelim \
           - \
           $inputfile > $outputfile ;
