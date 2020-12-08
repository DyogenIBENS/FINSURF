# **FINSURF** 

![alt text](./logo_finsurf.png?raw=true "FINSURF")
## Introduction

FINSURF (Functional Identification of Non-coding Sequences Using Random Forests) is a tool designed to analyse lists of sequences variants in the human genome. 

It assigns a score to each variant, reflecting its functional importance and therefore its likelihood to disrupt the physiology of its carrier. FINSURF scores Single Nucleotide Variants (SNV), insertions and deletions. Among SNVs, transitions and transversions are treated separately. 
Insertions are characterised by a score given to each base flanking the insertion point. Deletions are characterised by a score at every deleted base. FINSURF can (optionally) use a list of known or suspected disease genes, in order to restrict results to variants overlapping cis-regulatory elements linked to these genes. 

For a variant of interest, users can generate a graphical representation of "feature contributions », showing the relative contributions of genomic, functional or evolutionary information to its score.



FINSURF is implemented as python3 scripts.

## License

This code may be freely distributed and modified under the terms of the GNU General Public License version 3 (GPL v3)
and the CeCILL licence version 2 of the CNRS. These licences are contained in the files:

1. [LICENSE-GPL.txt](LICENSE-GPL.txt) (or on [www.gnu.org](https://www.gnu.org/licenses/gpl-3.0-standalone.html))
2. [LICENCE-CeCILL.txt](LICENCE-CeCILL.txt) (or on [www.cecill.info](https://cecill.info/licences/Licence_CeCILL_V2-en.html))

Copyright for this code is held by the [Dyogen](http://www.ibens.ens.fr/?rubrique43) (DYnamic and Organisation of GENomes) team
of the Institut de Biologie de l'Ecole Normale Supérieure ([IBENS](http://www.ibens.ens.fr)) 46 rue d'Ulm Paris and the individual authors.

- Copyright © 2020 IBENS/Dyogen : **Lambert MOYON**, Alexandra LOUIS, Thi Thuy Nga NGUYEN, Camille Berthelot and Hugues ROEST CROLLIUS

## Contact

Email finsurf {at} bio {dot} ens {dot} psl {dot} eu

*If you use FINSURF, please cite:*

Moyon Lambert, ....

# Quick start

**Below is a quick start guide to using FINSURF**

## Table of content
  - [Installation](#installation)
    - [Installing conda](#installing-conda)
    - [Installing FINSURF](#installing-finsurf)
  - [Usage](#usage)
    - [Setting up your working environment for FINSURF](#setting-up-your-working-environment-for-finsurf)
    - [Running FINSURF on example data](#running-finsurf-on-example-data)
  - [References](#references)

## Installation

### Installing conda

The Miniconda3 package management system manages all FINSURF dependencies, including python packages and other software.

To install Miniconda3:

- Download Miniconda3 installer for your system [here](https://docs.conda.io/en/latest/miniconda.html)

- Run the installation script: `bash Miniconda3-latest-Linux-x86_64.sh` or `bash Miniconda3-latest-MacOSX-x86_64.sh`, and accept the defaults

- Open a new terminal, run `conda update conda` and press `y` to confirm updates

### Installing FINSURF

- Clone the repository and go to FINSURF root folder
  ```
  git clone https://github.com/DyogenIBENS/FINSURF.git
  cd FINSURF
  ```

- Create the main conda environment.

  We recommend using [Mamba](https://github.com/mamba-org/mamba) for a faster installation:
  ```
  conda install -c conda-forge mamba
  mamba env create -f envs/finsurf.yaml
  ```

  **Alternatively,** you can use conda directly :
  ```
  conda env create -f envs/finsurf.yaml
  ```

- Download feature contributions and gene associations.
  
  You have to download the data files (80Go) that have to be intersect with your variants on <http://opendata.bio.ens.psl.eu/FINSURF/>

  ```
  wget http://opendata.bio.ens.psl.eu/FINSURF/finsurf_data.tar

  tar -xvf finsur_data.tar
  ```

  the architecture of the finsurf directory should then be:
- __FINSURF__
   - [LICENSE.txt](LICENSE.txt)
   - [README.md](README.md)
   - __env__
     - [finsurf.yaml](env/finsurf.yaml)
   - __scripts__
     - [finsurf.py](scripts/finsurf.py)
     - [plot\_contribution.py](scripts/plot_contribution.py)
     - [utils.py](scripts/utils.py)
   - __static__
     - __data__
       - 2020\-05\-11\_table\_genes\_FINSURF\_regions.tsv
       - FINSURF\_REGULATORY\_REGIONS\_GENES.bed.gz
       - FINSURF\_REGULATORY\_REGIONS\_GENES.bed.gz.tbi
       - __FINSURF\_model\_objects__
         - full\-model\_woTargs\_columns.txt
         - rename\_columns\_model.tsv
       - FULL\_FC\_transition.tsv.gz
       - FULL\_FC\_transition.tsv.gz.tbi
       - FULL\_FC\_transversion.tsv.gz
       - FULL\_FC\_transversion.tsv.gz.tbi
       - NUM\_FEATURES.tsv.gz
       - NUM\_FEATURES.tsv.gz.tbi
       - SCALED\_NUM\_FEATURES.tsv.gz
       - SCALED\_NUM\_FEATURES.tsv.gz.tbi
       - scores\_all\_chroms\_1e\-4.tsv.gz
       - scores\_all\_chroms\_1e\-4.tsv.gz.tbi
     - __samples__
       - [gene.txt](static/samples/gene.txt)
       - [variant.vcf](static/samples/variant.vcf)

## Usage

### Setting up your working environment for FINSURF

Before any FINSURF run, you should:
 - go to FINSURF root folder,
 - activate the conda environment with `conda activate finsurf`.

### Running FINSURF on example data

Before using FINSURF on your data, we recommend running a test with our example data to ensure that installation was successful and to get familiar with the pipeline, inputs and outputs.

#### Example 1: Simple FINSURF run



To run FINSURF on example data:

```
scripts/finsurf.py -i static/data/samples/variant.vcf -s static/data/scores_all_chroms_1e-4.tsv.gz -g static/data/FINSURF_REGULATORY_REGIONS_GENES.bed.gz -ig static/data/samples/gene.txt

```

The following output should be generated:
`res/result_*.txt`.




