# FINSURF 


 FINSURF is a ...

 FINSURF is implemented as ...


If you use FINSURF, please cite:

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
  - [Authors](#authors)
  - [License](#license)
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

## Usage

### Setting up your working environment for SCORPiOs

Before any FINSURF run, you should:
 - go to FINSURF root folder,
 - activate the conda environment with `conda activate finsurf`.

### Running FINSURF on example data

Before using FINSURF on your data, we recommend running a test with our example data to ensure that installation was successful and to get familiar with the pipeline, inputs and outputs.

#### Example 1: Simple FINSURF run

FINSURF uses a YAML configuration file to specify inputs and parameters for each run.
An example configuration file is provided: [config_example.yaml](config_example.yaml). This configuration file executes FINSURF on toy example data located in [data/example/](data/samples/), that you can use as reference for input formats.

The only required snakemake arguments to run SCORPiOs are `--configfile` and the `--use-conda` flag. Optionally, you can specify the number of threads via the `--cores` option. For more advanced options, you can look at the [Snakemake documentation](https://snakemake.readthedocs.io/en/stable/).

To run FINSURF on example data:

```
snakemake --configfile config_example.yaml --use-conda --cores 4
```

The following output should be generated:
`FINSURF_examples/FINSURF_output_0.txt`.


## Authors
* [**Lambert Moyon**](mailto:lambert.moyon@bio.ens.psl.eu)
* **Alexandra Louis**
* **Nga Thi Thui Nguyen**
* **Camille Berthelot**
* **Hugues Roest Crollius**

## License

This code may be freely distributed and modified under the terms of the GNU General Public License version 3 (GPL v3)
- [LICENSE GPLv3](LICENSE.txt)

## References

