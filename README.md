# Code and Data for DELVE 6 Discovery Paper
Repository maintained by William Cerny (william.cerny@yale.edu), on behalf of the DELVE Collaboration (https://delve-survey.github.io/)

Last updated: June 2, 2023


## Overview of Contents: 

1. instructions.md: Detailed, step-by-step instructions to reproduce tables and figures
2. /result_files/: Probabilistic membership catalog and MCMC chains from $\texttt{ugali}$
3. /code/: Codes used to extract quantitative results from MCMC chain and generate figures 
4. /figures/: Files for the figures in the manuscript (.pdf or .png format)
5. /catalogs/: Selected catalog data from DELVE DR2 used for characterization analysis
6. /raw_all/: Miscellaneous files from $\texttt{ugali}$ and $\texttt{simple}$, needed for codes
7. /isochrones/: Isochrone files needed to reproduce figures 
8. environment.yml: conda environment file 
9. Properties.tex: LaTeX file used to construct Table 1.

*Interested in only the member catalog? Look for the "delve6_may_mcmc.fits" file in /raw_all/.* 

## Setup 
To be able to re-generate the figures and results from the paper, you will first need to (1) download and activate the provided conda (Python) environment file, (2) download the MCMC chains, (3) move the provided isochrone files to the directory ugali is expecting. All the steps necessary to carry out these tasks are described in the file "instructions.md".

## Dependencies and Other Software: 
The results from this work are primarily based on results derived from the use of two open-source software packages: $\texttt{simple}$ and $\texttt{ugali}$. Here, we generally provide only the outputs of these software packages, and refer the reader to these packages' respective repositories for information on their use.

Other dependencies of the codes in this repository include, but are not limited to: $\texttt{astropy}$, $\texttt{numpy}$, $\texttt{scipy}$, $\texttt{ChainConsumer}$, $\texttt{healpy}$,  $\texttt{matplotlib}$, $\texttt{fitsio}$, $\texttt{jax}$, and more. The environment.yml file above specifies the necessary dependencies (see instructions.md for details).



## Authorship Disclaimer: 
WRC makes no claim of authorship and/or originality to the codes presented herein. Codes presented here include contributions from many authors, both within and outside of the DELVE Collaboration. This repository and its code is provided exclusively for the purpose of reproducing the results/figures presented in the manuscript.
