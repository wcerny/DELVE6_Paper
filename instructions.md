## Instructions for Reproducing Table 1 and All Figures [Under Construction]

### Part A: Setting up the Conda Environment
A1. Git clone this entire repository.

A2. Migrate to the repository directory using the terminal, and use the command: conda env create -f environment.yml

A3. Hopefully that works... if not, try manually pip installing each package in that file or using git clone for ChainConsumer and ugali


### Part B: Setting up isochrone files
\texttt{ugali} requires isochrone files to be placed in a specific directory in order to use some of its functionalities. Rather than providing the complete isochrone grid, here we provide only two small files needed to reproduce our analysis. These are located in the /isochrones/ directory. The easiest way to get ugali to recognize the provided files is to create a symlink to the directory provided here, and place it in a high level directory ~/.ugali. To do so, do as follows:

B1. Change directory to your home directory (cd ~)

B2. Make a directory called ".ugali"

B3. Use the command "ln -s [path to isochrone folder that is part of this repository] ." 
  
### Part C: Setting up isochrone files


