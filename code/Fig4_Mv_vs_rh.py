import os
import glob
import sys
import numpy as np
import scipy.ndimage
import matplotlib;matplotlib.use('Agg')
from matplotlib import rc;rc('text', usetex=True);rc('font', family='serif')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker as mticker
import fitsio as fits
import healpy as hp
import yaml
import pandas as pd 
from astropy import units as u

## stylistic
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['axes.labelsize'] = 'x-large'
plt.rc('axes', lw = 1.5)

## Define helper functions for lines of constant surface brightness 
def distanceToDistanceModulus(distance):
    """ Return distance modulus for a given distance (kpc).
    Parameters
    ----------
    distance : distance (kpc)
    Returns
    -------
    mod : distance modulus
    """
    return 5. * (np.log10(np.array(distance)) + 2.)

def surfaceBrightnessContour(surface_brightness):
    c_v = 19.78
    distance = np.linspace(1e-2, 1e3, 2) # kpc
    r_angle = 1.
    r_physical = np.tan(np.radians(r_angle)) * distance * 1000.
    abs_mag = surface_brightness - distanceToDistanceModulus(distance) - c_v - 2.5 * np.log10(r_angle**2)
    return r_physical, abs_mag


## Load in Dwarf Galaxy Datafile; flag those with resolved velocity dispersions
dsph = pd.read_csv('./LiteratureData/Dwarf_Properties_June2023.csv')
vdisp_resolved = np.isfinite(dsph['vlos_sigma'])


## Load in Ultra-Faint Star Cluster dataset; remove a few objects that are possible false-positives or brighter GCs
ufsc = pd.read_csv('./LiteratureData/UFSC_Properties_Verified_June2023.csv').reset_index()
ufsc = ufsc[ufsc['Pure'] == True] 

print(ufsc['MV_Reported'].values,ufsc['CircRadius_Physical_Derived'])

## Load in Harris GCs, remove duplicates with UFSCs
gc = pd.read_csv('./LiteratureData/GC_Harris_June2023.csv')
gc = gc[~gc['key'].isin(['ko_1','ko_2'])]

## Begin figure, plot the literature data

plt.figure(figsize = (4.5*1.1,4*1.1))
ax = plt.gca()

ax.scatter(dsph['rhalf_sph_physical'][vdisp_resolved], dsph['MV'][vdisp_resolved], marker = '^', edgecolor = 'mediumblue', facecolor = 'mediumblue', zorder = 10, label = 'Dynamically-Confirmed Dwarf Galaxies')
ax.scatter(dsph['rhalf_sph_physical'][~vdisp_resolved], dsph['MV'][~vdisp_resolved], marker = '^', facecolor = 'None', edgecolor = 'mediumblue', zorder = 10,label = 'Candidate Dwarf Galaxies')
ax.scatter(np.asarray(ufsc['CircRadius_Physical_Derived']), np.asarray(ufsc['MV_Reported']), marker = 'o', edgecolor = 'red', facecolor = 'None',zorder = 12, label = 'Recently-Discovered Ultra-Faint Star Clusters')
ax.scatter(gc['rhalf_sph_physical'], gc['MV'], label = 'Classical Globular Clusters', s=25, marker='x', lw=0.5, c='k', rasterized=False)

## Plot DELVE 6
ax.scatter(10,-1.5, s = 250, color = 'yellow', edgecolor = 'black', marker = '*', label = 'DELVE 6', zorder = 999)
ax.errorbar(10,-1.5,xerr = [[3],[4]], yerr=[[-0.6],[0.4]],  color = 'black', ls = 'None',capsize = 1, zorder = 998)

## Plot Lines of Constant Surface Brightness
for s in [24, 26, 28, 30,32]:
    x, y = surfaceBrightnessContour(s)
    ax.plot(x, y, c='gray', ls='--', lw=0.5, zorder= 50)
    ax.text(x[1]/13, y[1]+10.4, s='$\mu = {}$~mag~arcsec$^{{-2}}$'.format(s), color='gray', rotation=39.5, horizontalalignment='right', fontsize=8, zorder=-10)

## Legend, Axes Limits, and Stylistic Parameters
plt.legend(loc = 'upper left',fontsize = 8.1)
ax.tick_params(direction = 'in', which = 'both', labelsize = 13)
ax.tick_params(which = 'major', length = 6, width = 1.2)
ax.tick_params(which = 'minor', length = 3, width = .8, axis = 'both')
ax.minorticks_on()
ax.set_xscale('log')
ax.set_yscale('linear')
ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
ax.set_xlim(0.4, 1300.0)
ax.set_ylim(2.9, -16.0)
ax.set_xlabel(r'$r_{1/2}$ (pc)', fontsize = 13.5)
ax.set_ylabel(r'$M_V$ (mag)', fontsize = 13.5)

## Save Figure
plt.savefig('../figures/Fig4_PopulationComparison.pdf', bbox_inches = 'tight')
