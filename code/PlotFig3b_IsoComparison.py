import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import gridspec
plt.rc('text', usetex=True); plt.rc('font', family='serif')
plt.rc('axes', lw = 2)
import fitsio as fits 
import ugali.utils.projector
from mpl_toolkits.axes_grid1 import make_axes_locatable

# import ugali modules
from ugali.isochrone import factory as isochrone_factory
import ugali.utils.plotting
import ugali.utils.projector

# import custom modules 
import results_parser
import util_functions

## Filenames
member_cat_filename = '../result_files/delve6_may_mcmc.fits'
parameters_file = '../raw_all/delve6_may_mcmc.yaml'

## Store some useful variables
params = results_parser.Result(parameters_file)
ra = params.val('ra')
dec = params.val('dec')
print(ra,dec)

## Load in necessary data for lefthand panels
member_catalog = fits.read(member_cat_filename)
g0 = member_catalog['WAVG_EXT_CORRECTED_G']
r0 = member_catalog['WAVG_EXT_CORRECTED_R']

## Project data into X/Y coordinates
proj = ugali.utils.projector.Projector(ra, dec)
x, y = proj.sphereToImage(member_catalog['RA'], member_catalog['DEC'])

### Isochrone
iso_old = isochrone_factory(name='Bressan2012', survey='des', age=13.5, z=0.0001,distance_modulus=params.val('distance_modulus'), band_1='g', band_2='r')
iso_new = isochrone_factory(name='Bressan2012', survey='des', age=10, z=0.001,distance_modulus=params.val('distance_modulus'), band_1='g', band_2='r')


#################### Make Figure #####################

fig, (ax) = plt.subplots(ncols=1, figsize = (3.5,4.5))
fig.subplots_adjust(wspace=0.5)

ax.minorticks_on()
ax.tick_params(which = 'major', direction = 'in', length = 5, width = 1, labelsize = 14)
ax.tick_params(which = 'minor', direction = 'in', length = 2.5, width = 0.5)


plt.sca(ax)
ax.set_xlim(-0.5,1.5); ax.set_ylim(23.8,16)
ax.set_xticks([-0.5,0,0.5,1,1.5])

ugali.utils.plotting.drawIsochrone(iso_old, lw=1.2, color='mediumblue', zorder=1, label = r'[Fe/H] = -2.2, 13.5 Gyr')
ugali.utils.plotting.drawIsochrone(iso_new, lw=1.2, color='red', zorder=1, label = r'[Fe/H] = -1.2, 10 Gyr')
angsep = ugali.utils.projector.angsep(ra, dec, member_catalog['RA'], member_catalog['DEC']) ## angular separations in degrees
angsep_cut = angsep < (2*0.43/60) # cut for < 2*r_{1/2}
ax.scatter(g0[angsep_cut] - r0[angsep_cut], g0[angsep_cut],c = 'black', s = 15, zorder = -1)

## overplot BHB and RHB candidates
bhb = (member_catalog['QUICK_OBJECT_ID'] == 11174900190016)
rhb = (member_catalog['QUICK_OBJECT_ID'] == 11174900190015)
ax.scatter(g0[angsep_cut & bhb] - r0[angsep_cut & bhb], g0[angsep_cut & bhb],c = 'mediumblue', s = 20, zorder = 10,edgecolor = 'black',lw = 0.5)
ax.scatter(g0[angsep_cut & rhb] - r0[angsep_cut & rhb], g0[angsep_cut & rhb],c = 'red', s = 20, zorder = 10,edgecolor = 'black',lw = 0.5)
ax.set_xlabel(r'$(g-r)_0$', fontsize = 15)
ax.set_ylabel(r'$g_0$', fontsize = 15)
ax.annotate('BHB?', xy = (-0.35,19.75), color = 'mediumblue')
ax.annotate('RHB?', xy = (0.17,20.5), color = 'red')

maglim_g0, maglim_r0 = 23.8, 23.8
full_catalog = util_functions.construct_catalog('../catalogs/',ra,dec)
angsep = ugali.utils.projector.angsep(ra, dec, full_catalog['RA'], full_catalog['DEC']) ## angular separations in degrees
## construct and plot magnitude errors
bin_centers = np.arange(16.5,24.0, 0.5)
tuple_bins = [(b - 0.2, b + 0.2) for b in bin_centers]

mag_errors_g0 = []
color_errors = []
for (bin_min, bin_max) in tuple_bins:
    g0 = full_catalog['WAVG_EXT_CORRECTED_G']
    r0 = full_catalog['WAVG_EXT_CORRECTED_R']
    data_selection = (angsep < 1) & (g0 > bin_min) & (g0 < bin_max) & (r0  > bin_min) & (r0 < bin_max) # errors within 30 arcmin
    mag_err_g0 = np.mean(full_catalog[data_selection]['WAVG_MAGERR_PSF_G'])
    mag_err_r0 = np.mean(full_catalog[data_selection]['WAVG_MAGERR_PSF_R'])
    color_err = np.sqrt(mag_err_g0**2 + mag_err_r0**2)
    mag_errors_g0.append(mag_err_g0)
    color_errors.append(color_err)

ax.errorbar([1.3] * len(bin_centers), bin_centers, xerr = color_errors, yerr = mag_errors_g0, ls = 'None', color = 'black')

## save figure
plt.savefig('../figures/Fig3b_IsochroneComparison.pdf', bbox_inches = 'tight')
