import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import gridspec
plt.rc('text', usetex=True); plt.rc('font', family='serif')
plt.rc('axes', lw = 2)
import fitsio as fits 
import ugali.utils.projector
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ugali.isochrone import factory as isochrone_factory
import ugali.utils.plotting
import ugali.utils.projector

## load custom modules
import util_functions
import results_parser

## Filenames
member_cat_filename = '../result_files/delve6_may_mcmc.fits'
parameters_file = '../raw_all/delve6_may_mcmc.yaml'

## Store some useful variables
params = results_parser.Result(parameters_file)
ra = params.val('ra')
dec = params.val('dec')

## Load in necessary data for lefthand panels
member_catalog = fits.read(member_cat_filename)
g0 = member_catalog['WAVG_EXT_CORRECTED_G']
r0 = member_catalog['WAVG_EXT_CORRECTED_R']
highprob = member_catalog['PROB'] > 0.1

## Project data into X/Y coordinates
proj = ugali.utils.projector.Projector(ra, dec)
x, y = proj.sphereToImage(member_catalog['RA'], member_catalog['DEC'])

### Define isochrone to be plotted and used for filtering. Here, assumed to be tau=13.5Gyr and Z = 0.0001 ([Fe/H] = -2.19 for this isochrone model)
iso = isochrone_factory(name='Bressan2012', survey='des', age=13.5, z=0.0001,distance_modulus=params.val('distance_modulus'), band_1='g', band_2='r')

#################### Make Figure #####################
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize = (14.5,14.5/4))
fig.subplots_adjust(wspace=0.5)
for ax in [ax1,ax2,ax3]: 
    ax.minorticks_on()
    ax.tick_params(which = 'major', direction = 'in', length = 5, width = 1, labelsize = 14)
    ax.tick_params(which = 'minor', direction = 'in', length = 2.5, width = 0.5)


### Axis 1: Spatial Distribution plot/map
xyplot_members = ax1.scatter(x[highprob],y[highprob], c = member_catalog['PROB'][highprob], s = 12, cmap = 'plasma', zorder = 0, edgecolor = 'black', lw = .8)
xyplot_nonmems = ax1.scatter(x[~highprob],y[~highprob], c = 'lightgrey', s = 10, zorder = -1, alpha = 0.5, rasterized = True)

cb1, cax1 = util_functions.squared_colorbar(xyplot_members)
xyplot_members.set_clim(0.0,1.0)
cb1.set_label('Membership Probability', fontsize = 14)
ax1.set_xlabel(r'$\Delta \alpha_{2000}$ (deg)', fontsize = 15)
ax1.set_ylabel(r'$\Delta \delta_{2000}$ (deg)', fontsize = 15)
ax1.set_xlim(0.12,-0.12); ax1.set_xticks([-0.1,-0.05,0,0.05,0.1])
ax1.set_ylim(-0.12,0.12); ax1.set_yticks([-0.1,-0.05,0,0.05,0.1])

### Axis 2: CMD plot
plt.sca(ax2)
ax2.set_xlim(-0.5,1.5); ax2.set_ylim(23.8,16)

# plot CMD, limited to stars in lefthand panel
in_left_panel = (x > -0.12) & (x < 0.12) & (y > -0.12) & (y < 0.12)
ugali.utils.plotting.drawIsochrone(iso, lw=1, color='k', zorder=1)
cmdplot_members = ax2.scatter(g0[in_left_panel & highprob] - r0[in_left_panel & highprob], g0[in_left_panel & highprob],
                              c = member_catalog['PROB'][in_left_panel & highprob], s = 25, cmap = 'plasma', vmax = 5,
                              zorder = 0, edgecolor = 'black', lw = 1)
cmdplot_members.set_clim(0,5)

cmdplot_nonmems = ax2.scatter(g0[in_left_panel & ~highprob] - r0[in_left_panel & ~highprob], g0[in_left_panel & ~highprob],
                              c = 'lightgrey', s = 20, zorder = -1, alpha = 0.5, rasterized = True)

cb2, cax2 = util_functions.squared_colorbar(cmdplot_members)
cmdplot_members.set_clim(0.0,1.0)
cb2.set_label('Membership Probability', fontsize = 15)
ax2.set_xlabel(r'$(g-r)_0$', fontsize = 15)
ax2.set_ylabel(r'$g_0$', fontsize = 15)


### Axis 3: Radial Density Profile
plt.sca(ax3)
maglim_g0, maglim_r0 = 23.8, 23.8

# generate data cutout, apply basic selections
full_catalog = util_functions.construct_catalog('../catalogs',ra,dec)
angsep = ugali.utils.projector.angsep(ra, dec, full_catalog['RA'], full_catalog['DEC']) ## deg
catalog_filter = util_functions.generate_data_filter(data = full_catalog, maglim_g0 = maglim_g0, maglim_r0 = maglim_r0, iso = iso)

# derive background field density. Magnitude limit and inner/outer radius of background annulus are hardcoded here to match config file
field_density = util_functions.get_background_density(full_catalog, maglim_g0, maglim_r0, 0.5, 1.5, angsep, catalog_filter) 

# define bins and derive densities in those bins
bins = np.logspace(np.log10(0.05 / 60.0), np.log10(12 / 60.0), 15)

centers, densities, density_errors = util_functions.measure_density_in_bins(full_catalog, maglim_g0, maglim_r0, angsep, catalog_filter, bins)

# plot the profile!
util_functions.plot_densities_and_circularized_profile_model(ax3, params, field_density, centers, densities, density_errors) ## no useful return

# derive magnitude errors for CMD (within r = 15 arcmin radius)
bin_centers = np.arange(16.5,24.0, 0.5)
tuple_bins = [(b - 0.1, b + 0.1) for b in bin_centers]

# mag errors for CMD
mag_errors_g0 = []
color_errors = []
for (bin_min, bin_max) in tuple_bins:
	g0 = full_catalog['WAVG_EXT_CORRECTED_G']
	r0 = full_catalog['WAVG_EXT_CORRECTED_R']
	data_selection = (angsep < 15.) & (g0 > bin_min) & (g0 < bin_max) & (r0  > bin_min) & (r0 < bin_max)
	mag_err_g0 = np.mean(full_catalog[data_selection]['WAVG_MAGERR_PSF_G'])
	mag_err_r0 = np.mean(full_catalog[data_selection]['WAVG_MAGERR_PSF_R'])
	color_err = np.sqrt(mag_err_g0**2 + mag_err_r0**2)
	mag_errors_g0.append(mag_err_g0)
	color_errors.append(color_err)

# plot magnitude errors at an arbritary color of (g-r)_0 = 1.3
ax2.errorbar([1.3] * len(bin_centers), bin_centers, xerr = color_errors, yerr = mag_errors_g0, ls = 'None', color = 'black')
ax2.set_yticks([17,19,21,23])

plt.savefig('../figures/Fig2_DELVE6_MembershipFigure.pdf', bbox_inches = 'tight')
