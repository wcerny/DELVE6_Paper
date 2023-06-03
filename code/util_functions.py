import os 
import scipy 
import glob
import numpy as np
import matplotlib.pyplot as plt, matplotlib.colors as colors
import fitsio as fits
import results_parser
import healpy as hp
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ugali libraries
import ugali.utils.healpix
import ugali.utils.projector
import ugali.utils.plotting
from ugali.isochrone import factory as isochrone_factory
from ugali.analysis import kernel

def cut_isochrone_path(g, r, g_err, r_err, maglim, isochrone):
    """
    Cut to identify objects within isochrone selection.
    """
    
    index_transition = np.nonzero(isochrone.stage >= isochrone.hb_stage)[0][0] + 1

    mag_1_rgb = isochrone.mag_1[0: index_transition] + isochrone.distance_modulus
    mag_2_rgb = isochrone.mag_2[0: index_transition] + isochrone.distance_modulus
    mag_1_rgb = mag_1_rgb[::-1]
    mag_2_rgb = mag_2_rgb[::-1]

    # Cut one way...
    f_isochrone = scipy.interpolate.interp1d(mag_2_rgb, mag_1_rgb - mag_2_rgb, bounds_error=False, fill_value = 999.)
    color_diff = np.fabs((g - r) - f_isochrone(r))
    cut_2 = (color_diff < np.sqrt(0.1**2 + r_err**2 + g_err**2))
     # ...and now the other
    f_isochrone = scipy.interpolate.interp1d(mag_1_rgb, mag_1_rgb - mag_2_rgb, bounds_error=False, fill_value = 999.)
    color_diff = np.fabs((g - r) - f_isochrone(g))
    cut_1 = (color_diff < np.sqrt(0.1**2 + r_err**2 + g_err**2))
    cut = np.logical_or(cut_1, cut_2)

    # Cut for horizontal branch
    mag_1_hb = isochrone.mag_1[isochrone.stage == isochrone.hb_stage][1:] + isochrone.distance_modulus
    mag_2_hb = isochrone.mag_2[isochrone.stage == isochrone.hb_stage][1:] + isochrone.distance_modulus
    f_isochrone = scipy.interpolate.interp1d(mag_2_hb, mag_1_hb - mag_2_hb, bounds_error=False, fill_value = 999.)
    color_diff = np.fabs((g - r) - f_isochrone(r))
    cut_4 = (color_diff < np.sqrt(0.1**2 + r_err**2 + g_err**2))
    f_isochrone = scipy.interpolate.interp1d(mag_1_hb, mag_1_hb - mag_2_hb, bounds_error=False, fill_value = 999.)
    color_diff = np.fabs((g - r) - f_isochrone(g))
    cut_3 = (color_diff < np.sqrt(0.1**2 + r_err**2 + g_err**2))
    cut_hb = np.logical_or(cut_3, cut_4)
    cut = np.logical_or(cut, cut_hb)
    mag_bins = np.arange(17., maglim+0.1, 0.1)
    mag_centers = 0.5 * (mag_bins[1:] + mag_bins[0:-1])
    magerr = np.tile(0., len(mag_centers))
    for ii in range(0, len(mag_bins) - 1):
        cut_mag_bin = (g > mag_bins[ii]) & (g < mag_bins[ii + 1])
        magerr[ii] = np.median(np.sqrt(0.1**2 + r_err[cut_mag_bin]**2 + g_err[cut_mag_bin]**2))

    else:
        return cut



def construct_catalog(catdir, ra, dec, nside = 32):
	'''
	A quick function to generate a single array containing all the catalog data. Not exactly efficient, but we're working with a small data volume.
	Returns a data array.
	'''

	# Select target region via healpix
	pix_nside_select = ugali.utils.healpix.angToPix(nside,ra,dec)
	pix_nside_neighbors = np.concatenate([[pix_nside_select], hp.get_all_neighbours(nside, pix_nside_select)])

	# Construct data array
	infiles = glob.glob(catdir + '/*.fits')	
	data_array = []

	for f in infiles:		
		data_array.append(fits.read(f))

	data = np.concatenate(data_array)
	
	return data 


def generate_data_filter(data, maglim_g0, maglim_r0, iso):
	'''
	A function to generate a boolean mask for carrying out (1) a magnitude limit, (2) star/galaxy separation, and (3) an isochrone filter.
	Returns a boolean array of length "data".
        '''

	try:
		mag_1 = data['WAVG_EXT_CORRECTED_G']
		mag_2 = data['WAVG_EXT_CORRECTED_R']
	except:
		mag_1 = data['WAVG_MAG_PSF_G'] - data['EXTINCTION_G']
		mag_2 = data['WAVG_MAG_PSF_R'] - data['EXTINCTION_R']

	mag_err_1 = data['WAVG_MAGERR_PSF_G']
	mag_err_2 = data['WAVG_MAGERR_PSF_R']

	valid_mags = (mag_1 < maglim_g0) & (mag_2  < maglim_r0)
	stars = (data['EXTENDED_CLASS_G'] <= 2) & (data['EXTENDED_CLASS_G'] >= 0) ## hardcoded star-galaxy separation

	iso_filter = cut_isochrone_path(mag_1, mag_2, mag_err_1, mag_err_2, maglim_g0, iso)
 
	return stars & valid_mags & iso_filter


def get_background_density(data, maglim_g0, maglim_r0, inner_radius_deg, outer_radius_deg, angsep_values_deg, data_filter):
	'''
	A function which takes in a catalog and some configuration parameters and returns the field background density.
	To most accurately represent the fitted model, the config parameters should be matched to the ugali config file.

	Returns a float, namely the output background density in units arcmin^-2
	'''

	areal_cut = (angsep_values_deg > inner_radius_deg) & (angsep_values_deg < outer_radius_deg)
	annular_data = data[data_filter & areal_cut]

	return len(annular_data) / (np.pi * (outer_radius_deg**2 - inner_radius_deg**2) * 60**2)  


def measure_density_in_bins(data, maglim_g0, maglim_r0, angsep_values_deg, data_filter, bins):
	'''
	A function to generate the data that goes into a radial profile. User provides the bins, which are assumed to define circular annuli.  
        Outputs bin centers (in deg), densities (in arcmin^-2), and Poisson density errors (in arcmin^-2)
	'''

	centers = 0.5*(bins[1:] + bins[0:-1]) # deg
	area = np.pi*(bins[1:]**2 - bins[0:-1]**2) * 60**2 # arcmin^2

	hist = np.histogram(angsep_values_deg[data_filter], bins=bins)[0] # counts
	densities = hist/area
	density_errors = np.sqrt(hist)/area ## Poisson error

	return centers, densities, density_errors

def plot_densities_and_circularized_profile_model(ax, parameter_obj, bkg_density, centers, densities, density_errors): 
	'''
	A function to plot density datapoints (provided in last three arguments) in addition to a Plummer model (specified by parameter_obj and bkg_density).
	The plotting is performed on a user-specified matplotlib axis object.
	Returns nothing.
	

	Note: although this could use ugali.analysis.source.Source().load(section = 'source') to get the parameters of the best-fit model
	directly from the output .yaml results file, we do not use this here because we would prefer to fix the ellipticity to be 0 for cases
        where we have an upper bound. For this case, we would also want to use the circularized r_{1/2} as opposed to the 
	semi-major axis / extension a_{1/2}, and we can ignore the PA.
	'''

	fine_centers = np.linspace(centers.min(), centers.max(), 1000) ## purely for plotting model as a smooth curve

	ra, dec, circularized_ext, distance = parameter_obj.val('lon'), parameter_obj.val('lat'), parameter_obj.val('extension_radial'), parameter_obj.val('distance')
	kernel = ugali.analysis.kernel.EllipticalPlummer(lon = ra, lat = dec, extension = circularized_ext, ellipticity = 0, position_angle = 0) # all in deg (except ellipticity)
	normalization = parameter_obj.val('npred')
	model_pdf = kernel._pdf(fine_centers)
	radial_label = 'Plummer Profile'


	# plot model using finely-spaced bin centers defined earlier
	ax.plot(fine_centers * 60.0, (model_pdf / 3600.0 * normalization) + bkg_density, c='b', linestyle='-', linewidth=1, label=radial_label, zorder=2, rasterized=False)
	
	# plot background density line 
	ax.axhline(y=bkg_density, c='grey', linestyle='--', linewidth=1, label='Background Field', zorder=1, rasterized=False)

	# plot binned density data 
	ax.errorbar(centers * 60.0, densities, yerr=density_errors, fmt='none', elinewidth=0.5, ecolor='k', zorder=3, rasterized=False)
	ax.scatter(centers * 60.0, densities, s=15, c='k', edgecolor='none', zorder=4, rasterized=False)


	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlim(0.1,11)
	ax.set_ylim(bkg_density / 2.5, densities.max() * 2.5)
	ax.set_ylabel(r'Stellar Density (arcmin$^{-2}$)', fontsize = 15)
	ax.set_xlabel(r'Radius (arcmin)', fontsize = 15)

	return 

def squared_colorbar(mappable):
    '''
    A function to plot square subplots that retain their square shape
    even when plotted with colorbars. 
    Author: Joseph Long; taken from https://joseph-long.com/writing/colorbars/
    '''

    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar, cax
