import numpy as np
import astropy
from astropy.coordinates import SkyCoord
import astropy.units as u
import ugali.utils.stats
from ugali.utils.stats import Samples
import warnings; warnings.filterwarnings('ignore')

def mod_to_kpc(mod):
    d_pc = 10**((mod+5)/5) 
    return d_pc / 1000


def calculate_separations_and_sys_error(chainfile, ra_median, dec_median, mod_median):
  
    print('-'*75)
    chain = np.load(chainfile)

    dhelio_kpc_samples = mod_to_kpc(chain['distance_modulus'])
    dist_samples = np.array(dhelio_kpc_samples, dtype = [('dkpc',float)])
    dist_samples = Samples(dist_samples.view(np.recarray))

    distance_kpc, [distance_kpc_lower, distance_kpc_upper] = dist_samples.peak_interval('dkpc',burn=0,clip=10,alpha=0.32)
    distance_kpc_stat_err_upper = distance_kpc_upper - distance_kpc
    distance_kpc_stat_err_lower = distance_kpc - distance_kpc_lower
    print('Heliocentric Distance Stat. Error Interval:', round(distance_kpc,0), [-1*round(distance_kpc_stat_err_lower,0), round(distance_kpc_stat_err_upper,0)])
    
    mod_plus_sys_error  = mod_median + 0.1
    mod_minus_sys_error = mod_median - 0.1

    dist = mod_to_kpc(mod_median);
    dist_upper = mod_to_kpc(mod_plus_sys_error)
    dist_lower = mod_to_kpc(mod_minus_sys_error)

    coord_nominal = SkyCoord(ra_median*u.deg,dec_median*u.deg,distance=dist*u.kpc)
    coord_upper   = SkyCoord(ra_median*u.deg,dec_median*u.deg,distance=dist_upper*u.kpc)
    coord_lower   = SkyCoord(ra_median*u.deg,dec_median*u.deg,distance=dist_lower*u.kpc)

    ## Heliocentric Distance sys errors
    dhelio_sys_err_upper = dist_upper - dist
    dhelio_sys_err_lower = dist - dist_lower
    print('Heliocentric Distance Sys. Errors:', [-1*round(dhelio_sys_err_lower,0), round(dhelio_sys_err_upper,0)])


################################ Galactocentric Distance #######################################
    GC = SkyCoord(266.4168262*u.deg,-29.0077969*u.deg,distance=8.178*u.kpc) # distance from GRAVITY Collaboration (2019)

    ## Galactocentric Distance sys errors
    dgc_nominal = coord_nominal.separation_3d(GC).value 
    dgc_given_upper = coord_upper.separation_3d(GC).value
    dgc_given_lower = coord_lower.separation_3d(GC).value

    dgc_sys_err_upper = dgc_given_upper - dgc_nominal
    dgc_sys_err_lower = dgc_nominal - dgc_given_lower

## stat errors
    coord_nominal_all_distances = SkyCoord(ra_median*u.deg,dec_median*u.deg,distance= dhelio_kpc_samples*u.kpc)
    dgc_separation_samples = coord_nominal_all_distances.separation_3d(GC).value
    dgc_sep_samples = np.array(dgc_separation_samples, dtype = [('dgc_sep_kpc',float)])

    dgc_sep_samples = Samples(dgc_sep_samples.view(np.recarray))
    dgc,[dgc_lower, dgc_upper] = dgc_sep_samples.peak_interval('dgc_sep_kpc',burn=0,clip=10,alpha=0.32)
    dgc_stat_err_upper = dgc_upper - dgc
    dgc_stat_err_lower = dgc - dgc_lower

    print('Galactocentric Distance Median and Stat. Error Interval:', round(dgc,0), [-1*round(dgc_stat_err_lower,0), round(dgc_stat_err_upper,0)])
    print('Galactocentric Distance Sys. Errors:', [-1*round(dgc_sys_err_upper,0), round(dgc_sys_err_lower,0)])

################################ LMC Distance #######################################
    LMC = SkyCoord(80.8939*u.deg,-69.7561*u.deg,distance=49.97*u.kpc) #Pietrzynski 2013

# sys errors
    dlmc_nominal = coord_nominal.separation_3d(LMC).value
    dlmc_given_upper = coord_upper.separation_3d(LMC).value
    dlmc_given_lower = coord_lower.separation_3d(LMC).value

    dlmc_sys_err_upper = dlmc_given_upper - dlmc_nominal
    dlmc_sys_err_lower = dlmc_nominal - dlmc_given_lower

## stat errors
    dlmc_separation_samples = coord_nominal_all_distances.separation_3d(LMC).value
    dlmc_sep_samples = np.array(dlmc_separation_samples, dtype = [('dlmc_sep_kpc',float)])

    dlmc_sep_samples = Samples(dlmc_sep_samples.view(np.recarray))
    dlmc, [dlmc_lower, dlmc_upper] = dlmc_sep_samples.peak_interval('dlmc_sep_kpc',burn=0,clip=10,alpha=0.32)
    dlmc_stat_err_upper = dlmc_upper - dlmc 
    dlmc_stat_err_lower = dlmc - dlmc_lower

    print('LMC Distance Median and Stat. Error Interval', round(dlmc,0), [-1*round(dlmc_stat_err_lower,0), round(dlmc_stat_err_upper,0)])
    print('LMC Distance Sys.  Errors', [-1*round(dlmc_sys_err_lower,0), round(dlmc_sys_err_upper,0)])

################################ LMC Distance #######################################
    SMC = SkyCoord(12.54*u.deg,-73.11*u.deg,distance= 62.44*u.kpc) # Graczyk et al (2020), doi: 10.3847/1538-4357/abbb2b

    # sys errors
    dsmc_nominal = coord_nominal.separation_3d(SMC).value
    dsmc_given_upper = coord_upper.separation_3d(SMC).value
    dsmc_given_lower = coord_lower.separation_3d(SMC).value

    dsmc_sys_err_upper = dsmc_given_upper - dsmc_nominal
    dsmc_sys_err_lower = dsmc_nominal - dsmc_given_lower

    ## stat errors
    dsmc_separation_samples = coord_nominal_all_distances.separation_3d(SMC).value
    dsmc_sep_samples = np.array(dsmc_separation_samples, dtype = [('dsmc_sep_kpc',float)])

    dsmc_sep_samples = Samples(dsmc_sep_samples.view(np.recarray))
    dsmc, [dsmc_lower, dsmc_upper] = dsmc_sep_samples.peak_interval('dsmc_sep_kpc',burn=0,clip=10,alpha=0.32)
    dsmc_stat_err_upper = dsmc_upper - dsmc
    dsmc_stat_err_lower = dsmc- dsmc_lower 
    print('SMC Distance Median and Stat. Error Interval',round(dsmc,0), [-1*round(dsmc_stat_err_lower,0), round(dsmc_stat_err_upper,0)])
    print('SMC Distance Sys. Errors:', [-1*round(dsmc_sys_err_upper,0), round(dsmc_sys_err_lower,0)])
    print('SMC Projected Angular Separation (deg)', round(coord_nominal.separation(SMC).deg,0))


    return [round(dist,0),  round(dhelio_sys_err_upper,0), round(dhelio_sys_err_lower,0)], \
           [round(dlmc,0), round(dlmc_stat_err_upper,0), round(dlmc_stat_err_lower,0), round(dlmc_sys_err_upper,0), round(dlmc_sys_err_lower,0)], \
           [round(dgc,0), round(dgc_stat_err_upper,0), round(dgc_stat_err_lower,0), round(dgc_sys_err_upper,0), round(dgc_sys_err_lower,0)], \
           [round(dsmc,0), round(dsmc_stat_err_upper,0), round(dsmc_stat_err_lower,0), round(dsmc_sys_err_upper,0), round(dsmc_sys_err_lower,0)]




