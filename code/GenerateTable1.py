import results_parser
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from proper_motion_functions import * 
from calculate_separations import calculate_separations_and_sys_error

# helper function for metallicity
def z2feh(z):
# Taken from Table 3 and Section 3 of Bressan et al. 2012
# Confirmed in Section 2.1 of Marigo et al. 2017
    Z_init  = z                # Initial metal abundance
    Y_p     = 0.2485           # Primordial He abundance (Komatsu 2011)
    c       = 1.78             # He enrichment ratio 

    Y_init = Y_p + c * Z_init 
    X_init = 1 - Y_init - Z_init

    Z_solar = 0.01524          # Solar metal abundance
    Y_solar = 0.2485           # Solar He abundance (Caffau 2011)
    X_solar = 1 - Y_solar - Z_solar

    return np.log10(Z_init/Z_solar * X_solar/X_init)


## Run proper motion MCMC 
members = pd.read_csv('./GaiaData/DELVE6_GaiaMembers.csv')
nwalkers = 200
nburn = 500
nsteps = 2500
nthreads = 4
prior_limits = [-10,10,-10,10]
verbose = True


data_for_pm_MCMC_JAX = jnp.array([members['pmra'].values, members['pmdec'].values, 
            members['pmra_error'].values, members['pmdec_error'].values, members['pmra_pmdec_corr'].values], dtype=jnp.float32)


pm_mcmc_sampler, flat_chain = jax_run_pm_mcmc(data_for_pm_MCMC_JAX, nburn = nburn, nwalkers = nwalkers, nsteps = nsteps,
                                  nthreads = nthreads, limits = prior_limits, verbose = verbose)

pmra_median, pmra_lower, pmra_upper = np.percentile(flat_chain[:,0],[50,16,84])
pmdec_median, pmdec_upper, pmdec_lower = np.percentile(flat_chain[:,1],[50,84,16]) ## flipped because pmdec is negative

## Use 'results.py' module to parse ugali output for parameters that do not require upper/lower limits
may_results = results_parser.Result('../raw_all/delve6_may_mcmc.yaml')

## strings
IAU_Name = may_results.texname
constellation = may_results.constellation

## general numerical results from ugali and the proper motion MCMC 
ra_median, ra_err_upper, ra_err_lower = round(may_results.val('ra'),3), round(may_results.err_upper('ra'),3), round(may_results.err_lower('ra'),3)
dec_median, dec_err_upper, dec_err_lower = round(may_results.val('dec'),3), round(may_results.err_upper('dec'),3), round(may_results.err_lower('dec'),3)
modulus_median, modulus_err_upper, modulus_err_lower = round(may_results.val('distance_modulus'),2), round(may_results.err_upper('distance_modulus'),2), round(may_results.err_lower('distance_modulus'),2)
Mv_median, Mv_err_upper, Mv_err_lower = round(may_results.val('Mv_martin'),2), round(may_results.err_upper('Mv_martin'),2), round(may_results.err_lower('Mv_martin'),2)
PA_median, PA_err_upper, PA_err_lower = round(may_results.val('position_angle'),0), round(may_results.err_upper('position_angle'),0), round(may_results.err_lower('position_angle'),0)
dhelio_median, dhelio_stat_err_upper, dhelio_stat_err_lower =  round(may_results.val('distance'),0), round(may_results.err_upper('distance'),0), round(may_results.err_lower('distance'),0)

rhalf_ang_median, rhalf_ang_err_upper, rhalf_ang_err_lower = round(may_results.val('extension_radial_arcmin'),2), round(may_results.err_upper('extension_radial_arcmin'),2), round(may_results.err_lower('extension_radial_arcmin'),2)

rhalf_phys_median, rhalf_phys_err_upper, rhalf_phys_err_lower = round(may_results.val('physical_size_radial')*1000,0), round(may_results.err_upper('physical_size_radial')*1000,0), round(may_results.err_lower('physical_size_radial')*1000,0)


pmra_median, pmra_err_upper, pmra_err_lower = round(pmra_median,2), round(pmra_upper - pmra_median,2), round(pmra_median - pmra_lower,2)
pmdec_median, pmdec_err_upper, pmdec_err_lower = round(pmdec_median,2), round(pmdec_upper - pmdec_median,2), -1* round(np.abs(pmdec_median) - np.abs(pmdec_lower),2)

## load MCMC chains for parameters needing upper/lower limits 
chain = np.load('../raw_all/delve6_may_mcmc.npy')
ellipticity_upperlim = np.percentile(chain['ellipticity'],95)
age_lowerlim = np.percentile(chain['age'],5)
feh_upperlim = np.percentile(z2feh(chain['metallicity']),95)


## Calculate Separations relative to Magellanic Clouds
[dhelio1, dhelio_sys_err_upper, dhelio_sys_error_lower], \
[dlmc, dlmc_stat_err_upper, dlmc_stat_err_lower, dlmc_sys_err_upper, dlmc_sys_err_lower], \
[dgc, dgc_stat_err_upper, dgc_stat_err_lower, dgc_sys_err_upper, dgc_sys_err_lower], \
[dsmc, dsmc_stat_err_upper, dsmc_stat_err_lower, dsmc_sys_err_upper, dsmc_sys_err_lower] = calculate_separations_and_sys_error('../raw_all/delve6_may_mcmc.npy', ra_median, dec_median, modulus_median)


################################################################ Generate the Table ########################################################################

# Define the table header
table_header = r'\begin{deluxetable}{l c c}' + '\n'
table_header += r'\tablewidth{0pt}' + '\n'
table_header += r'\tabletypesize{\footnotesize}' + '\n'
table_header += r'\tablecaption{\label{tab:d6properties} Properties of DELVE~6}' + '\n'
table_header += r'\tablehead{\colhead{Parameter} & \colhead{Value} & \colhead{Units}}' + '\n'
table_header += r'\startdata' + '\n'

# Define the table rows
table_rows = [
    r'IAU Name & %s & ... \\'%IAU_Name,
    r'Constellation & %s & ...\\'%constellation,
    r'\ra & $%.3f^{+%.3f}_{-%.3f}$ & deg \\'%(ra_median, ra_err_upper, ra_err_lower),
    r'\dec & $%.3f^{+%.3f}_{-%.3f}$ & deg\\'%(dec_median, dec_err_upper, dec_err_lower),
    r'$r_\text{h}$ & $%.2f^{+%.2f}_{-%.2f}$ & arcmin  \\'%(rhalf_ang_median, rhalf_ang_err_upper, rhalf_ang_err_lower),
    r'$r_{1/2}$ & $%d^{+%d}_{-%d}$ & pc  \\'%(rhalf_phys_median, rhalf_phys_err_upper, rhalf_phys_err_lower),
    r'\ellip & $< %.2f$ & ... \\'%ellipticity_upperlim,
    r'\PA & $%d^{+%d}_{-%d}$ & deg \\'%(PA_median, PA_err_upper, PA_err_lower),
    r'$M_V$\tablenotemark{a} & $%.1f^{+%.1f}_{-%.1f}$ & mag \\'%(Mv_median, np.abs(Mv_err_lower), np.abs(Mv_err_upper)), ## flipped ordering because Mv is negative
    r'\age & $> %.1f$ & Gyr \\'%age_lowerlim,
    r'\feh & $< %.2f$ & dex \\'%feh_upperlim,
    r'$(m-M)_0$ & $%.2f^{+%.2f}_{-%.2f} \rm \ (stat.) \ \pm 0.1\tablenotemark{b}$ (sys.) & mag\\'%(modulus_median, modulus_err_upper, modulus_err_lower),
    r'$D_{\odot}$ & $%d^{+%d}_{-%d} \rm \ (stat.) \  ^{+%d}_{-%d} (sys.)$ & kpc\\'%(dhelio_median, dhelio_stat_err_upper, dhelio_stat_err_lower, dhelio_sys_err_upper, dhelio_sys_error_lower),
    r'$D_{\rm GC}$ & $%d^{+%d}_{-%d} \rm \ (stat.) \  ^{+%d}_{-%d} (sys.)$ & kpc\\'%(dgc, dgc_stat_err_upper, dgc_stat_err_lower, dgc_sys_err_upper, dgc_sys_err_lower),
    r'$D_{\rm LMC}$ & $%d^{+%d}_{-%d} \rm \ (stat.) \ ^{+%d}_{-%d} (sys.)$ & kpc\\'%(dlmc, dlmc_stat_err_upper, dlmc_stat_err_lower, dlmc_sys_err_upper, dlmc_sys_err_lower),
    r'$D_{\rm SMC}$ & $%d^{+%d}_{-%d} \rm \ (stat.) \  ^{+%d}_{-%d} (sys.)$ & kpc\\'%(dsmc, dsmc_stat_err_upper, dsmc_stat_err_lower, dsmc_sys_err_upper, dsmc_sys_err_lower),
    r'$E(B-V)$\tablenotemark{c} & 0.036 & mag \\', ## hardcoded to avoid having to store extinction columns in /catalogs/
    r'\hline',
    r'$\mu_{\alpha} \cos \delta$ & $%.2f^{+%.2f}_{-%.2f}$ & mas yr$^{-1}$ \\'%(pmra_median,pmra_err_upper,pmra_err_lower), 
    r'$\mu_{\delta}$ & $%.2f^{+%.2f}_{-%.2f}$ & mas yr$^{-1}$ \\'%(pmdec_median,np.abs(pmdec_err_upper),pmdec_err_lower) ## flipped ordering because pmra is negative
    ]


# Define the table footer
table_footer = r'\enddata' + '\n'
table_footer += r'\tablecomments{Uncertainties for each parameter were derived from the highest-density interval containing 68\% of the marginalized posterior distribution. For the ellipticity, metallicity, and age, the posterior distribution peaked at the boundary of the allowed parameter space. Therefore, we quote the upper, upper, and lower bound for these three parameters (respectively) at 95\% confidence.}' + '\n'
table_footer += r'\tablenotetext{a}{Our estimate of $M_V$ was derived following the procedure from \citet{Martin:2008} and does not include uncertainty in the distance.}' + '\n'
table_footer += r'\tablenotetext{b}{The statistical uncertainty is derived from our \ugali MCMC. We include a systematic uncertainty of $\pm0.1$ associated with isochrone modeling following \citet{2015ApJ...813..109D}. This systematic error is not included in the uncertainty on $r_{1/2}$.}' + '\n'
table_footer += r'\tablenotetext{c}{This $E(B-V)$ value refers to the mean reddening of all sources within $r_{1/2}$, as determined via the maps of \cite{Schlegel:1998} with the recalibration from \citet{2011ApJ...737..103S}}' + '\n'
table_footer += r'\end{deluxetable}'

# Combine all table components
table = table_header + '\n'.join(table_rows) + '\n' + table_footer

# Write the table to a file
filename = '../Properties.tex'
with open(filename, 'w') as file:
    file.write(table)

print('Done saving:', filename)

