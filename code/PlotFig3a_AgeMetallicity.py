import glob 
import numpy as np
import chainconsumer
from chainconsumer import ChainConsumer
import matplotlib
import matplotlib.pyplot as plt


## set RCparams.. should migrate to style file
plt.rc('text',usetex= True)
plt.rc('font', family = 'serif')
plt.rc('axes', linewidth = 2)
plt.rc('xtick', direction = 'in', labelsize = 16); plt.rc('xtick.major', size = 6, width = 1.); plt.rc('xtick.minor', size = 3, width = 0.5);
plt.rc('ytick', direction = 'in', labelsize = 16); plt.rc('ytick.major', size = 6, width = 1.); plt.rc('ytick.minor', size = 3, width = 0.5);


def z2feh(z):
    '''
    A function to convert metallicities (Z) to solar-scaled iron abundances [Fe/H]. 
    Procedure is taken rom Table 3 and Section 3 of Bressan et al. 2012 and Confirmed in Section 2.1 of Marigo et al. 2017.
    This function has been scavenged from the ugali package.
    '''

    Z_init  = z                # Initial metal abundance
    Y_p     = 0.2485           # Primordial He abundance (Komatsu 2011)
    c       = 1.78             # He enrichment ratio 

    Y_init = Y_p + c * Z_init 
    X_init = 1 - Y_init - Z_init

    Z_solar = 0.01524          # Solar metal abundance
    Y_solar = 0.2485           # Solar He abundance (Caffau 2011)
    X_solar = 1 - Y_solar - Z_solar

    return np.log10(Z_init/Z_solar * X_solar/X_init)


# load chains and store relevant subset
full_chains = np.load('../result_files/delve6_may_mcmc.npy')
age_samples = full_chains['age']
metal_samples = full_chains['metallicity']

# convert samples in Z to samples in [Fe/H]
FeH_samples = [z2feh(m) for m in metal_samples]

# re-stack into a format useable by ChainConsumer
data = np.vstack([age_samples,FeH_samples]).T

# bulk of plotting 
c = ChainConsumer()
c.add_chain(data, parameters=[r"$\tau$ (Gyr)", r"[Fe/H]"])
c.configure(summary = False, plot_hists = True,  sigma2d=True, plot_point = True,  contour_labels = "sigma", serif = True, sigmas=[1, 2, 3], smooth = 2, diagonal_tick_labels= False,usetex = True, max_ticks = 9, tick_font_size = 14, label_font_size = 17)
fig = c.plotter.plot(figsize= (4,4))
ax = plt.gca(); ax.minorticks_on()

# stylistic parameters
a1,a2,a3,a4 = fig.get_axes()
a3.minorticks_on()
a3.set_yticks([-2.1,-1.9,-1.7,-1.5,-1.3,-1.1])
a3.set_xticks([8.,9.,10.,11.,12.,13.])
a3.set_ylim(-2.193, -1.0)
a4.set_ylim(-2.193, -1.0)

# save 
fig.savefig('Fig3a_AgeMetallicity.pdf',bbox_inches = 'tight')
