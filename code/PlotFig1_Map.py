import gala
import matplotlib.pyplot as plt 
import numpy as np
import astropy.units as u
import fitsio as fits, pandas as pd 
from gala.coordinates import MagellanicStreamNidever08
from astropy.coordinates import SkyCoord

plt.rc('font', family = 'serif')
plt.rc('text', usetex = True)
plt.rc('axes', lw = 2)

## Load datafiles
Bica08_main = fits.read('./LiteratureData/Bica2008_All.fits') ## all objects from 2008 catalog 
Bica20_unmatched = pd.read_csv('./LiteratureData/Bica2020_Unique.csv') ## custom file - all objects from 2020 catalog not already in 2008 catalog
Bica20_sats = pd.DataFrame(fits.read('./LiteratureData/Bica2020_ExtendedObjects.fits').byteswap().newbyteorder()) ##

Bica20_ufd = Bica20_sats[Bica20_sats['Type'] == 'UFG']
Bica20_ufsc = Bica20_sats[Bica20_sats['Type'] == 'UFC']

## some dwarfs were flagged by Bica2020 as possible star clusters; undo that
misclassified = ['Tuc V, Tucana V, DES J2337-6316',
                 'Phe II, Phe 2, Phoenix II, DES J2339.9-5424',
                 'Pic I, Pictor I, Pictor 1, DES J0443.8-5017'
                ]

disk = ['OGLL 845, Gaia 3','OGLL 874, DES 5','OGLL 863, DES 4']


## fix misclassified objects
Bica20_ufsc = Bica20_ufsc[~np.isin(Bica20_ufsc['Names'],misclassified)].reset_index()
Bica20_sats_reclassified = Bica20_sats[np.isin(Bica20_sats['Names'],misclassified)]
Bica20_ufd = pd.concat([Bica20_ufd,Bica20_sats_reclassified]).reset_index()


## transform to Magellanic Stream coordinates for all systems
Bica08_Cel = SkyCoord(ra = Bica08_main['RAJ2000'] * u.deg, dec = Bica08_main['DEJ2000'] * u.deg)
Bica20_Cel = SkyCoord(ra = Bica20_unmatched['RAJ2000_1'] * u.deg, dec = Bica20_unmatched['DEJ2000_1'] * u.deg)
Bica20_ufsc_Cel =  SkyCoord(ra = Bica20_ufsc['RAJ2000'] * u.deg, dec = Bica20_ufsc['DEJ2000'] * u.deg)
Bica20_ufd_Cel =  SkyCoord(ra = Bica20_ufd['RAJ2000'] * u.deg, dec = Bica20_ufd['DEJ2000'] * u.deg)

DELVE2_Cel = SkyCoord(ra = 28.77 * u.deg, dec = -68.25 * u.deg)
HydrusI_Cel = SkyCoord(ra = 37.389 * u.deg,dec =  -79.3089 * u.deg)
YMCA1_Cel = SkyCoord(ra = 110.8378 * u.deg, dec = -64.8319* u.deg)
Delve6_Cel = SkyCoord(ra = 33 * u.deg, dec = -66 * u.deg)
SMCNOD_Cel = SkyCoord(ra = 12.00 * u.deg,dec = -64.80 * u.deg)

ms_nidever08 = MagellanicStreamNidever08()

Bica08_MS = Bica08_Cel.transform_to(ms_nidever08)
Bica20_MS = Bica20_Cel.transform_to(ms_nidever08)
Bica20_ufsc_MS = Bica20_ufsc_Cel.transform_to(ms_nidever08)
Bica20_ufd_MS = Bica20_ufd_Cel.transform_to(ms_nidever08)


DELVE2_MS = DELVE2_Cel.transform_to(ms_nidever08)
HydrusI_MS = HydrusI_Cel.transform_to(ms_nidever08)
YMCA1_MS = YMCA1_Cel.transform_to(ms_nidever08)
Delve6_MS = Delve6_Cel.transform_to(ms_nidever08)
SMCNOD_MS = SMCNOD_Cel.transform_to(ms_nidever08)


## make figure
plt.figure(figsize = (6,6))
ax = plt.gca()

plt.xlim(-40,20)
plt.ylim(-30,30)

## plot Bica 2008, 2020 clusters
plt.scatter(Bica08_MS.L, Bica08_MS.B, color = 'black', s = 2, rasterized = True)
plt.scatter(Bica20_MS.L, Bica20_MS.B, color = 'black', s = 2, label = 'Bica (2008, 2020) Clusters', rasterized = True)

## plot ultra-faint dwarfs from Bica 2020
for index, row in Bica20_ufd.iterrows():
    cel_coordinate = SkyCoord(ra = row['RAJ2000'] * u.deg, dec = row['DEJ2000'] * u.deg)
    ms_coordinate   = cel_coordinate.transform_to(ms_nidever08)
    name = row['Names'].split(',')[0]

    del_x = 2.2
    del_y = 1

    if name == 'Ret II':
        del_x = 3

    if name == 'Ret III':
        del_y = -3.5

    if name == 'Pic II':
        del_x = 0

    if name == 'Tuc V':
        del_y = -2.5
        del_x = 2.4

    if name == 'Tuc IV':
        del_y = -2.8
        del_x = .3

    if name == 'Tuc III':
        del_x = 4
        del_y = 1.45

    if name == 'Car III':
        name = 'CarII+III'

    if index == 0:
        plt.scatter(ms_coordinate.L.deg, ms_coordinate.B.deg, color = 'mediumblue', marker = '^',  s = 30, label = 'Ultra-Faint Dwarf Galaxies')
    else:
        plt.scatter(ms_coordinate.L.deg, ms_coordinate.B.deg, color = 'mediumblue', marker = '^',  s = 30)
    ax.annotate(text = name, xy = (ms_coordinate.L.deg + del_x, ms_coordinate.B.deg+del_y),
                                                         xycoords = 'data', color = 'mediumblue')


## plot ultra-faint star clusters from Bica 2020
for index, row in Bica20_ufsc.iterrows():
    cel_coordinate = SkyCoord(ra = row['RAJ2000'] * u.deg, dec = row['DEJ2000'] * u.deg)
    ms_coordinate   = cel_coordinate.transform_to(ms_nidever08)
    name = row['Names'].split(',')[0]

    del_x = 2.2
    del_y = 1

    if name == 'Torrealba 1':
        name = 'To 1'
        del_x = .25

    if name == 'DES 1':
        del_x = 5

    if name == 'OGLL 845':
        del_y = -2.5
        del_x = 8

    if name == 'OGLL 863':
        del_y = 1.
        del_x = 2.85

    if name == 'OGLL 874':
        del_x = -.65
        del_y = 0.2

    if name == 'SMASH 1':
        del_x = 3.5
        del_y = -2.15


    if index == 0:
         plt.scatter(ms_coordinate.L.deg, ms_coordinate.B.deg, color = 'red', marker = 'o',  s = 30, label = 'Ultra-Faint Star Clusters')
    else:
        plt.scatter(ms_coordinate.L.deg, ms_coordinate.B.deg, color = 'red', marker = 'o',  s = 30)
    ax.annotate(text = name, xy = (ms_coordinate.L.deg + del_x, ms_coordinate.B.deg+del_y),
                                                         xycoords = 'data', color = 'red')

## plot a handful of extra targets
plt.scatter(DELVE2_MS.L, DELVE2_MS.B, color = 'mediumblue', marker = '^',  s = 30)
ax.annotate(text = 'DELVE 2', xy = (DELVE2_MS.L.deg-1, DELVE2_MS.B.deg - 0.75)
            , xycoords = 'data', color = 'mediumblue')

plt.scatter(YMCA1_MS.L, YMCA1_MS.B, color = 'red', s = 25)
ax.annotate(text = 'YMCA-1', xy = (YMCA1_MS.L.deg+3.25, YMCA1_MS.B.deg+0.9), xycoords = 'data', color = 'red')


plt.scatter(HydrusI_MS.L, HydrusI_MS.B, color = 'mediumblue', marker = '^',  s = 30)
ax.annotate(text = 'Hyi I', xy = (HydrusI_MS.L.deg+1.9, HydrusI_MS.B.deg+1.2)
            , xycoords = 'data', color = 'mediumblue')


plt.scatter(SMCNOD_MS.L,SMCNOD_MS.B, color = 'peru', marker = '.',  s = 220)
ax.annotate(text = 'SMCNOD', xy = (SMCNOD_MS.L.deg - 0.3, SMCNOD_MS.B.deg+0.8)
            , xycoords = 'data', color = 'peru')


plt.scatter(Delve6_MS.L, Delve6_MS.B, facecolor = 'yellow', edgecolor = 'black', s = 300, lw = 1, marker = '*')
ax.annotate(text = 'DELVE~6', xy = (Delve6_MS.L.deg + 4, Delve6_MS.B.deg+2),
            xycoords = 'data', color = 'black', fontsize = 13)



## Format and Aesthetic Parameters
plt.gca().invert_xaxis()
plt.xlabel(r'$L_{\rm MS}$ (deg)', fontsize = 20); plt.ylabel(r'$B_{\rm MS}$ (deg)', fontsize = 20)

ax.minorticks_on()
ax.tick_params(which = 'major', direction = 'in', length = 9, right = True,
               top = True, width = 1.5, labelsize = 17)
ax.tick_params(which = 'minor', direction = 'in', length = 4.5, right = True,
               top = True, width = 1.5, labelsize = 17)

[label.set_y(-.005) for label in ax.xaxis.get_majorticklabels()]
[label.set_x(-.005) for label in ax.yaxis.get_majorticklabels()]

plt.legend(loc = 'lower left', fontsize = 12.)


## Save Figure
plt.savefig('../figures/Fig1_Map.pdf', bbox_inches = 'tight')



