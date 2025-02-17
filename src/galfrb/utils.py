# standard libraries 
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import h5py
import os
import sys
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy.interpolate import interp2d, RegularGridInterpolator
from scipy.special import beta as sp_beta
from scipy.stats import ks_2samp
sign = lambda x: 2*np.heaviside(x, 0.5) - 1 # define sign function
from astropy.cosmology import FlatLambdaCDM # https://docs.astropy.org/en/stable/cosmology/index.html
from astropy.cosmology import z_at_value 
cosmo = FlatLambdaCDM(H0=70, Om0=0.3) # define the cosmology to be used in this script


# constants
Lum_g_sun = 0.13 # times the solar luminosity (bolometric)
Lum_r_sun = 0.15 # times the solar luminosity. I computed using the r-band effective values: lamba = 6180 A and Delta lambda = 1200 A
Mg_sun    = 5.11 # AB magnitude
Mr_sun    = 4.65 # AB magnitude
## Source paper : https://iopscience.iop.org/article/10.3847/1538-4365/aabfdf

# load the Neural Network of leja+22 which computes the proabability density function in the parameter space logmstar - logsfr - redshift
# Add the 'libs' directory to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "libs"))

rltv_pth = os.path.dirname(os.path.abspath(__file__))

from sfr_leja.code.sample_nf_probability_density import sample_density, load_nf, threedhst_mass_completeness

# Load data used in Sharma et al. (2024) 
# includes 26 DSA FRBs + 26 from previous surveys
from read_transients_data import read_frbs_hosts_data


def flux_to_mag(flux: float, zeropoint: float = -48.6) -> float:
    """
    Convert flux in Jy to AB magnitude.

    https://en.wikipedia.org/wiki/AB_magnitude
    https://www.star.bris.ac.uk/~mbt/topcat/sun253/Fluxes.html

    Input:
        - flux : Flux density measured in Jy
        - zeropoint : Normalization constant (default: -48.6)
    
    Returns:
        - mag : AB magnitude
    """
    return 2.5 * (23 - np.log10(flux)) + zeropoint

def Lg_Lg_sun_to_Lg_L_sun(Lg_in_Lg_sun: float, ratio: float = Lum_g_sun) -> float:
    """
    Change luminosity units from Lg_sun to L_sun 
    """
    return ratio * Lg_in_Lg_sun # convert to Lsun

def Lr_Lr_sun_to_Lr_L_sun(Lr_in_Lr_sun: float, ratio: float = Lum_r_sun) -> float:
    """
    Change luminosity units from Lr_sun to L_sun 
    """
    return ratio * Lr_in_Lr_sun # convert to Lsun

def Lg_L_sun_to_Lg_Lg_sun(Lg_in_L_sun: float, ratio: float = Lum_g_sun) -> float:
    """
    Change luminosity units from L_sun to Lg_sun 
    """
    return (1./ratio) * Lg_in_L_sun # convert to Lsun

def Lr_L_sun_to_Lr_Lr_sun(Lr_in_L_sun: float, ratio: float = Lum_r_sun) -> float:
    """
    Change luminosity units from L_sun to Lr_sun 
    """
    return (1./ratio) * Lr_in_L_sun # convert to Lsun


def load_input_data(fits_file: str = None, 
                    drop_zero_flux_gal: str = True):
    """
    Load SDSS+WISE input data from Chang et al. catalog

    Input:
        - fits_file : path to datafile
        - drop_zero_flux_gal : flag to drop out data of negative/zero flux

    Returns:
        - redshift : the redshift of each galaxy
        - flux_g : Flux in g-band [Jy]
        - flux_r : Flux in r-band [Jy]
    """

    if (fits_file==None) : fits_file = rltv_pth + "/data/sw_input.fits"
    # Open the fits file using a context manager
    with fits.open(fits_file) as hdul:

        #hdul.info()
        data = hdul[1].data
        header = hdul[1].header
        
        # Optionally display the data and header
        #print(data)
        #print(header)

        #print(data.columns.names)
        
        # load data of interest
        redshift = data['redshift']
        flux_g = data['flux_g']
        flux_r = data['flux_r']
        
        if drop_zero_flux_gal :
            ind = flux_r > 0
            redshift = redshift[ind]
            flux_g = flux_g[ind]
            flux_r = flux_r[ind]
            # 45 galaxies have no information about flux in r filter !
            
    return redshift, flux_g, flux_r

def available_outputs() :
    # Get the list of items in the current directory
    items = os.listdir('./output')

    # Print a header for clarity
    print("Available output directories:\n" + "-"*75)

    # Print each item on a new line
    for idx, item in enumerate(items, start=1):
        print(f"{idx}. {item}")
    print("-"*75)

    return items


def bootstrap_data(data: list = [], bins: list = [], iters: int = 1000, conf_interval: float = 95., seed: int = 42): 
    '''
    Bootstrapping parameters:
    - data: raw data
    - bins: list of bin edges
    - iters: number of bootstrap iterations
    - conf_interval: confidence interval percentage
    - seed: seed for random number generator
    '''
    rng = np.random.default_rng(seed)  # Create a reproducible random generator
    bootstrap_cdfs = np.zeros((iters, len(bins)))

    # Perform bootstrapping
    for i in range(iters):
        sample = rng.choice(data, size=len(data), replace=True)
        cdf_values = [np.mean(sample <= x) for x in bins]
        bootstrap_cdfs[i, :] = cdf_values

    # Calculate mean CDF and confidence intervals
    cdf_mean = np.mean(bootstrap_cdfs, axis=0)
    ci_lower = np.percentile(bootstrap_cdfs, (100 - conf_interval)/2., axis=0)
    ci_upper = np.percentile(bootstrap_cdfs, 50 + conf_interval/2., axis=0)

    return cdf_mean, ci_lower, ci_upper


def make_basic_plots(z_dist=True, 
                     rmag_dist=True, 
                     z_rmag=True ) :

    test_redshift, test_flux_g, test_flux_r = load_input_data()

    # redshift distribution
    if z_dist :
        plt.figure()
        _ = plt.hist(test_redshift, bins=300, density=True, histtype='step')
        plt.xlabel("redshift")
        plt.ylabel("PDF")
        plt.text(0.7,3.,s=f"- Number of galaxies: ${len(test_redshift)}$\n" + \
            f"- Number of galaxies with negative redshift: ${len(test_redshift[test_redshift < 0])}$\n" + \
            f"- Number of galaxies with redshift $z > 0.2$: ${len(test_redshift[test_redshift > 0.2])}$\n" + \
            f"- Number of galaxies with redshift $z > 1.0$: ${len(test_redshift[test_redshift > 1.])}$\n" + \
            f"- Minimum redshift: ${min(test_redshift):.5f}$\n" + \
            f"- Maximum redshift: ${max(test_redshift):.3f}$\n" + \
            f"- Mean redshift: ${np.mean(test_redshift):.3f}$\n", color='gray', fontsize=18)
        plt.show()

    # r-mag distribution
    if rmag_dist :
        plt.figure()
        test_rmag_obs = flux_to_mag(flux=test_flux_r)
        plt.hist(test_rmag_obs, density=True, bins=500, color='k', histtype='step')
        plt.axvline(17.8, ls=':', marker='', lw=1.5, label='$r-mag=17.8$')
        plt.xlabel("rmag"); plt.ylabel("PDF")
        plt.xlim(12,25)
        plt.legend()
        plt.show()

    if z_rmag :
        plt.figure()
        plt.scatter(test_redshift, test_rmag_obs, marker='.', s=1, alpha=0.02)
        plt.ylabel("r-mag"); plt.xlabel("redshift")
        plt.xlim(-0.1,1.); plt.ylim(12,25)

    return



def load_data(fits_file=None, sfr_filter=True, lgsfr_cut=-2, lum_units='L_sun'):
    """
    Load SDSS+WISE SED fitting results from Chang et al. catalog

    Input:
        - fits_file : path to datafile
        - sfr_filter : whether to drop all spurious SFR values (i.e., logsfr<-2)
        - lgsfr_cut : minimum allowed log(sfr) value
        - lum_units : ['L_sun', 'Lg_sun']

    Returns:
        - lgsfr_50 : median sfr values
        - lgmstar_50 : median stellar mass values
        - g_mag : rest-frame photometry (logL) in g-band [Lsun]
        - r_mag : rest-frame photometry (logL) in r-band [Lsun]
        - flags : an array with a flag (indicating quality of data)
    """

    if (fits_file==None) : fits_file = rltv_pth + "/data/sdss_wise_magphys_catalog.fits"
    # Open the fits file using a context manager
    with fits.open(fits_file) as hdul:

        #hdul.info()
        data = hdul[1].data
        header = hdul[1].header
        
        # Optionally display the data and header
        #print(data)
        #print(header)

        # load data of interest
        lgsfr_50 = data['lsfr50_all']; lgmstar_50 = data['lmass50_all']
        g_mag = data['lrest_g']; r_mag = data['lrest_r']
        flags = data['flag']

    if sfr_filter : 
        sfr__clean_ind = (lgsfr_50 > lgsfr_cut)
        return lgsfr_50[sfr__clean_ind], lgmstar_50[sfr__clean_ind], g_mag[sfr__clean_ind], r_mag[sfr__clean_ind], flags[sfr__clean_ind]
    
    if lum_units=='Lg_sun' : g_mag = np.log10(Lg_L_sun_to_Lg_Lg_sun(Lg_in_L_sun=10**g_mag)) 
    
    return lgsfr_50, lgmstar_50, g_mag, r_mag, flags


def sample_color(sfr, 
                 Nsample=1, 
                 density=None, 
                 color_arr=None, 
                 sfr_arr=None, 
                 plot=False):
    
    """
    Input:
        - sfr: logSFR value
        - Nsample : # of color samples
        - density : pdf in sfr-color space
        - color_arr : color grid
        - sfr-arr : sfr grid in log form
        - plot : whether to plot the pdf

    Returns: 
        - g_r_sample : A sample of colors drawn from the pdf
    """

    f_color = density[abs(sfr - sfr_arr).argmin()]
    cdf = np.nancumsum(f_color)
    cdf /= cdf[-1]
    if cdf[-1] < 0.99 : print("something is wrong here.") 

    if plot :
        plt.figure()
        plt.plot(color_arr, f_color, marker='', ls='-', color='k', label='$\mathrm{\log (SFR / M_\odot ~ yr^{-1})}=$'+f'${sfr:.2f}$')

        plt.xlabel("rest frame g-r"); plt.ylabel("$P(g-r | SFR)$")
        plt.legend()
        plt.tight_layout()
        plt.show()

    r_arr = np.random.rand(Nsample)
    g_r_sample = np.zeros(Nsample) 

    minnonzero_idx = np.nonzero(cdf)[0][0]
    clr_clean = color_arr[(minnonzero_idx-1):]
    cdf_clean = cdf[(minnonzero_idx-1):]

    epsilon = 0.05 #artificial unertainty in sampling to create smooth samples
    for i in range(Nsample) :
        g_r_sample[i] = clr_clean[abs(r_arr[i] - cdf_clean).argmin()] + epsilon * (2*np.random.rand() - 1)
    
    if plot :
        plt.figure()
        plt.hist(g_r_sample, bins=100, histtype='step')
        plt.xlabel("rest frame g-r"); plt.ylabel("$P(g-r | SFR)$")
        plt.show()
    return g_r_sample 

#  how to use
# sample_color(sfr=-1.5, Nsample=1000, density=test_dens, color_arr=test_col, sfr_arr=test_sfr, plot=False)



def generate_density_sfr_color(color_bins=500, 
                               color_range = (-1,1.5), 
                               sfr_bins=1000, 
                               sfr_range=(-5,4.), 
                               sfr_filter=True,
                               lgsfr_cut=-2,
                               plot=False):
    """
    It generates a pdf in sfr-color space
    Based on SDSS + WISE SED fit data
    Adopted from : 
    https://irfu.cea.fr/Pisp/yu-yen.chang/sw.html
    https://ui.adsabs.harvard.edu/abs/2015ApJS..219....8C/abstract

    Input :
        - color_bins : resolution in color
        - color_range : range of possible color values
        - sfr_bins : resolution in log(sfr)
        - sfr_filter : If True then it applies a cut at low sfr values
        - lgsfr_cut : If sfr_filter is activated then this is the minimum allowed log(sfr) value
        - plot : if True then the density is plotted

    Returns:
        - f_i : density in sfr-color space
        - color_array : an array of color grid values  
        - sfr_array : an array of logsfr grid values  

    """

    # load the data of interest
    log_sfr, log_mstar, Lg_band, Lr_band, flags = load_data(sfr_filter=sfr_filter, lgsfr_cut=lgsfr_cut)
    # compute the color g-r
    color_gr = -2.5 * (Lg_band - Lr_band)
    # keep only the good quality data
    ind = (flags == 1) 

    # compute the density by binning the data
    f_i, xiedges, yiedges = np.histogram2d(color_gr[ind], log_sfr[ind], \
                                        bins=[color_bins, sfr_bins], \
                                        range=[color_range, sfr_range])
    f_i = np.flipud(np.rot90(f_i))
    f_i[f_i == 0] = None  # mask zero values

    # Density plot (2D histogram)
    if plot :
        plt.figure()
        extent_i = [xiedges[0], xiedges[-1], yiedges[0], yiedges[-1]]
        # Plot the density as a heatmap
        im = plt.imshow(np.log10(f_i), extent=extent_i, aspect='auto', origin='lower', cmap='jet')

        # Add colorbar for density plot
        cbar = plt.colorbar(im, orientation='vertical', pad=0.01)
        cbar.set_label(r'$\mathrm{\log ~PDF(g-r, SFR)}$') 
        plt.xlabel('rest-frame g-r'); plt.ylabel('$\mathrm{\log(SFR/M_\odot ~ yr^{-1})}$')
        plt.tight_layout()
        plt.xlim(-0.2,0.8)
        plt.ylim(-2.5,2)
        plt.legend()
        plt.show()

    # return density, color_array, sfr_array 
    # f(logsfr, color)
    return f_i, (0.5*(xiedges[1:] + xiedges[:-1])), (0.5*(yiedges[1:] + yiedges[:-1]))



def plot_various_quantities(color_sfr_plot=True, 
                   mass_sfr_plot=True, 
                   Lg_Lr_plot=True, 
                   color_mass_plot=True, 
                   color_mtolg_plot=True,
                   Li_Leja_data=False):
    
    """
    This is a function that generates various plots
    to test the data. E.g., plot color-M/L density space
    """

    log_sfr, log_mstar, Lg_band, Lr_band, flags = load_data()


    color_gr = -2.5 * (Lg_band - Lr_band) 

    mtoLg_rest = log_mstar - np.log10(Lg_L_sun_to_Lg_Lg_sun(Lg_in_L_sun=10**Lg_band))

    ind = (flags == 1)

    print(color_gr[ind].shape)

    if color_sfr_plot :
        plt.figure()
        #plt.scatter(color_gr[ind], log_sfr[ind], marker='.', color='k', s=1, alpha=0.1, label='SFSS+WISE data')

        # Density plot (2D histogram)
        f_i, xiedges, yiedges = np.histogram2d(color_gr[ind], log_sfr[ind], \
                                            bins=[1000, 1000], \
                                            range=[(-1,1.5), (-5,4)])
        
        f_i = np.flipud(np.rot90(f_i))
        f_i[f_i == 0] = None  # mask zero values
        extent_i = [xiedges[0], xiedges[-1], yiedges[0], yiedges[-1]]

        # Plot the density as a heatmap
        im = plt.imshow(np.log10(f_i), extent=extent_i, aspect='auto', origin='lower', cmap='jet')

        # Add colorbar for density plot
        cbar = plt.colorbar(im, orientation='vertical', pad=0.01)
        cbar.set_label(r'$\mathrm{\log ~PDF(g-r, SFR)}$') 


        plt.xlabel('rest-frame g-r'); plt.ylabel('$\mathrm{\log(SFR/M_\odot ~ yr^{-1})}$')
        plt.tight_layout()
        plt.xlim(-0.2,0.8)
        plt.ylim(-2.5,2)
        plt.legend()
        plt.show()

    if mass_sfr_plot :
        plt.figure()
        #plt.scatter(color_gr[ind], log_sfr[ind], marker='.', color='k', s=1, alpha=0.1, label='SFSS+WISE data')

        # Density plot (2D histogram)
        f_i, xiedges, yiedges = np.histogram2d(log_mstar[ind], log_sfr[ind], \
                                            bins=[500, 500], \
                                            range=[(6,12), (-5,4)])
        
        f_i = np.flipud(np.rot90(f_i))
        f_i[f_i == 0] = None  # mask zero values
        extent_i = [xiedges[0], xiedges[-1], yiedges[0], yiedges[-1]]

        # Plot the density as a heatmap
        im = plt.imshow(np.log10(f_i), extent=extent_i, aspect='auto', origin='lower', cmap='jet')

        # Add colorbar for density plot
        cbar = plt.colorbar(im, orientation='vertical', pad=0.01)
        cbar.set_label(r'$\mathrm{\log ~PDF(M_star, SFR)}$') 


        plt.xlabel('$\mathrm{\log(M_\star / M_\odot)}$'); plt.ylabel('$\mathrm{\log(SFR/M_\odot ~ yr^{-1})}$')
        plt.tight_layout()
        plt.xlim(6.5,12.5)
        plt.ylim(-2.5,2)
        plt.legend()
        plt.show()



    if color_mass_plot :
        plt.figure()
        #plt.scatter(log_mstar[ind],log_sfr[ind], marker='.', color='k', s=1,alpha=0.1)
        # Density plot (2D histogram)
        f_i, xiedges, yiedges = np.histogram2d(color_gr[ind], log_mstar[ind], \
                                            bins=[300, 300], \
                                            range=[(-1,1.5), (7,12)])
                                            #range=[(6,12.5), (-3.5,3)])
        
        f_i = np.flipud(np.rot90(f_i))
        f_i[f_i == 0] = None  # mask zero values
        extent_i = [xiedges[0], xiedges[-1], yiedges[0], yiedges[-1]]

        # Plot the density as a heatmap
        im = plt.imshow(np.log10(f_i), extent=extent_i, aspect='auto', origin='lower', cmap='jet')

        # Add colorbar for density plot
        cbar = plt.colorbar(im, orientation='vertical', pad=0.01)
        cbar.set_label(r'$\mathrm{\log ~PDF(M_\star, g-r)}$') 


        plt.ylabel('$\mathrm{\log(M_\star / M_\odot)}$'); plt.xlabel('rest frame g-r')#plt.ylabel('$\mathrm{\log(SFR/M_\odot ~ yr^{-1})}$')
        plt.tight_layout()
        plt.xlim(-0.2,0.8)
        #plt.ylim(-5,2)
        plt.legend()
        plt.show()


    if Lg_Lr_plot :
        plt.figure()
        plt.scatter(Lg_band[ind],Lr_band[ind], s=1, color='k', marker='.', alpha=0.02)
        plt.plot([7,12], [7,12], marker='', ls='--', lw=2, color='red')
        plt.xlabel('rest frame $\mathrm{\log (L_g / L_\odot )}$')
        plt.ylabel('rest frame $\mathrm{\log (L_r / L_\odot )}$')
        plt.xlim(8,11.5); plt.ylim(8,11.5)
        plt.tight_layout()
        plt.show()


    if color_mtolg_plot: 

        plt.figure()
        #plt.scatter(log_mstar[ind],log_sfr[ind], marker='.', color='k', s=1,alpha=0.1)
        # Density plot (2D histogram)
        f_i, xiedges, yiedges = np.histogram2d(color_gr[ind], mtoLg_rest[ind], \
                                            bins=[500, 500], \
                                            range=[(-0.5,1.2), (-2,2)])
        
        f_i = np.flipud(np.rot90(f_i))
        f_i[f_i == 0] = None  # mask zero values
        extent_i = [xiedges[0], xiedges[-1], yiedges[0], yiedges[-1]]

        # Plot the density as a heatmap
        im = plt.imshow(np.log10(f_i), extent=extent_i, aspect='auto', origin='lower', cmap='jet', zorder=1)

        # Add colorbar for density plot
        cbar = plt.colorbar(im, orientation='vertical', pad=0.01)
        cbar.set_label(r'$\mathrm{\log ~PDF(M_\star/L_g, g-r)}$') 



        if Li_Leja_data :


            m_l, g_r, density  = compute_color_ml_density(plot=False)
            for j in range(len(g_r)) :
                
                cdf_tmp = np.cumsum(density[j,:])
                cdf_tmp /= cdf_tmp[-1]
                p_16, p_84 = abs(0.02 - cdf_tmp).argmin(), abs(0.98 - cdf_tmp).argmin() # 2sigma mask

                density[j,:p_16] = -99999; density[j,p_84:] = -99999
            
            density = np.ma.masked_where(density < 0, density)
                
            # -- plot results -- #
            plt.imshow(density.T ,origin='lower', aspect='auto', extent=[g_r[0], g_r[-1], np.log10(m_l[0]), np.log10(m_l[-1])], alpha=0.8, zorder=0)



        plt.ylabel('$\mathrm{\log(M_\star / L_g) ~~ [M_\odot/L_{g,\odot}]}$'); plt.xlabel('rest frame g-r')#plt.ylabel('$\mathrm{\log(SFR/M_\odot ~ yr^{-1})}$')
        plt.tight_layout()
        plt.xlim(-0.2,0.8) ; plt.ylim(-2,1)
        #plt.ylim(-5,2)
        plt.legend()
        plt.show()


    return



## Li & Leja color-MtoL distribution 
 

def load_color_data(fname='./color_g_r_dist_Li_Leja20.csv', plot=True) :
    """"
    (Old function...will not be used)
    Extract the pdf in color g-r from Li & Leja Fig. 2
    Next, use an interpolation scheme to obtain
    a continuous PDF for later usage
    """

    # Load the CSV file into a NumPy array
    _data = np.genfromtxt(fname, delimiter=',', skip_header=1) 
    # interpolate the distribution
    color_dist = interp1d(x=_data[:,0], y=_data[:,1], bounds_error=False, fill_value=(0.,0.))

    if plot : 
        # plot the data and the interpolated predictions
        plt.figure()
        plt.plot(_data[:,0], _data[:,1], color='k', marker='.', label="Li and Leja data from fig 2")
        x = 2*np.copy(_data[:,0])
        x.sort()
        plt.plot(x, color_dist(x), ls='-', marker='', label='interpolation')
        plt.title("Adopted from Li \& Leja (2022)")
        plt.xlabel("rest-frame g-r")
        plt.ylabel("PDF")
        plt.legend()
        plt.show()

    #
    return _data[:,0], _data[:,1], color_dist


def compute_color_coefs():
    """
    A function used to calculate the
    pdf in mass-to-light (from Li & Leja 2022)
    """

    gr1, gr2, gr3 = - 0.3, 0.5, 1.3
    A_matrix = np.array([[1, gr1, gr1**2], [1, gr2, gr2**2], [1, gr3, gr3**2]], dtype=float)
    invA = np.linalg.inv(A_matrix)
    try:
        np.allclose(A_matrix @ invA, np.eye(3))
    except: 
        raise Exception("THe inversion of the matrix did not carried out properly.")
    
    # values of params at g-r = {-0.3, 0.5, 1.3}
    params_color = {
        'lambda_arr' : [-0.989, -0.451, -0.33],
        'p_arr' : [1.259, 0.826, 0.512],
        'q_arr' : [0.36, 19.653, 77.119]
    }

    return invA @ params_color['lambda_arr'], invA @ params_color['p_arr'], invA @ params_color['q_arr']

# initiate a dictionary that stores all parameters
pars_ml_color = {
                'a_arr' : [-0.659, 1.541, 0.149, -0.121],
                "p" : None, #p coefs
                "sigma" : 0.062,
                "lambda" : None, # lambda coefs
                "q" : None, # q coefs
                'a_arr_err' : [0.002, 0.007, 0.001, 0.006],
                "p_err" : None,
                "sigma_err" : 0.0035,
                "lambda_err" : None,
                "q_err" : None
            }

pars_ml_color['lambda'], pars_ml_color['p'], pars_ml_color['q'] = compute_color_coefs()

def load_pars_ml_color() :
    """
    A function that can be called to
    load the parameters of the pdf in mass-to-light
    """

    return pars_ml_color


def pdf_mass_to_light(m_l=None, g_r=None, redshift=None, pars=None):
    """"
    Represents the probability density function in M/L_g at given g-r, z
    Adopted from Li & Leja (2022)

    Input:
        - m_l : mass-to-light array
        - g_r : color value
        - redshift : refshift value
        - pars : various useful parameters involved in the calculation

    Returns:
        - probability distribution function
    """

    pc1, pc2, pc3          = pars['p'][0], pars['p'][1], pars['p'][2]
    qc1, qc2, qc3          = pars['q'][0], pars['q'][1], pars['q'][2]
    lc1, lc2, lc3          = pars['lambda'][0], pars['lambda'][1], pars['lambda'][2]
    muc1, muc2, muc3, muc4 = pars['a_arr'][0], pars['a_arr'][1], pars['a_arr'][2], pars['a_arr'][3]


    # compute skewed generalized Student's (sgt) distribution coefficients
    mu    = muc1 + muc2 * g_r + muc3 * g_r * g_r + muc4 * redshift
    l     = lc1 + lc2 * g_r + lc3 * g_r * g_r
    p     = pc1 + pc2 * g_r + pc3 * g_r * g_r
    q     = qc1 + qc2 * g_r + qc3 * g_r * g_r
    sigma = pars['sigma']

    return ( p )/( 2 * sigma * pow(q,1/p) * sp_beta(1/p, q) ) \
        * ( 1 + (abs(np.log10(m_l) - mu)**p) / (q * (sigma**p) * (( l * sign(np.log10(m_l) - mu) + 1 )**p) ) )**(-(q + 1/p))



def compute_color_ml_density(m_l = np.logspace(-2.2,1.2,2000), 
                          g_r = np.linspace(-0.25,1.2,500),
                          z_av=True, 
                          zo=None, 
                          contour_2sigma=True, 
                          plot=False):

    """ 
    Adopted from Li & Leja (2022)

    Input: 
        - m_l : mass-to-light grid points
        - g_r : color grid points
        - z_av : if you want to marginalize over redshift
        - zo : If z_av=False then you have to specify a redshift value
        - contour_2sigma : flag to mask out points out 2 sigma contours
        - plot : flag for plotting the probabaility density

    Returns:
        - m_l : mass-to-light grid points
        - g_r : color grid points
        - full_density : 2-dimensional pdf
    """

    #grid resolution
    #m_l = np.logspace(-2.2,1.2,2000) 
    #g_r = np.linspace(-0.25,1.2,500)

    _, _, interp_color_dist = load_color_data(plot=False)

    if (zo==None) and (z_av) : z_arr = np.linspace(0.0,3,100)
    else : z_arr = np.array([zo])

    # compute probabaility density 
    # average over redshift (see Fig. 3 in li & Leja 2022)
    density = np.zeros(shape=(len(g_r), len(m_l))) # initiate array
    for redshift in z_arr :
        for i in range(len(g_r))  :
            density[i,:] += (1/len(z_arr)) \
                * interp_color_dist(g_r[i]) \
                    * pdf_mass_to_light(m_l=m_l, g_r=g_r[i], redshift=redshift, pars=pars_ml_color)
    
    full_density = np.copy(density) # obtain a copy of the initial density

    if plot :
        if contour_2sigma :

            for j in range(len(g_r)) :
                
                cdf_tmp = np.cumsum(density[j,:])
                cdf_tmp /= cdf_tmp[-1]
                p_16, p_84 = abs(0.02 - cdf_tmp).argmin(), abs(0.98 - cdf_tmp).argmin() # 2sigma mask

                density[j,:p_16] = -99999; density[j,p_84:] = -99999
            
            density = np.ma.masked_where(density < 0, density)

        else : # just mask low values of probability         
            density = np.ma.masked_where(density < 1e-3*density.max(), density)
        
        
        # -- plot results -- #
        plt.imshow(density.T ,origin='lower', aspect='auto', extent=[g_r[0], g_r[-1], np.log10(m_l[0]), np.log10(m_l[-1])], alpha=0.8)
        plt.plot(g_r, -0.659 + 1.541*g_r + 0.149 * g_r * g_r - 0.121*(z_arr[0]+z_arr[-1])/2, \
                marker='', ls='--', color='red', label='Eq. (12) in Li \& Leja (2022)')
        
        plt.colorbar(label='Probability')
        plt.ylim(-2.1,1.1); plt.xlim(-0.2,1.1)
        plt.xlabel('rest-frame g-r')
        plt.ylabel(r"rest-frame $\log(M/L_g)$")
        plt.title(r"Probability density in $M/L_g ~-~ g-r$ space")
        #plt.text(y=0.5, x=-0.15, s='Adopted from Li \& Leja (2022)\n'\
        #        +r'Averaged over redshift $z\in[0.5,3]$'\
        #            +'\n2 sigma mask applied to the data',color='grey', fontsize=12)
        #plt.savefig('./figs/color_ml_density.png', format='png')

    return m_l, g_r, full_density # return the unmasked density










### new functions







def sample_mass_to_light(sfr, 
                         z, 
                         m_l_arr=np.logspace(-2.2,1.2,2000), 
                         Nsample=1, 
                         density_sfr_color=None, 
                         color_arr=None, 
                         sfr_arr=None) :

    """
    Mass-to-light sampler

    Input: 
        - sfr : log of galaxy's sfr
        - m_l_array : an array of m_l values (including the resolution)
        - Nsample : How many colors to sample (set to 1). It doesn't work for more
        - density_sfr_color : give a 2-d array of pdf in sfr-color in order to sample a color
        - color_arr : an array with color grid values (including resolution)
        - sfr_arr : an array with log(sfr) grid values (including resolution)

    Returns:
        - m_l : sample mass-to-light ratio (rest frame M/Lg) -- not in logarithmic form
        - sample_color : the sampled color g-r value
        - sfr : input log sfr
        - z : input redshift  
    """

    # sample a color for given sfr and redshift
    sampled_color = sample_color(sfr=max(sfr + 0.1, sfr_arr[0]), Nsample=Nsample, density=density_sfr_color, color_arr=color_arr,sfr_arr=sfr_arr, plot=False)

    #m_l_arr = np.logspace(-2.2,1.2,2000)
    # load the pdf describing the distribution in the mass-to-light ratio <-- offered by Li & Leja 2022
    parameters_ml_color = load_pars_ml_color() # load the parameters that are being used in the pdf(ml)
    pdf_mtol = pdf_mass_to_light(m_l=m_l_arr, g_r=sampled_color[0], redshift=z, pars=parameters_ml_color)
    # compute the cumulative to sample from it
    cdf_mtol = np.cumsum(pdf_mtol)
    cdf_mtol /= cdf_mtol[-1]

    return m_l_arr[abs(np.random.rand() - cdf_mtol).argmin()], sampled_color[0], sfr, z 


def run_kstest(data_sample1=None, data_sample2=None, printout=False):
        # run ks-test
        """
        Notes related to hypothesis testing can be found here: 
        https://github.com/astrostatistics-in-crete/2024_summer_school/blob/main/02_Hypothesis/Hypothesis.ipynb
        """

        pvalue = ks_2samp(data_sample1, data_sample2, alternative='two-sided', method='auto').pvalue
        if printout : print(f"The ks-test has returned a p-value equal to {pvalue:.2e}.")
        #plt.text(x=np.max(massbins)-2.1, y=0.05, s=f"p-value={pvalue:.1e}", fontsize=10, color='blue')
        return pvalue



def from_MtoLg_to_MtoLr(g_r=None, MtoLg=None) :
    """
    Conver M/Lg to M/Lr

    Input:
        - g_r : rest frame color g-r
        - MtoLg : rest frame mass-to-light in g-band

    Returns :
        - MtoLr : mass to light ratio in r-band  
    """

    return  MtoLg * 10**(- g_r / 2.5)

def Kr_correction(g_r=None, z=None) :
    """
    K-correction in r-band adopted from
    https://arxiv.org/abs/2212.14539
    valid in redshift range ~(0.6, 2)

    It is defined as 
        m_rest - m_obs = K_r

    Input :
        - g_r : rest frame color g-r
        - z : redshift

    Returns:
        - Kr-correction
    """

    return (13.3 * g_r - 0.5 - 3.5 * (z - 1.47)**2) * np.log10(1 + z) 









###################################
# Functions related to generating #
#    a mock sample of galaxies    #
##################################








def Phi_Leja(logm=np.linspace(8, 12, 100)[:, None], 
             z=0, 
             errors=False
             ): 
     '''
     Stellar mass function adopted from Leja et al. (2020)
     
     Input:
          - logm: Grid of logarithm of stellar mass
          - z: redshift
          - errors: if True, then the algorithm provides 1-sigma uncertainties in the stellar mass function
     
     Returns:
        - phi : The median of the stellar mass function (dN/dlogM)
     '''

     def schechter(logm, logphi, logmstar, alpha, m_lower=None): 
          """ 
          Generate a Schechter function (in dlogm).
          """

          phi = ((10**logphi) * np.log(10) * \
          10**((logm - logmstar) * (alpha + 1)) * \
               np.exp(-10**(logm - logmstar)))
          return phi


     def schechter_median(logm, z):

          yphi1 = [-2.44,-3.08,-4.14]
          yphi2 = [-2.89,-3.29,-3.51]
          ymstar = [10.79,10.88,10.84]
          logphi1 = parameter_at_z0(yphi1,z)
          logphi2 = parameter_at_z0(yphi2, z)
          logmstar = parameter_at_z0(ymstar, z)
          alpha1, alpha2 = - 0.28, - 1.48
          
          phi1 = ((10**logphi1) * np.log(10) * \
               10**((logm - logmstar) * (alpha1 + 1)) * \
               np.exp(-10**(logm - logmstar)))
          
          phi2 = ((10**logphi2) * np.log(10) * \
               10**((logm - logmstar) * (alpha2 + 1)) * \
               np.exp(-10**(logm - logmstar)))

          return phi1 + phi2



     def parameter_at_z0(y,z0,z1=0.2,z2=1.6,z3=3.0):
          """
          Compute parameter at redshift ‘z0‘ as a function 
          of the polynomial parameters ‘y‘ and the 
          redshift anchor points ‘z1‘, ‘z2‘, and ‘z3‘.
          """
          y1, y2, y3 = y
          a = (((y3-y1) + (y2 - y1) / (z2 - z1) * (z1 - z3)) / \
               (z3**2 - z1**2 + (z2**2 - z1**2) / (z2 - z1) * (z1 - z3))) 
          b = ((y2 - y1) - a * (z2**2 - z1**2)) / (z2 - z1)
          c = y1 - a * z1**2 - b * z1
          return a * z0**2 + b * z0 + c

     if (errors):
          # Continuity model median parameters + 1-sigma uncertainties. 
          pars = {'logphi1': [-2.44,-3.08,-4.14],
                    'logphi1_err': [0.02, 0.03, 0.1],
                    'logphi2': [-2.89,-3.29,-3.51],
                    'logphi2_err': [0.04, 0.03, 0.03], 
                    'logmstar': [10.79,10.88,10.84], 
                    'logmstar_err': [0.02, 0.02, 0.04], 
                    'alpha1': [-0.28], 
                    'alpha1_err': [0.07], 
                    'alpha2': [-1.48], 
                    'alpha2_err': [0.1]
               }

          # Draw samples from posterior assuming independent Gaussian uncertainties. 
          # # Then convert to mass function at ‘z = z0‘. 
          draws =  {}
          ndraw = 1000 
          z0 = z
          for par in ['logphi1', 'logphi2', 'logmstar', 'alpha1', 'alpha2']: 
               samp = np.array([np.random.normal(median,scale = err,size = ndraw) \
                              for median, err in zip(pars[par], pars[par+'_err'])])
               if par in ['logphi1', 'logphi2', 'logmstar']: 
                    draws[par] = parameter_at_z0(samp,z0) 
               else: draws[par] = samp.squeeze()

          # Generate Schechter functions. 
          #logm = np.linspace(8, 12, 100)[:, None] #log(M) grid
          phi1 = schechter(logm, draws['logphi1'],  # primary component
                    draws['logmstar'], draws['alpha1']) 
          phi2 = schechter(logm, draws['logphi2'], # secondary component 
                    draws['logmstar'], draws['alpha2']) 
          phi = phi1 + phi2 # combined mass function 

          # Compute median and 1-sigma uncertainties as a function of mass. 
          phi_50, phi_84, phi_16 = np.percentile(phi, [50, 84, 16], axis =  1)

          phi = phi_50

     else :
          phi = schechter_median(logm, z)  
     
     return phi 


def plot_Leja_mfunc() :

    Nz = 100 #redshift bins
    Mstar = np.logspace(7,12, 1500)
    z = np.linspace(0.2, 3, Nz)
    f, ax = plt.subplots(1,1, figsize=(10,5))
    cmap = plt.cm.plasma #set colormap
    colors = cmap(np.linspace(0,1,Nz)) #sample colormap with as many lines as you want to draw
    for i in range(Nz):
        y = Phi_Leja(logm = np.log10(Mstar), z=z[i]) #generate curve given choice of z
        plt.semilogy(np.log10(Mstar), y, color=colors[i], lw=0.5, marker='', linestyle='-') #draw curve

    norm = plt.Normalize(z.min(), z.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    f.colorbar(sm, ax=ax, ticks=np.linspace(z.min(), z.max(), 8), label=r"$z$")

    plt.xlabel("$\mathrm{\log M_{\star} ~ [M_\odot]}$"); plt.ylabel("$\mathrm{\Phi(M_{\star}, z) ~ [arb.]}$")
    plt.title("Stellar mass function adopted from Leja et al. (2020)")
    plt.ylim(1e-5,1e-1)
    plt.xlim(7,12)
    plt.tight_layout()
    plt.show()
    return


def log_mass_function(Mstar=1e8, 
                      z=None, 
                      mfunc_ref='Schechter',
                      mfunc_slope=0,
                      mfunc_mstar0=1e6
                      ):
    '''
    Stellar mass function

            Phi dlogm = dn/dlogM  dlogM

    Input:
        - Mstar: Stellar mass at z (not in logarithmic scale)
        - z: redshift
        - mfunc_ref: which parameteric formula to be used as a mass function
        - mfunc_slope: slope coefficient for artificially changing the stellar mass-function slope
        - mfunc_Mstar0: stellar-mass coefficient below which the stellar mass slope changes artifically

    Returns:
        - log_phi : the logarithm of the stellar mass function (log (dn/dlogm))
    '''
    log_phi = []
    if mfunc_ref=='Schechter' :
    
        '''
        Adopted from Bochenek et al. (2021)
        Constants:
            - phi_0: normalization
            - Gamma: power-law index
            - lgM_c: logarithm of cutoff stellar mass
        '''
        phi_0 = 8.34
        lgM_c = 10.6
        Gamma = -0.1

        log_phi = phi_0 - lgM_c + Gamma*(np.log10(Mstar) - lgM_c) - np.log10(np.e) * 10**(np.log10(Mstar) - lgM_c)

    elif mfunc_ref=='Leja' :

        log_phi = np.log10(Phi_Leja(logm=np.log10(Mstar), z=z, errors=False))

    else : 
        raise(Exception("The requested mass-function is not defined"))
  
    return log_phi + mfunc_slope * np.log10(1 + (mfunc_mstar0/Mstar))



def SFMS(Mstar, 
         z, 
         sfr_ref='Speagle', 
         mode='ridge', 
         sample_size=1, 
         posterior=False
         ):
    
    '''
    Star-forming main sequence 

    Input:
        - Mstar: Stellar mass at z
        - z: redshift
        - sfr_ref: which parameteric formula to be used
        - mode: which fit is used from Leja+2022 (options: 'ridge', 'mean')
        - sample_size: how many times to sample the posterior
        - posterior: enables sampling of the posterior
    '''

    if (sfr_ref=='Speagle') :

        '''
        Adopted from Speagle et al. (2014)
        https://ui.adsabs.harvard.edu/abs/2014ApJS..214...15S/abstract
        
        Parameters:
            - t: Age of the Universe at z [in Gyr]
        Constants:
            - k1: 0.84+-0.02
            - k2: 0.026+-0.003
            - k3: 5.51+-0.24
            - k4: 0.11+-0.03
        '''
        t = cosmo.age(z).value

        n_coeff = 4
        sfms_pars = {
                'k1' : 0.84, 
                'k2' : 0.026,
                'k3' : 6.51,
                'k4' : 0.11,
                'e_k1' : 0.02,
                'e_k2' : 0.003,
                'e_k3' : 0.24,
                'e_k4' : 0.03
                }
        
        def sample_coeffs():

            mean = [sfms_pars['k'+f'{(i+1):.0f}'] for i in range(n_coeff)]
            cov = np.diag([sfms_pars['e_k'+f'{(i+1):.0f}'] for i in range(n_coeff)])**2
            draw = np.random.multivariate_normal(mean, cov, size=sample_size)[0]
            return draw

        
        if posterior :
            coeffs = sample_coeffs()
            lg_SFR = (coeffs[0] - coeffs[1] * t) * np.log10(Mstar) - (coeffs[2] - coeffs[3] * t)

        else :
            coeffs = {}
            for par in ['k1', 'k2', 'k3', 'k4'] :
                coeffs[par] = sfms_pars[par] 
        
            lg_SFR = (coeffs['k1'] - coeffs['k2'] * t) * np.log10(Mstar) - (coeffs['k3'] - coeffs['k4'] * t)




    elif (sfr_ref=='Leja'):
        
        # parameters of the continuity model for SFMS
        # mean
        if (mode=='mean') : 
            sfms_pars = {
                    'alpha' : [-0.06707, 0.3684, -0.1047],
                    'beta'  : [0.8552, -0.1010, -0.001816],
                    'c'     : [0.2148, 0.8137, -0.08052],
                    'lgMc'  : [10.29, -0.1284, 0.1203],
                    'e_alpha' : [0.00821, 0.0128, 0.0044],
                    'e_beta'  : [0.0068, 0.0107, 0.00329],
                    'e_c'     : [0.0045, 0.0069, 0.00219],
                    'e_lgMc'  : [0.01, 0.0135, 0.0044]
                    }
        
        #ridge
        elif (mode=="ridge") : 
            sfms_pars = {
                    'alpha'   : [0.03746, 0.3448, -0.1156],
                    'beta'    : [0.9605, 0.0499, -0.05984],
                    'c'       : [0.2516, 1.118, -0.2006],
                    'lgMc'    : [10.22, 0.3826, -0.04491],
                    'e_alpha' : [0.01739, 0.0297, 0.0107],
                    'e_beta'  : [0.01, 0.01518, 0.00482],
                    'e_c'     : [0.0082, 0.013, 0.0039],
                    'e_lgMc'  : [0.01, 0.0188, 0.00613]
        }
            
        else : raise Exception(f"The combination of sfr_ref={sfr_ref} and mode={mode} is not available.")

        def sample_coeff(par=None):

            mean = sfms_pars[par]
            cov = np.diag(sfms_pars['e_'+par])**2.
            draw = np.random.multivariate_normal(mean, cov, size=sample_size)
            return parameter_at_z(y=draw[0], z0=z)

        def parameter_at_z(y, z0):
            
            xio, xi1, xi2 = y
            return xio + xi1*z0 + xi2*z0*z0  #Eq. 10 in Leja et al. (2021)
    
        coeffs = {}
        for par in ['alpha', 'beta', 'c', 'lgMc'] :
            if posterior==True : coeffs[par] = sample_coeff(par=par)
            else : coeffs[par] = parameter_at_z(y=sfms_pars[par], z0=z)
        
        lg_mstar = np.log10(Mstar) 
        lg_SFR = coeffs['c'] + (lg_mstar - coeffs['lgMc']) * (coeffs['beta'] + np.heaviside(lg_mstar - coeffs['lgMc'], 1) * (coeffs['alpha'] - coeffs['beta']) )
        # remember to change the heaviside function for better performance (see the Documentation)

    return lg_SFR # in Msun/yr






#####################################
#     Adopted from Leja's paper     #
#     trained normalizing flow      #
#     NN used to model the          #
#     posterior distribution in     #
#     logmstar-logsfr-z space       #
#####################################





def load_leja_posterior(dens_params={}):

    """
    This function makes use of the trained
    neural network offered 
    """
    
    flow = load_nf()
    prob_dens = sample_density(flow, dens_params=dens_params, redshift_smoothing=True)
    zgrid = np.linspace(dens_params['zmin'], dens_params['zmax'], len(prob_dens[0,0,:]))
    mgrid = np.linspace(dens_params['mmin'], dens_params['mmax'], dens_params['nlogm'])
    sfrgrid = np.linspace(dens_params['sfrmin'], dens_params['sfrmax'], dens_params["nsfr"])
    return prob_dens, zgrid, mgrid, sfrgrid


def load_stored_leja_nn(datafile=None) :
    """
    Load stored NN probability density

    Input:
        - datafile: name of datafile where the data is stored

    Returns:
        - data_prop_arr: probability density matrix
        - data_z_arr: redshift values array
        - data_logm_arr: grid point values of logmstar
        - datalogsfr_arr: grid point values of logsfr
    """
    #import h5py
    # Open the HDF5 file and read the data
    with h5py.File(datafile, 'r') as hf:

        # Access arrays
        data_prob_arr = hf['p_prob_arr'][:]
        data_z_arr = hf['p_z_arr'][:]
        data_logm_arr = hf['p_logm_arr'][:]
        data_logsfr_arr = hf['p_logsfr_arr'][:]

        # Access metadata
        nlogm = hf.attrs['nlogm']
        nsfr = hf.attrs['nsfr']
        dz = hf.attrs['dz']
        ndummy = hf.attrs['ndummy']
        mmin = hf.attrs['mmin']
        mmax = hf.attrs['mmax']
        sfrmin = hf.attrs['sfrmin']
        sfrmax = hf.attrs['sfrmax']
        zmin = hf.attrs['zmin']
        zmax = hf.attrs['zmax']
        description = hf.attrs['description']

    # Print the information in the required format
    print(f"""
    The stored star formation density was sampled using the following parameters:
    nlogm={nlogm}, nsfr={nsfr}, dz={dz}
    ndummy={ndummy}
    mmin={mmin}, mmax={mmax}
    sfrmin={sfrmin}, sfrmax={sfrmax}
    zmin={zmin}, zmax={zmax}
    {description}
    If another resolution is desired, please use utls.load_leja_posterior()"
    """)

    data_dens_params = {
            "nlogm"  : nlogm, #120,
            "nsfr"   : nsfr, #120,
            "dz"     : dz, #0.1,
            "ndummy" : ndummy, #31,
            "mmin"   : mmin, #6.5, 
            "mmax"   : mmax, #12.5,
            "sfrmin" : sfrmin, #-3.,
            "sfrmax" : sfrmax, #3.,
            "zmin"   : zmin, #0.2, 
            "zmax"   : zmax, #1.2
        }

    return data_dens_params, data_prob_arr, data_z_arr, data_logm_arr, data_logsfr_arr 



def interpolate_prob(prob_arr=None, z_arr=None, logm_arr=None, logsfr_arr=None, ztarget=1., plot=False):

    zidx = np.abs(z_arr-ztarget).argmin()
    posterior = prob_arr[:,:,zidx]
    if plot : plot_posterior(posterior, z_arr, logm_arr, logsfr_arr, ztarget=ztarget, title='NN Posterior') # plot data
    
    # interpolate the probability density for given redshift
    #print(np.shape(logm_arr), np.shape(logsfr_arr), np.shape(posterior))
    interpolator = interp2d(logsfr_arr, logm_arr, posterior, kind='cubic')
    
    if plot : 
        plot_posterior(interpolator(logsfr_arr, logm_arr), z_arr, logm_arr, logsfr_arr, ztarget=ztarget, title='Interpolated Posterior') #plot interpolation prediction
        plot_posterior(np.log10(np.abs(interpolator(logsfr_arr, logm_arr) - posterior)), z_arr, logm_arr, logsfr_arr, ztarget=ztarget, title='Residuals') # plot residuals

    return interpolator


def plot_posterior(prob_arr=None, z_arr=None, logm_arr=None, logsfr_arr=None, ztarget=1., title=''):
    
    #prob_arr, z_arr, logm_arr, logsfr_arr = load_leja_posterior()
    
    plt.figure()
    
    zidx = np.abs(z_arr-ztarget).argmin()
    try :
        posterior = prob_arr[:,:,zidx]
    except :
        posterior = prob_arr
        
    im = plt.imshow(posterior.T, cmap='binary', norm='linear', origin='lower', aspect='auto', extent=[min(logm_arr), max(logm_arr), min(logsfr_arr), max(logsfr_arr)])
    plt.colorbar(im, label='Posterior')
    plt.xlabel("$\mathrm{M_{star} ~ [M_\odot]}$"); plt.ylabel("$\mathrm{\log{SFR_{SED}} ~ [M_\odot ~ yr^{-1}]}$")
    plt.title(title)
    plt.plot(logm_arr, SFMS(10**logm_arr, z=ztarget, sfr_ref='Leja', mode='ridge', posterior=False), marker='', ls='-', color='red', lw=2, label='Leja+22 fit')
    plt.axvline(x=threedhst_mass_completeness(zred=ztarget), ymin=0, ymax=1,ls=':', color='green', label='Mass completeness threshold')
    plt.text(7,0.9*max(logsfr_arr), s=f"z={ztarget:.1f}", color='gray', fontsize=16)
    
    if (title != 'Residuals'):
        sfr_ridge = np.ndarray(shape=(len(logm_arr),), dtype=float)
        sfr_p16 = np.copy(sfr_ridge); sfr_p84 = np.copy(sfr_ridge); sfr_p50 = np.copy(sfr_ridge); sfr_p2 = np.copy(sfr_ridge); sfr_p98 = np.copy(sfr_ridge)
        for i in range(len(logm_arr)):
            sfr_ridge[i] = logsfr_arr[np.argmax(posterior[i,:])]
            c_post = np.cumsum(posterior[i,:])
            c_post /= c_post[-1]
            sfr_p16[i] = logsfr_arr[np.abs(0.16 - c_post).argmin()]
            sfr_p84[i] = logsfr_arr[np.abs(0.84 - c_post).argmin()]
            sfr_p50[i] = logsfr_arr[np.abs(0.5 - c_post).argmin()]
            sfr_p2[i]  = logsfr_arr[np.abs(0.02 - c_post).argmin()]
            sfr_p98[i] = logsfr_arr[np.abs(0.98 - c_post).argmin()]
            #if np.argmax(posterior[i,:]) >= np.abs(0.5 - c_post).argmin() : raise Exception("something is wrong buddy")
        #plt.plot(logm_arr, sfr_ridge, marker='', ls=':', color='orange', label='mode of posterior')
        plt.fill_between(logm_arr, sfr_p16, sfr_p84, color='orange', alpha=.2, label=r'$1\sigma$')
        plt.fill_between(logm_arr, sfr_p2, sfr_p98, color='orange', alpha=.1, label=r'$2\sigma$')
        plt.plot(logm_arr, sfr_p50, marker='', ls='-', color='orange', label='median')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

    return


def load_Sharma(desc='Sharma_only'):
    """
    Load the data used in Sharma et al. (2024) paper

    Input:
    - desc: what dats should be loaded? ['Sharma_only', 'Sharma_full']

    Returns:
    - Stellar mass Mstar in Msun units
    - redshift
    """
    if desc=='Sharma_only' :
        data = np.loadtxt(rltv_pth + '/data/sharma_m_sfr_z_data.dat', unpack=True)
        print("Sharma data are loaded")
        return 10**(data[0]), data[2] # data[0]=logM_star , data[2] = redshift
    
    elif desc=='Sharma_full' :
        data = read_frbs_hosts_data()
        print("All Sharma data are loaded including Gordon + Bhardwaj data")
        return 10**(data['logM'].values), data['z']

    else: raise Exception(f'{desc} dataset is not available.') 


# Load ULXs (adopted from Sharma et al. 2024)
def read_ULXsources_hosts_data(Dcut=False):
    """
    Read ULX sources host data from Kovlakas+2020, filter based on criteria,
    calculate necessary values, and return a DataFrame with relevant columns.

    Returns:
    -------
    ULX_df : DataFrame
    DataFrame with columns 'logSFR', 'logSFR_errl', 'logSFR_erru', 'logM', 
    'logM_errl', 'logM_erru', 'z'.
    """
    ULX_data1 = Table.read('other_transients_data/kovlakas_et_al_hosts.fits')
    ULX_data1["pgc"] = ULX_data1["PGC"]
    ULX_data2 = Table.read('other_transients_data/kovlakas_et_al_sources.fits')
    ULX_data = astropy.table.join(ULX_data1, ULX_data2, keys="pgc", 
                                  join_type='inner')

    ULX_data = ULX_data[~ULX_data["unreliable"]]
    if Dcut:
        ULX_data = ULX_data[ULX_data["D"] < 40]
    ULX_data = ULX_data[ULX_data["LX"] > 1e39]
    ULX_data = ULX_data[~ULX_data["nuclear"]]
    ULX_data = ULX_data[ULX_data["logM"] > 1]  # to avoid nan entries
    ULX_data = ULX_data[ULX_data["logSFR"] > -5]  # to avoid nan entries
    ULX_data = unique(ULX_data, keys='PGC')

    ULX_logM = ULX_data["logM"]
    ULX_logSFR = ULX_data["logSFR"]
    ULX_logM_err = ULX_data["logM"] * 0.01
    ULX_logSFR_err = ULX_data["logSFR"] * 0.1
    ULX_z = z_at_value(Planck13.luminosity_distance, ULX_data["D"])

    ULX_df = pd.DataFrame({"logSFR": ULX_logSFR, 
                           "logSFR_errl": ULX_logSFR_err, 
                           "logSFR_erru": ULX_logSFR_err,
                           "logM": ULX_logM, 
                           "logM_errl": ULX_logM_err, 
                           "logM_erru": ULX_logM_err, 
                           "z": ULX_z})

    return ULX_df




