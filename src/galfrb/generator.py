# imports
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys
import h5py
from datetime import date
from scipy.optimize import newton, brentq # root finder
from scipy.stats import ks_2samp
from tabulate import tabulate
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from . import utils as utls # helper functions

import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#plt.style.use("nick_style")


from astropy import units
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM # https://docs.astropy.org/en/stable/cosmology/index.html
from astropy.cosmology import z_at_value 
cosmo = FlatLambdaCDM(H0=70, Om0=0.3) # define the cosmology to be used in this script

from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d, interp2d, RegularGridInterpolator

from scipy.special import beta as sp_beta
import math
sign = lambda x: 2*np.heaviside(x, 0.5) - 1 # define sign function

from tqdm import tqdm

# load the Neural Network of leja+22 which computes the proabability density function in the parameter space logmstar - logsfr - redshift
# Add the 'libs' directory to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "libs"))
from sfr_leja.code.sample_nf_probability_density import sample_density, load_nf, threedhst_mass_completeness, cosmos15_mass_completeness

def completeness_threshold(zred: float) -> float :
    '''
    Compute the mass completeness threshold value
    -----
    Input:
        zred     : float
        redshift
    -----
    Returns:
        logmstar : float
        mass completeness threshold 
    -----
    Notes:
        We artificially add an offest 0.5 to ensure good quality
        of the pdf in sfr.
    '''
    if zred <= 0.2 : return cosmos15_mass_completeness(zred=0.3) + 0.3
    else : return threedhst_mass_completeness(zred=zred) + 0.3

Lum_r_sun = 0.15 # times the solar luminosity. I computed using the r-band effective values: lamba = 6180 A and Delta lambda = 1200 A
    



def Gen_mock_gal(Nsample     = 1000, 
                z_arr       = [0.],  
                plot_cdf    = False,
                plot_pdf    = True,
                logm_min    = 6.5,
                logm_max    = 12.5, 
                nbins       = 5000,
                weight      = None,
                sfr_ref     = 'Speagle',
                mfunc_ref   = 'Leja',
                mfunc_slope = 0,
                mfunc_mstar0 = 1e6,
                posterior   = False,
                posterior_interp = None,
                lgsfr_grid  = None,
                sample_size = 1,
                mode        = 'ridge',
                sfr_sampling = False,
                completeness_handling ='hybrid',
                sigma_norm  = 1., 
                space_dist  = 'delta',
                z_min       = None,
                z_max       = None,
                zbins       = 100
                ):
    '''
    Generate a sample of star-forming 
    background galaxies using the
    stellar mass-function as sampling
    function weighted by a weight function

    Input:
        - Nsample: # of samples
        - z: array of redshift values 
        - plot_cdf: if True then the cumulative distribution function is plotted
        - logm_min: Minimuma stellar mass
        - logm_max: Maximum stellar mass
        - nbins: # of bins in the integration scheme
        - weight: flag for weight function to be used | options: [None, 'SFR', 'mass']
        - sfr_ref: SFMS parametric formula
        - mfunc_ref: stellar mass function reference
        - mfunc_slope: slope coefficient for artificially changing the stellar mass-function slope
        - mfunc_Mstar0: stellar-mass coefficient below which the stellar mass slope changes artifically
        - posterior: if True then it samples SFR-Mstar posterior distribution
        - posterior_interp: interpolator of the posterior distribution
        - lgsfr_grid: sfr resolution in sampling the posterior
        - sample_size: number of sfr-mstar formula to be sampled from the posterior distribution
        - mode: which parametric formula to use when Leja's SFR is called (ridge/mean)
        - sfr_sampling: If drawing sfr samples is desired (for mcut purposes)
        - completeness_handling: How to treat the sfr-mass relationship below mass completeness threshold ['hybrid', 'cutoff']
        - sigma_norm: if comp_handling='hybrid' then the sigma controls the Gaussian spread below mass comp. threshold 
        - space_dist: distribution of mock galaxies in comoving space
        - z_min: minimum redshift if space_dist != delta
        - z_max: maximum redshift if space_dist != delta
        - zbins: number of bins between zmin and zmax if space_dist != delta


    The mass function should be of the form 


                    Phi dlogM = W * dn / dlogM  dlogM
    
    
    where W is the weight function.

    Returns: 
        - Samples of logMstar
    ''' 
    if mfunc_ref not in ['Leja', 'Schechter'] : raise Exception("The mfunc_ref option you chose is currently not available")
    #----------
    # old way using numerical intergation + root finder    
    
    # def cum_mfunc(z=None, mlow=10**(6.5), mhigh=10**(12.5), bins=3000) :
    #     '''
    #     Compute the cumulative distribution function
    #     of the stellar mass function
    #     '''

    #     if mhigh < mlow or mhigh > 10**(12.5) : return -np.inf
    #     mstar = np.logspace(np.log10(mlow), np.log10(mhigh), bins)
    #     pz = Phi_Leja(logm=np.log10(mstar), z=z)
    #     norm = 0.

    #     #norm = cumtrapz(pz, mstar, initial=0)#[-1]
    #     for i in range(len(mstar) - 1)  : # apply trapezoidal integration -- 2nd order 
    #         norm += (pz[i] + pz[i+1]) * (mstar[i+1] - mstar[i]) / 2. 

    #     print("integral = ", norm, f"mstar={mhigh:.3e}")
    #     return norm


    # def draw_Mstar(r=None, norm=1., z=None, mfunc_ref='Leja'): 

    #     print(f"r={r}")

    #     draw = brentq(lambda x: (cum_mfunc(z=z, mhigh=10**x)/ norm) - r, a=6.5, b=12.5, maxiter=100, xtol=1e-3)
    #     #draw = newton(lambda x: (cum_mfunc(z=z, mhigh=10**x)/ norm) - r, x0 = (6. + 5*r), maxiter=100, tol=1e-3)

    #     return 10**draw 


    # def generate_background_galaxies(Nsample=1, z_arr=[0]) :
    
    #     draws = np.ndarray(shape=(len(z_arr), Nsample), dtype=np.float64)
    #     for i, zi in zip(range(len(z_arr)), z_arr) :
    #         k = 0
    #         norm = cum_mfunc(z=zi)
            
    #         #mstar = np.logspace(6.5, 12.5, 3000)
    #         #pz = Phi_Leja(logm=np.log10(mstar), z=zi)
    #         #cdf_values = cumtrapz(pz, mstar, initial=0)
    #         #cdf_values /= cdf_values[-1]  # Normalize the CDF to 1
    #         # Create an interpolation function for the inverse CDF
    #         #inverse_cdf = interp1d(cdf_values, mstar, bounds_error=False, fill_value=(10**(6.5), 10**(12.5)))


    #         for ri in np.random.random(size=Nsample) :
    #             #draws[i,k] = inverse_cdf(ri) 
    #             draws[i,k] = draw_Mstar(r=ri, norm = norm, z=zi) 
    #             k+=1
        
    #     plt.hist(draws[0], density=True, bins=np.logspace(6,13,100))
    #     #plt.plot(np.linspace(7,12,100) , Phi_Leja(logm=np.linspace(7,12,100), z=zi))

    #     plt.yscale('log'); plt.xscale('log')

    #     return draws


    #-----------
    zprime = 0.3 # it is the redshift at which we compute the density function of sfr-mass-z when z<0.2
    # function that samples the sfr-mass-redshift space
    def draw_sfr(mstar_i=None, z=None, hybrid_sigma=None):
        """
        Samples the star formation rate (SFR) for a given stellar mass, redshift, and standard deviation 
        in SFR based on the mass completeness threshold.

        Parameters:
        ----------
        mstar_i : float
            Stellar mass value in solar masses (Msun) at which the SFR is to be drawn.
        z : float
            Redshift value.
        hybrid_sigma : float
            Standard deviation of SFR in solar masses per year (Msun/yr) computed at the mass completeness threshold.

        Returns:
        -------
        sfr_i : float
            Drawn SFR value in logarithmic scale (log SFR).
        sfr_mode_i : float
            Mode of the Star Forming Main Sequence (SFMS) at the given stellar mass (`mstar_i`).
        """

        posterior_dens = posterior_interp(lgsfr_grid, np.log10(mstar_i)) # this returns a (M,N) matrix
        # flag that determines the prescribed sfr below the mass completeness threshold
        #completeness_handling = 'hybrid' # options ["cutoff", "hybrid"]

        # compute the ridge(mode) value of sfr for given mass and redshift
        sfr_mode_i = utls.SFMS(Mstar=mstar_i, z=z, sfr_ref=sfr_ref, mode='ridge', sample_size=sample_size, posterior=False)
        logm_comp = completeness_threshold(zred=z) # mass completeness threshold at this redshift
        #zprime = 0.3 # the redshift at which we compute the pdf in the case where z<=0.2

        if ( np.log10(mstar_i) < logm_comp - 0.2) and ( completeness_handling == 'cutoff' ) : 
            sfr_i = - np.inf # set it to zero

        elif ( np.log10(mstar_i) < logm_comp - 0.2 ) and ( completeness_handling == 'hybrid' ) : 
            sfr_i = np.random.normal(loc=sfr_mode_i, scale=hybrid_sigma)
        
        elif ( np.log10(mstar_i) < logm_comp - 0.2 ) and ( completeness_handling == 'sharma-like' ) and (z>0.2): 
            posterior_dens_at_mcomp = posterior_interp(lgsfr_grid, logm_comp) #load pdf in logm_comp
            cdf_sfr_posterior_at_mcomp = np.cumsum(np.clip(posterior_dens_at_mcomp, 0, None))
            cdf_sfr_posterior_at_mcomp /= cdf_sfr_posterior_at_mcomp[-1]
            sfr_idx = np.abs(np.random.rand()-cdf_sfr_posterior_at_mcomp).argmin()
            try: 
                sfr_i = lgsfr_grid[sfr_idx]
            except:
                print(sfr_idx)
                raise Exception("sfr_idx out of range")
            # subtract the difference 
            # Delta = sfr_mode at m=mstar - sfr_mode at m=mcomp   
            # this is at the same redshift 
            sfr_i += sfr_mode_i - utls.SFMS(Mstar=10**logm_comp, z=z, sfr_ref=sfr_ref, mode='ridge', sample_size=sample_size, posterior=False)

        elif ( np.log10(mstar_i) < logm_comp ) and (completeness_handling == 'sharma-like') and (z<=0.2) : 
            posterior_dens_at_mcomp = posterior_interp(lgsfr_grid, logm_comp)
            cdf_sfr_posterior_at_mcomp = np.cumsum(np.clip(posterior_dens_at_mcomp, 0, None))
            cdf_sfr_posterior_at_mcomp /= cdf_sfr_posterior_at_mcomp[-1]
            sfr_idx = np.abs(np.random.rand()-cdf_sfr_posterior_at_mcomp).argmin()
            try: 
                sfr_i = lgsfr_grid[sfr_idx]
            except:
                print(sfr_idx)
                raise Exception("sfr_idx out of range")
            # subtract the difference 
            # Delta = sfr_mode at m=mstar and z=z - sfr_mode at m=mcomp and z=zprime   
            sfr_i += sfr_mode_i - utls.SFMS(Mstar=10**logm_comp, z=zprime, sfr_ref=sfr_ref, mode='ridge', sample_size=sample_size, posterior=False)
            

        else :
            #sfr_posterior = posterior_dens
            cdf_sfr_posterior = np.cumsum(np.clip(posterior_dens, 0, None))
            cdf_sfr_posterior /= cdf_sfr_posterior[-1]
            sfr_idx = np.abs(np.random.rand()-cdf_sfr_posterior).argmin()
            try: 
                sfr_i = lgsfr_grid[sfr_idx]
            except:
                print(sfr_idx)
                raise Exception("sfr_idx out of range")
            
            # subtract the difference to account for redshift evolution
            if completeness_handling=='sharma-like' and (z<=0.2) : sfr_i += sfr_mode_i - utls.SFMS(Mstar=mstar_i, z=zprime, sfr_ref=sfr_ref, mode='ridge', sample_size=sample_size, posterior=False)
            
        # add artificial noise to smooth the discreteness of the sampling
        epsilon = 0.1
        sfr_i += epsilon*(2*np.random.rand() - 1) 
        return sfr_i, sfr_mode_i


    # weight function
    def weight_func(mstar=None, z=None, weight=None) :
        
        if weight=='SFR' and (mode!='nn') :
            return 10**utls.SFMS(Mstar=mstar, z=z, sfr_ref=sfr_ref, mode=mode, sample_size=sample_size, posterior=posterior) # star formation rate of star-forming main sequence galaxies
        elif weight=='mass' :
            return mstar/np.max(mstar) #weight by stellar mass
        elif weight=='SFR' and (mode=='nn'):
            
            posterior_dens = posterior_interp(lgsfr_grid, np.log10(mstar)) # this returns a (M,N) matrix
            sfr_sample = np.zeros(shape=(len(mstar),), dtype=float)
            rand_arr = np.random.rand(len(mstar))
            logm_comp = completeness_threshold(zred=z)
            #see draw_sfr() for documentation
            #completeness_handling = 'hybrid' # "cutoff"
            if completeness_handling == 'hybrid' :


                # measure the spread in the sfr values for masses slightly larger
                # then the mass completeness threshold and subsequently 
                # use it to prescribe the spread in sfr for lower mass values
                
                #offset = [12,16,20,int(0.1*len(mstar))]; hybrid_sigma = 0.
                # offset in mass
                offset = [0.2, 0.4, 0.6, 0.8]; hybrid_sigma = 0.
                
                #q_threshold = np.abs(threedhst_mass_completeness(zred=z) - np.log10(mstar)).argmin()
                for offs in offset :

                    q_threshold = np.abs(logm_comp + offs - np.log10(mstar)).argmin()
                    tmp_posterior = np.clip(posterior_dens[q_threshold],0,None) # posterior at the mass threshold value

                    #tmp_posterior = np.clip(posterior_dens[q_threshold+offs],0,None) # posterior at the mass threshold value
                    tmp_cpost = np.cumsum(tmp_posterior)
                    tmp_cpost /= tmp_cpost[-1]
                    sigma_left, sigma_right = lgsfr_grid[np.abs(0.16 - tmp_cpost).argmin()], lgsfr_grid[np.abs(0.84 - tmp_cpost).argmin()]
                    hybrid_sigma += (1/len(offset))*abs(sigma_right - sigma_left) / (2 * sigma_norm) # assume sigma is conserved across different mass values
            
            if completeness_handling == 'sharma-like' :
                #zprime = 0.3
                posterior_dens_at_mcomp = posterior_interp(lgsfr_grid, logm_comp)
                cdf_sfr_posterior_at_mcomp = np.cumsum(np.clip(posterior_dens_at_mcomp, 0, None))
                cdf_sfr_posterior_at_mcomp /= cdf_sfr_posterior_at_mcomp[-1]
    

            for q in range(len(mstar)):
                if ( np.log10(mstar[q]) < logm_comp -0.2 ) and (completeness_handling == 'cutoff') : 
                    sfr_sample[q] = - np.inf # set it to zero
                    continue
                elif ( np.log10(mstar[q]) < logm_comp -0.2 ) and (completeness_handling == 'hybrid') : 
                    tmp_mean = utls.SFMS(Mstar=mstar[q], z=z, sfr_ref=sfr_ref, mode='ridge', sample_size=sample_size, posterior=False)
                    sfr_sample[q] = np.random.normal(loc=tmp_mean, scale=hybrid_sigma)
                    continue
                elif ( np.log10(mstar[q]) < logm_comp - 0.2 ) and (completeness_handling == 'sharma-like') and (z>0.2):
                    sfr_idx = np.abs(np.random.rand()-cdf_sfr_posterior_at_mcomp).argmin()
                    try: 
                        sfr_sample[q] = lgsfr_grid[sfr_idx]
                    except:
                        print(sfr_idx, len(lgsfr_grid))
                        raise Exception("sfr_idx out of range")
                    # subtract the difference 
                    # Delta = sfr_mode at m=mstar - sfr_mode at m=mcomp   
                    # this is at the same redshift 
                    sfr_mode_i = utls.SFMS(Mstar=mstar[q], z=z, sfr_ref=sfr_ref, mode='ridge', sample_size=sample_size, posterior=False)
                    sfr_sample[q] += sfr_mode_i - utls.SFMS(Mstar=10**logm_comp, z=z, sfr_ref=sfr_ref, mode='ridge', sample_size=sample_size, posterior=False)
                    continue
                elif ( np.log10(mstar[q]) < logm_comp - 0.2 ) and (completeness_handling == 'sharma-like') and (z<=0.2):
                    sfr_idx = np.abs(np.random.rand()-cdf_sfr_posterior_at_mcomp).argmin()
                    try: 
                        sfr_sample[q] = lgsfr_grid[sfr_idx]
                    except:
                        print(sfr_idx, len(lgsfr_grid))
                        raise Exception("sfr_idx out of range")
                    # subtract the difference 
                    # Delta = sfr_mode at m=mstar and z=z - sfr_mode at m=mcomp and z=zprime 
                    sfr_mode_i = utls.SFMS(Mstar=mstar[q], z=z, sfr_ref=sfr_ref, mode='ridge', sample_size=sample_size, posterior=False)
                    sfr_sample[q] += sfr_mode_i - utls.SFMS(Mstar=10**logm_comp, z=zprime, sfr_ref=sfr_ref, mode='ridge', sample_size=sample_size, posterior=False)
                    continue

                sfr_posterior = posterior_dens[q]
                cdf_sfr_posterior = np.cumsum(np.clip(sfr_posterior, 0, None))
                cdf_sfr_posterior /= cdf_sfr_posterior[-1]
                sfr_idx = np.abs(rand_arr[q]-cdf_sfr_posterior).argmin()
                try: 
                   sfr_sample[q] = lgsfr_grid[sfr_idx]
                except:
                   print(sfr_idx)
                   raise Exception("sfr_idx out of range")
                # subtract the difference to account for redshift evolution
                if completeness_handling=='sharma-like' and (z<=0.2) : 
                    sfr_sample[q] += utls.SFMS(Mstar=mstar[q], z=z, sfr_ref=sfr_ref, mode='ridge', sample_size=sample_size, posterior=False) - utls.SFMS(Mstar=mstar[q], z=zprime, sfr_ref=sfr_ref, mode='ridge', sample_size=sample_size, posterior=False)

                
                #sfr_sample[q] = - np.inf # if you don't want contribution from nn prediction
                #sfr_sample[q] = (lgsfr_grid[-1] - lgsfr_grid[0])*rand_arr[q] + lgsfr_grid[0] # uniform distribution above mass threshold
            return 10**sfr_sample #sampled from the posterio distribution
        else :
            return 1




    z_values, distances = None, None #to save redshift and distance of each mock galaxy
    if isinstance(z_arr, (int, float)): z_arr = [z_arr]
    if (space_dist=='delta') :
        if not posterior and plot_pdf : plt.figure()
        draws = np.ndarray(shape=(len(z_arr), Nsample), dtype=np.float64)

        mstar = np.logspace(logm_min, logm_max, nbins)
        colors = sns.color_palette("bright", len(z_arr))

        for i, zi in zip(range(len(z_arr)), z_arr) :
            k = 0
            yaxis_max = 0
            
            p_z    = weight_func(mstar=mstar, z=zi, weight=weight) * 10**utls.log_mass_function(Mstar=mstar, z=zi, mfunc_ref=mfunc_ref, mfunc_mstar0=mfunc_mstar0, mfunc_slope=mfunc_slope)
            #p_z    = weight_func(mstar=mstar, z=zi, weight=weight) * utls.Phi_Leja(logm=np.log10(mstar), z=zi) #  weight * (dn/dlogm)
            cdf_pz = cumtrapz(p_z, np.log10(mstar), initial=0) #integrate over logm so that the cdf has units of counts times the units of the weight function
            norm   = cdf_pz[-1]  
            cdf_pz /= norm # Normalize the CDF to 1

            if (plot_cdf) :
                plt.step(np.log10(mstar), cdf_pz, marker='', ls='-', color=colors[i], drawstyle='steps-mid', lw=1.5, label=f'z={zi:.1f}',zorder=0)
                continue

            # Create an interpolation function for the inverse CDF
            inv_cdf = interp1d(cdf_pz, np.log10(mstar), bounds_error=False, fill_value=(np.log10(mstar.min()), np.log10(mstar.max())))

            for ri in np.random.random(size=Nsample) :
                draws[i,k] = inv_cdf(ri) 
                k+=1
            
            if posterior : continue
            print(f'The galaxy sample for z={zi:.3f} has been obtained succesfully.')
            if plot_pdf :
                plt.hist(draws[i], density=True, bins=np.linspace(logm_min,logm_max,80), histtype='step', color=colors[i], alpha=0.5, lw=2)
                plt.plot(np.log10(mstar) , p_z / norm , marker='', ls='--', color=colors[i], lw=1.5, label=f'z={zi:.2f}', zorder=0)
                yaxis_max = max(np.max(p_z/norm), yaxis_max)


        if sfr_sampling :
            # we need to draw sfr values
            # works only if len(z_arr) = 1 
            log_sfr = np.zeros(shape=(len(draws[0]),), dtype=float)
            log_sfr_mode = np.copy(log_sfr)

            # compute 1sigma spread at the mass completeness threshold
            posterior_dens = posterior_interp(lgsfr_grid, np.log10(mstar)) 
            
            #q_threshold = np.abs(threedhst_mass_completeness(zred=zi) - np.log10(mstar)).argmin()
            #offset = [11,16,20,int(0.1*len(mstar))]; hybrid_sigma = 0.
            offset = [0.2, 0.4, 0.6, 0.8]; hybrid_sigma = 0.; logm_comp = completeness_threshold(zred=zi)

            if completeness_handling=='hybrid' :
                for offs in offset:
                    
                    q_threshold = np.abs(logm_comp + offs - np.log10(mstar)).argmin()
                    tmp_posterior = np.clip(posterior_dens[q_threshold],0,None) # posterior at the mass threshold value

                    #tmp_posterior = np.clip(posterior_dens[q_threshold+offs],0,None) # posterior at the mass threshold value
                    tmp_cpost = np.cumsum(tmp_posterior)
                    tmp_cpost /= tmp_cpost[-1]
                    sigma_left, sigma_right = lgsfr_grid[np.abs(0.16 - tmp_cpost).argmin()], lgsfr_grid[np.abs(0.84 - tmp_cpost).argmin()]
                    hybrid_sigma += (1/len(offset))*abs(sigma_right - sigma_left) / (2 * sigma_norm) # assume sigma is conserved across different mass values
                    #print(hybrid_sigma,(1/len(offset))*abs(sigma_right - sigma_left) / 2., q_threshold, offs)
            idx = 0
            for mstar_i in 10**draws[0] : # this works only if draws are obtained for one redshift value
                log_sfr[idx], log_sfr_mode[idx] = draw_sfr(mstar_i=mstar_i, z=zi, hybrid_sigma=hybrid_sigma)
                idx += 1

        if posterior and sfr_sampling : 
            return draws[0], log_sfr, log_sfr_mode
        elif posterior and not sfr_sampling : 
            return draws

        if plot_pdf :
            plt.title("Cumulative density function of stellar mass function")
            plt.ylabel("$\int_{}^{M_\star} \mathrm{\Phi_\star (M,z) ~ dM}$")
            plt.xscale('linear')
            plt.xlim(logm_min,logm_max + 0.5)
            plt.legend()
            plt.xlabel("$\mathrm{\log M_\star ~~[M_\odot]}$")
        if not plot_cdf and plot_pdf : 
            plt.yscale('log') 
            #ymax = 3*Phi_Leja(logm=logm_min, z=zi) / norm
            plt.ylim(1e-6 * yaxis_max, 10*yaxis_max)
            plt.title("Galaxy stellar mass samples from the chosen sampling function")
            plt.ylabel("$\mathrm{P(M_{\star}, z) ~ [arb.]}$")
            plt.text(x= 1 + logm_min, y=1e-5*yaxis_max, \
                    s=f"Number of galaxies = {Nsample:.0f}\nWeighted by {weight}\nsfr_ref={sfr_ref}\nmfunc_ref={mfunc_ref}"+"\n$M_\star \in [10^{6.5}, 10^{12.5}]$", color='gray')

        if plot_pdf :
            plt.tight_layout()
            # save figure
            #if not plot_cdf : plt.savefig('figs/sample_galaxy_masses.png', format='png')
            plt.show()

        if plot_cdf :
            print('Sample generator has been activated\n')
            Gen_mock_gal(Nsample=10000, z_arr=[0.,2.,4.], plot_cdf=False, weight=weight)


    

    elif (space_dist=='uniform') :
        raise Exception("outdated 'space_dist=delta' script. please update it!")

        plt.figure()
        draws = np.ndarray(shape=(len(z_min), Nsample), dtype=np.float64)
        distances = np.ndarray(shape=(len(z_min), Nsample), dtype=np.float64)
        z_values = np.ndarray(shape=(len(z_min), Nsample), dtype=np.float64)

        mstar = np.logspace(logm_min, logm_max, nbins)
        colors = sns.color_palette("bright", len(z_min))

        for i in range(len(z_min)) :
            k = 0
            yaxis_max = 0
            
            dc_min = cosmo.comoving_distance(z_min[i]).value #minimum comoving distance
            dc_max = cosmo.comoving_distance(z_max[i]).value #mximum comoving distance
            kappa = (dc_min / dc_max)**3 
            for ri, di in zip(np.random.random(size=Nsample), (np.random.random(size=Nsample)*(1-kappa) + kappa)**(1/3) * dc_max)  :
                '''
                    - ri: random number used to sample the stellar mass sampling function
                    - di: comoving distance sampled from a spatial distribution function
                '''
                
                zi = z_at_value(cosmo.comoving_distance, di * units.Mpc) # invert the comoving distance = redshift relation

                # sample a stellar mass
                p_z    = weight_func(mstar=mstar, z=zi, weight=weight) * utls.Phi_Leja(logm=np.log10(mstar), z=zi) #  weight * (dn/dlogm)
                cdf_pz = cumtrapz(p_z, np.log10(mstar), initial=0) #integrate over logm so that the cdf has units of counts times the units of the weight function
                norm   = cdf_pz[-1]  
                cdf_pz /= norm # Normalize the CDF to 1

                if (plot_cdf) :
                    plt.step(np.log10(mstar), cdf_pz, marker='', ls='-', color=colors[i], drawstyle='steps-mid', lw=1.5, label=f'z={zi:.1f}',zorder=0)
                    continue

                # Create an interpolation function for the inverse CDF
                inv_cdf = interp1d(cdf_pz, np.log10(mstar), bounds_error=False, fill_value=(np.log10(mstar.min()), np.log10(mstar.max())))
                draws[i,k] = inv_cdf(ri) #store logm_star
                distances[i,k] = di #store comoving distance
                z_values[i,k] = zi #store redshift
                k+=1

            print(f'The galaxy sample for z={zi:.1f} has been obtained succesfully.')
            plt.hist(draws[i], density=True, bins=np.linspace(logm_min,logm_max,80), histtype='step', color=colors[i], alpha=0.5, lw=2)
            plt.plot(np.log10(mstar) , p_z / norm , marker='', ls='--', color=colors[i], lw=1.5, label=f'z={zi:.1f}', zorder=0)
            yaxis_max = max(np.max(p_z/norm), yaxis_max)


        plt.title("Cumulative density function of stellar mass function")
        plt.ylabel("$\int_{}^{M_\star} \mathrm{\Phi_\star (M,z) ~ dM}$")
        if not plot_cdf : 
            plt.yscale('log') 
            #ymax = 3*Phi_Leja(logm=logm_min, z=zi) / norm
            plt.ylim(1e-6 * yaxis_max, 10*yaxis_max)
            plt.title("Galaxy samples from the stellar mass function")
            plt.ylabel("$\mathrm{P(M_{\star}, z) ~ [arb.]}$")
            plt.text(x= 1 + logm_min, y=1e-5*yaxis_max, \
                s=f"Number of galaxies = {Nsample:.0f}\nWeighted by {weight}\nsfr_ref={sfr_ref}\nmfunc_ref={mfunc_ref}"\
                    +"\n$M_\star \in [10^{6.5}, 10^{12.5}]$"+f"\nSpatial distribution: {space_dist}", color='gray')
        
        plt.xscale('linear')
        plt.xlim(logm_min,logm_max + 0.5)
        plt.legend()
        plt.xlabel("$\mathrm{\log M_\star ~~[M_\odot]}$")
        plt.tight_layout()
        # save figure
        #if not plot_cdf : plt.savefig('figs/sample_galaxy_masses.png', format='png')
        plt.show()

        if plot_cdf :
            print('Sample generator has been activated\n')
            Gen_mock_gal(Nsample=10000, z_arr=[0.,2.,4.], plot_cdf=False, weight=weight)
      

    return draws, distances, z_values #logm_star, distance, redshift arraya




def magnitude_cut(log_mstar = [], 
                  log_sfr = [],
                  log_sfr_mode = [],
                  z = None, 
                  rmag_cut = None, 
                  plot = False,
                  z_values = None,
                  space_dist = 'delta',
                  ml_sampling = 'uniform',
                  prescribed_ml_func = None, 
                  density_sfr_color = None,
                  color_gr_grid = None,
                  sfr_grid = None,
                  Kr_correction = False
                  ):
    """
    Apply the magntitude-limit threshold
    Based on Sharma et al. (2024)
    Use of the sun's absolute magntitude found in:
    https://iopscience.iop.org/article/10.3847/1538-4365/aabfdf (Table 3)
    See also: https://mips.as.arizona.edu/~cnaw/sun_2006.html
    
    - Generate mass-to-light ratio values following a normal distribution
    - Convert to absolute magntiude in r-band
    - use distance modulus to obtain aparent magnitude
    - Incorporate extinction/reddening corrections (under constuction)
    - Apply selection criterion

    Extension:
        - Use sfr information to convert into M/L ratio (e.g., quiescent galaxies have higher M/L ratio)

    Input:
        - log_mstar: array of mstar values
        - log_sfr: array of sfr values
        - log_sfr_mode: array of SFMS values 
        - z: redshift of the galaxy population
        - rmag_cut: magnitude threshold driven by observations
        - plot: plot flag (set True if a histogram is desired)
        - ml_sampling : which mode to use in order to sample mass-to-light ratio values (available : 'prescribed', 'advanced')
        - prescribed_ml_func : functional used to compute M/L for given sfr, sfr_mode (active only when ml_sampling='prescribed')
        - density_sfr_color: prob. density in sfr-color space (from Chang et al.)
        - color_gr_grid: color array for color-sfr grid
        - sfr_grid: sfr array for color-sfr grid -- in logarithmic form
        - Kr_correction: flag to apply Kr-correction (works only if ml_sampling='advanced')

    Returns:
        - color_sample: if ml_sampling='advanced', then a color value for each galaxy is sampled and returned
        - Kcorr_sample: if ml_sampling='advanced' and kr_correction=True, then a Kcorr value for each galaxy is estimated and returned
        - ml_sample: if ml_sampling='advanced', then a M/Lr value for each galaxy is sampled and returned
        - rmag: an array of the r-band magnitudes for each galaxy (without the cutoff filter)
        - ind: a list of indices with galaxies of rmag<rmag_cut 
    """

    Mr_sun = 4.65 #absolute magnitude of sun in r-band
    # Update : November 1, 2024
    # Sharma is using the following convention for the normalization 
    # of the mass-to-light ratio
    #        M/L  =  M/Msun / L_r/Lsun
    # Thus, if log(M/L) ~ N(0,sigma)
    #         log(M/Msun / L_r/ L_r,sun) ~ N(log(0.13), sigma)
    mu_sharma = np.log10(1.) # this is for M/Lr with Lr normalized to Lbol,sun

    # available spatial distributions
    avail_space_dists = ['delta', 'uniform-z', 'uniform-vol']

    #set up the mass-to-light ratio distribution
    size = len(log_mstar)
    mu, sigma = mu_sharma, None # mean and standard deviation
    rng = np.random.default_rng() # random number generator

    #compute luminosity distances
    if space_dist=='delta' :
        lum_dist = cosmo.luminosity_distance(z=z).value # in [Mpc]
        z_values = [z]*size # needed to compute kr-correction and sample mass-to-light ratio
    else :
        try :
            len(z_values) == size
            lum_dist = cosmo.luminosity_distance(z=z_values).value # in [Mpc]
        except :
            raise Exception('The magnitude cut failed. It seems like you asked for a non-delta spatial distribution without specifying the redshifts')


    # if no sfr information is provided then skip mass to light ratio sampling from probabaility density
    # instead implement the prescription of Sharma et al. (2024)
    if (space_dist in avail_space_dists) and ml_sampling not in ['prescribed', 'advanced'] :
        if z <= 0.2 : sigma = 0.2 
        elif z > 0.2 and z <= 0.4 : sigma = 0.26 
        elif z <= 0.7 and z > 0.4 : sigma = 0.3 
        
        # generate a mass-to-light ratio sample
        MtoL_sample = rng.normal(loc=mu, scale=sigma, size=size)

        # absolute and apparent magnitude of each galaxy in r-band
        Mr_sample = Mr_sun - 2.5 * (log_mstar - np.log10(Lum_r_sun) - MtoL_sample)
        rmag = Mr_sample + 5*np.log10(lum_dist) + 25 # provided lum_dist is measured in Mpc
    
    # elif (space_dist'delta') and (len(log_sfr) != 0) and ml_sampling not in ['prescribed', 'advanced'] :
    #     if z <= 0.2 : sigma = 0.2 
    #     elif z > 0.2 and z <= 0.4 : sigma = 0.26 
    #     elif z <= 0.7 and z > 0.4 : sigma = 0.3 
        
    #     # generate a mass-to-light ratio sample
    #     MtoL_sample = rng.normal(loc=mu, scale=sigma, size=size)

    #     # absolute and apparent magnitude of each galaxy in r-band
    #     Mr_sample = Mr_sun - 2.5 * np.log10(10**(log_mstar - MtoL_sample)) 
    #     rmag = Mr_sample + 5*np.log10(lum_dist) + 25 # provided lum_dist is measured in Mpc

    elif (space_dist in avail_space_dists) and (len(log_sfr) != 0) and (ml_sampling=='prescribed') :
        if z <= 0.2 : sigma = 0.2 
        elif z > 0.2 and z <= 0.4 : sigma = 0.26 
        elif z <= 0.7 and z > 0.4 : sigma = 0.3 

        # SFR dependent mass-to-light ratio
        max_MtoL = np.log10(10.) # maximum mass-to-light ratio
        #define th prescribed M/Lr - sfr function 
        if prescribed_ml_func is None : MtoL_correction = lambda sfr_i, sfr_mode : max_MtoL * np.heaviside(sfr_mode - sfr_i, 0) #    min(np.heaviside(sfr_mode - sfr_i, 0) * (0.5 * (sfr_mode - sfr_i)), max_MtoL)
        #MtoL_correction = lambda sfr_i, sfr_mode : min(np.heaviside(sfr_mode - sfr_i, 0) * (0.5 * (sfr_mode - sfr_i)), max_MtoL)
        else : MtoL_correction = prescribed_ml_func

        # generate a mass-to-light ratio sample centered at mu with spread sigma
        MtoL_sample = rng.normal(loc=mu, scale=sigma, size=size)  # in logarithmic form
        for i in range(len(MtoL_sample)) :
            # take into acount M/L corrections due to SFR
        
            # add MtoL correction factor
            MtoL_sample[i] += MtoL_correction(log_sfr[i], log_sfr_mode[i])

        # absolute and apparent magnitude of each galaxy in r-band
        Mr_sample = Mr_sun - 2.5 * (log_mstar - np.log10(Lum_r_sun) - MtoL_sample)
        rmag = Mr_sample + 5*np.log10(lum_dist) + 25 # provided lum_dist is measured in Mpc
    

    # sample the probability density space of sfr-mstar to obtain an M/L ratio
    elif (space_dist in avail_space_dists) and (len(log_sfr) != 0) and (ml_sampling=='advanced') :

        MtoL_sample       = np.zeros(size, dtype=float)
        Kcorr_sample      = np.copy(MtoL_sample)
        color_sample      = np.copy(MtoL_sample)
        MtoLg_rest_sample = np.copy(MtoL_sample)

        #out = np.zeros(size, dtype=float)
        #out2 = np.zeros(size, dtype=float)
        # SFR dependent mass-to-light ratio
        for i in range(size) :
            # take into acount M/L corrections due to SFR        

            #Step 1: Sample rest-frame M/Lg & g-r 
            ## note that MtoLg here is not in logarithmic form. MtoLg units are (Msun/Lg,sun)
            MtoLg_sample, color_gr, _, _ = utls.sample_mass_to_light(sfr=log_sfr[i], z=z_values[i], density_sfr_color=density_sfr_color, color_arr=color_gr_grid, sfr_arr=sfr_grid)
            ## Change normalization units from Lg_sun to L_sun
            MtoLg_sample = 1. / utls.Lg_Lg_sun_to_Lg_L_sun(Lg_in_Lg_sun=1./MtoLg_sample)
            #Step 2: Convert to rest-frame M/Lr
            MtoLr_sample = utls.from_MtoLg_to_MtoLr(g_r=color_gr, MtoLg=MtoLg_sample)
            #Step 3: Apply Kr-correction to get the observed M/Lr
            if Kr_correction : K_corr = utls.Kr_correction(g_r=color_gr, z=z_values[i])
            else : K_corr = 0.
            # remember that Lr is normalized to Lbol,sun and not Lr_sun here! Be cautious.
            MtoL_sample[i] = np.log10(MtoLr_sample) - (K_corr/2.5) #MtoL_sample is in logarithmic scale
            Kcorr_sample[i] = K_corr ; color_sample[i] = color_gr
            MtoLg_rest_sample[i] = np.log10(MtoLg_sample)
            #out[i], out2[i] = color_gr, MtoL_sample[i]

        # absolute and apparent magnitude of each galaxy in r-band
        # it is important to normalize MtoLr is Lr,sun units in order to use the following formula.
        Mr_sample = Mr_sun - 2.5 * (log_mstar - np.log10(Lum_r_sun) - MtoL_sample)
        rmag = Mr_sample + 5*np.log10(lum_dist) + 25 # provided lum_dist is measured in Mpc

        # ##inspect the sampled color-MtoL values
        # plt.figure()
        # plt.scatter(out, out2, marker='.', s=1,alpha=0.1)
        # plt.xlabel("rest-frame g-r")
        # plt.ylabel("$rest-frame \mathrm{M/L_g}$")
        # plt.show()

    # apply selection critetion 
    ind = rmag<=rmag_cut

    # plot distribution of rmag
    if plot : 
        plt.figure()
        plt.hist(Mr_sample, bins=100, alpha=0.7, density=True, color='lightgray', label='Absolute rmag')
        plt.hist(rmag, bins=np.linspace(15,35,300), density=True, alpha=0.7, color='green', label='rmag w/o selection filter')
        print("maximum of rmag before cut=", np.max(rmag))
 
        plt.hist(rmag[ind], bins=np.linspace(15,35,300), density=True, color='red', alpha=0.7, label='rmag w/ selection filter')
        plt.axvline(x=rmag_cut, color='black', ls=':', lw=1, zorder=0, marker='', alpha=0.5, label='rmag_cut')
        print("after cut = ", np.max(rmag[ind]))
        plt.xlabel("r-band magnitude")
        plt.ylabel('PDF')
        plt.legend()
        plt.tight_layout()
        #plt.savefig('figs/mock_rmag_distribution.png', format='png')
        plt.show()
    

    try :
        return color_sample, Kcorr_sample, MtoL_sample, MtoLg_rest_sample, ind 
    except :
        try:
            return MtoL_sample, ind 
        except :
            return rmag, ind #complete list of rmag, indices where rmag<threshold
        



def mock_realization(zbins = [0.,0.3, 0.7], 
                 zgal = [0.15, 0.45], 
                 Nsample = 100000, 
                 weight = None, 
                 save = False, 
                 mfunc_ref = 'Leja', 
                 mfunc_slope = 0,
                 mfunc_mstar0 = 1e6,
                 sfr_ref = 'Speagle',
                 mode='ridge',
                 posterior=False, 
                 plot_cdf_ridge = True,
                 completeness_handling='hybrid',
                 sigma_norm=1.,
                 n_realizations = 100,
                 transparency = 0.01, 
                 data_source = None, 
                 ks_test = False,
                 sfr_sampling = False,
                 space_dist = 'delta',
                 nz_bins = 100,
                 z_min = None,
                 z_max = None,
                 p_dens_params = None,
                 p_prob_arr = None, 
                 p_z_arr = None, 
                 p_logm_arr = None, 
                 p_logsfr_arr = None,
                 ml_sampling='prescribed',
                 prescribed_ml_func=None,
                 density_sfr_color=None,
                 sfr_grid=None,
                 color_gr_grid=None,
                 Kr_correction=False,
                 plot_M_L=False,
                 store_output=False
                 ):
    
    """
    This function makes use of the mock star forming galaxies generator
    and yields comparison plots after applying selection criteria
    such as magntitude-limit cut etc.

    You can modify, add or remove whichever part of the code you wish.
    
    This code has the ability to take into account uncertaintis by
    sampling the posterior distribution function.
    
    Input parameters:
        - zbins: the redshift values separating the bins (number of bins = len(zbins) - 1)
        - zgal: the redshift at which each mock galaxy lies for each bin (applicable when space_dist='delta')
        - Nsample: number of galaxy samples per mock galaxy realization
        - weight: weight function to be used in the samplng distribution function (current options: 'SFR', 'mass', 'uniform')
        - save: flag for storing the figure
        - mfunc_ref: Which stellar mass function to be implemented in the calculation ('Leja', 'Schechter')
        - mfunc_slope: slope coefficient for artificially changing the stellar mass-function slope
        - mfunc_Mstar0: stellar-mass coefficient below which the stellar mass slope changes artifically
        - sfr_ref: Which star-forming main sequence formula to be used (if weight == 'SFR') ['Speagle', 'Leja', ]
        - mode: which mode of the sfr-Mstar parametric formula to be considered (available: 'ridge', 'mean', 'nn')
        - posterior: If True, then sampling of the posterior is activated,
        - plot_cdf_ridge: Flag to activate plotting of the cdf curved computed using mode='ridge', i.e., no posterior  
        - completeness_handling: if mode='nn', then it determines the prescribed sfrmass below m_completeness ['hybrid', 'cutoff']
        - sigma_norm: it controls the spread in the sfr-mass below m_completeness  
        - n_realizations: how many different samples from the posterior are to be generated
        - transparency: 'alpha' for each line in the plot
        - data_source: FRB host galaxy sample ['Sharma_only', 'Sharma_full']
        - ks_test: If True, the Kolmogorov-smirnoff test is carried out (not in the samples of the posterior, but in the ridge mode results)
        - sfr_sampling: flag to sample sfr value for each mock generated galaxy
        - space_dist: Distribution of mock galaxies in space ('delta' or 'uniform'). The 'uniform' mode is outdated.
        - z_min: array of min redshift value in each redshift bin
        - z_max: array of max redshift value in each redshift bin
        - p_dens_params: various parameters used to set up the PDF in logm-logsfr-z,
        - p_prob_arr: PDF in logm-logsfr-z space (Leja et al. 2020) 
        - p_z_arr: redshift array in logm-logsfr-z grid space 
        - p_logm_arr: logM array in logm-logsfr-z grid space 
        - p_logsfr_arr: logsfr array in logm-logsfr-z grid space 
        - ml_sampling: If mass-to-light ratio is to be sampled. Options: ['prescribed', 'advanced']
        - prescribed_ml_func: Prescribed lambda function to compute M/L for given sfr, sfr_mode, it is used only if ml_sampling='prescribed'
        - density_sfr_color: prob. density in sfr-color space provided M/L sampling is activated
        - sfr_grid: sfr array (in log form) for sfr-color grid resolution provided M/L sampling is activated
        - color_gr_grid: color g-r array for sfr-color grid resolution provided M/L sampling is activated
        - Kr_correction: flag to activate Kr-correction
        - plot_M_L: flag to plot the mass-to-light distribution and other related quantities
        - store_output: flag to store all samples into a h5 data file
    
    Returns:
    """
    if mode != 'nn' : sfr_sampling = False # it makes no sense to sample sfr if the chosen mode is not 'nn'
    # Print the input parameters with better formatting
    print("\nInput Parameters:") 
    print(f"  zbins:                 {zbins}")
    print(f"  zgal:                  {zgal}")
    print(f"  Nsample:               {Nsample}")
    print(f"  weight:                {weight}")
    print(f"  save:                  {save}")
    print(f"  mfunc_ref:             {mfunc_ref}")
    print(f"  mfunc_mstar0:          {mfunc_mstar0}")
    print(f"  mfunc_slope:           {mfunc_slope}")
    print(f"  sfr_ref:               {sfr_ref}")
    print(f"  mode (sfr):            {mode}")
    print(f"  posterior(sampling):   {posterior}")
    print(f"  completeness_handling: {completeness_handling}")
    print(f"  sigma_norm:            {sigma_norm}")
    print(f"  n_realizations:        {n_realizations}")
    print(f"  nz_bins:               {nz_bins}")
    print(f"  transparency:          {transparency}")
    print(f"  data_source:           {data_source}")
    print(f"  ks_test:               {ks_test}")
    print(f"  sfr_sampling:          {sfr_sampling}")
    print(f"  space_dist:            {space_dist}")
    print(f"  z_min (spatial):       {z_min}")
    print(f"  z_max (spatial):       {z_max}")
    print(f"  p_dens_params:         {p_dens_params}")
    print(f"  ml_sampling:           {ml_sampling}")
    print(f"  prescribed_ml_func:    {prescribed_ml_func}")
    print(f"  Kr_correction:         {Kr_correction}")
    print(f"  plot_M_L:              {plot_M_L}")
    print(f"  store_output:          {store_output}")
    print("---------------\n")

    # log_mstar_sample shape : (len(z_gal), Nsample)
    avail_space_dists = ['delta', 'uniform-z', 'uniform-vol']
    if plot_cdf_ridge : 
        # if you choose to use the neural network it 
        # will first plot the cdf using the ridge and
        # then on top of it will plot the one sampled
        # from the neural network posterior
        # no_post_mode is a parameter inserted in the Gen_mock_gal()
        # to determine which mode will be used in the SFR
        if mode=='nn' : no_post_mode = 'ridge' # to compute the mean cumulative distribution function
        else : no_post_mode = mode 

        if space_dist not in avail_space_dists :
            raise Exception(f"Currently available space distributions are {avail_space_dists}, but you chose {space_dist}. Please choose one of the viable ones.")
            # check once that the chosen space_dist is within the available options and don't check again for it throughtout the code
        if space_dist=='delta' :
            log_mstar_samples, _, z_values = Gen_mock_gal(Nsample=Nsample, z_arr=zgal, plot_cdf=False, weight=weight,\
                                                                    mfunc_ref=mfunc_ref, mfunc_mstar0=mfunc_mstar0, \
                                                                        mfunc_slope=mfunc_slope, sfr_ref=sfr_ref, mode=no_post_mode, sfr_sampling=False,\
                                                                        space_dist=space_dist, z_min=z_min, z_max=z_max) 
        elif space_dist in ['uniform-z', 'uniform-vol'] :
            Nsubsample = int(Nsample/nz_bins)
            log_mstar_samples = []; z_values = []; zij_bins = []
            for i in range(len(zgal)):
                # generate an array of z-values for the mock galaxy samples that follows a desired distribution
                if space_dist == 'uniform-z' : zi_bins = np.random.uniform(zbins[i], zbins[i+1], nz_bins)
                elif space_dist == 'uniform-vol' :
                    # here we pursue an r^2 distribution in comoving radial distance, so we first have to sample
                    # distances and then convert them to redshifts 
                    # dc_sample: comoving distances sampled from a spatial distribution function
                    dc_min = cosmo.comoving_distance(zbins[i]).value #minimum comoving distance
                    dc_max = cosmo.comoving_distance(zbins[i+1]).value #mximum comoving distance
                    kappa = (dc_min / dc_max)**3 
                    dc_sample = (np.random.random(size=nz_bins)*(1-kappa) + kappa)**(1/3) * dc_max # in Mpc
                    zi_bins = z_at_value(cosmo.comoving_distance, dc_sample * units.Mpc).value # invert the comoving distance = redshift relation        
                
                for zij in zi_bins:
                    log_mstar_sample, _, _ = Gen_mock_gal(Nsample=Nsubsample, z_arr=zij, plot_cdf=False, weight=weight,\
                                                                    mfunc_ref=mfunc_ref, mfunc_mstar0=mfunc_mstar0, \
                                                                        mfunc_slope=mfunc_slope, sfr_ref=sfr_ref, mode=no_post_mode, sfr_sampling=False,\
                                                                        space_dist='delta', z_min=z_min, z_max=z_max, plot_pdf=False)
                    # add subsample to the grand sample
                    log_mstar_samples.append(log_mstar_sample)
                    z_values.append([zij]*Nsubsample)
                
                zij_bins.append(zi_bins) # to be used later when generating subsamples of the posterior distribution
            
            # marge subsamples and reshape output to create a final grand sample
            log_mstar_samples = np.array(log_mstar_samples).reshape(3,-1)
            z_values = np.array(z_values).reshape(3,-1)
            del log_mstar_sample, zij, zi_bins # delete variables

    # load FRB data
    #corr_mstar0, corr_lfSFR0, corr_ztransient
    #corr_mstar, corr_ztransient = m_sfr_data[0,:], m_sfr_data[3,:]
    if data_source in ['Sharma_only', 'Sharma_full'] : 
        frbdata_mstar, frbdata_ztransient = utls.load_Sharma(desc=data_source) # it returns the stellar masses and redshifts of Sharma FRB host data
    else : raise Exception(f"The FRB host data source : <{data_source}> is not available. Try again.")
    #else : corr_mstar, corr_ztransient = m_sfr_data[0,:], m_sfr_data[3,:]


    # if the posterior of Leja is to be utilized, execute the following command to load the neural network and sample the probability density
    # if (posterior) and (space_dist=='delta') and (mode=='nn') :

    ## Update:
    # this has been transfered to the input to save time as computing the pdf is computationally expensive
    # you just have to do it once and you are done :)

        # print("loading the trained neural network and compute the probability density function in the logmstar-logsfr-redshift space")
        # p_dens_params = {
        #     "nlogm"  : 100,#120, 
        #     "nsfr"   : 100,#140,
        #     "dz"     : 0.1,#0.1,
        #     "ndummy" : 31,
        #     "mmin"   : 6.5, 
        #     "mmax"   : 12.5,
        #     "sfrmin" : -3.,
        #     "sfrmax" : 3.,
        #     "zmin"   : 0.2, 
        #     "zmax"   : 3.0
        # }

        # # load the probability distribution function
        # p_prob_arr, p_z_arr, p_logm_arr, p_logsfr_arr = load_leja_posterior(dens_params=p_dens_params)
        # print('finish computing the probability density')

    # make a directory to store the output + figures
    if store_output: 
        # Check if file exists
        part1 = f'./output/mf{mfunc_ref}_sf{sfr_ref}'
        part2 = f'_s{sigma_norm}' if mode=='nn' and completeness_handling=='hybrid' else ''
        part2 = f'_ch{completeness_handling}' if mode=='nn' and completeness_handling=='sharma-like' else ''
        subpart3 = f'nzbins{nz_bins}' if space_dist != 'delta' else ''
        part3 = f'_m{mode}_W{weight}_N{Nsample}_n{n_realizations}_sd{space_dist}_{subpart3}_ml{ml_sampling}'
        part4 = f'_k{Kr_correction}' if sfr_sampling==True else ''
        end = '_0/'
        folder_path = part1 + part2 + part3 + part4 + end

        #folder_path = f'./output/mf{mfunc_ref}_sf{sfr_ref}_s{sigma_norm}_m{mode}_W{weight}_N{Nsample}_n{n_realizations}_sd{space_dist}_ml{ml_sampling}_k{Kr_correction}_0/'
        count = 1; exists = True
        while exists :
            if os.path.exists(folder_path):
                folder_path = part1 + part2 + part3 + part4 + f'_{count}/'
                #folder_path = f'./output/mf{mfunc_ref}_sf{sfr_ref}_s{sigma_norm}_m{mode}_W{weight}_N{Nsample}_n{n_realizations}_sd{space_dist}_ml{ml_sampling}_k{Kr_correction}_{count}/'
                count += 1
            else : exists = False
        os.makedirs(folder_path)
        del count, exists, part1, part2, part3, subpart3, part4, end

        print(f"Data will be stored in {folder_path}")

    
    # set up the figure configuration
    # if space_dist != 'delta' : zgal = np.zeros(len(z_min))
    plt.subplots(1,len(zgal), figsize=(11,4), sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0})
    plt.subplot(101+10*len(zgal))
    plt.ylabel('CDF')

    MtoL_samples, logm_samples, lgsfr_samples, lgsfr_mode_samples, color_samples, Kcorr_samples, MtoLg_rest_samples, magcut_ind_flags, redshift_samples = [], [], [], [], [], [], [], [], []
    if ks_test : ks_pvalues = np.zeros(shape=(len(zgal), n_realizations), dtype=float)
    massbins = np.linspace(6.4,12.6,200) # set the stellar mass resolution 
    for i in range(len(zgal)) : # iterate over all different redshift bins

        plt.subplot(101 + 10*(len(zgal)) + i)
        idx = (zbins[i]<frbdata_ztransient) & (frbdata_ztransient<zbins[i+1])  # this selects the FRB hosts which lie in the desired chosen redshift range.
        if plot_cdf_ridge :
            sample = log_mstar_samples[i]
            if space_dist != 'delta': _, ind_magcut = magnitude_cut(log_mstar=sample, z=zgal[i], rmag_cut=23.5, plot=False, z_values=z_values[i], space_dist=space_dist)
            else : 
                try :
                    _, ind_magcut = magnitude_cut(log_mstar=sample, z=zgal[i], rmag_cut=23.5, plot=False, space_dist=space_dist)
                except :
                    print(f"The code crashed when the redshift was {zgal[i]} and you should check the magnitude_cut() routine")
                    raise Exception("Test the magnitude_cut() function.")
            #sfr_samples = SFMS(Mstar=10**log_mstar_samples[i], z=zgal[i])

            # plot the whole sample's cdf
            plt.hist(sample, cumulative=True, density=True, histtype='step', ls='--', color='green', bins=massbins, lw=1., label='mock galaxies w/o rmag limit')
            # applyt the selection cut and plot the cdf
            plt.hist(sample[ind_magcut], cumulative=True, density=True, histtype='step', ls='-', color='green', bins=massbins, lw=2, label='mock galaxies w/ rmag limit')

        # plot the FRB host data
        plt.hist(np.log10(frbdata_mstar[idx]), cumulative=True, density=True, histtype='step', color='black', bins=massbins, lw=2, label='FRB hosts')
        plt.title(f"{zbins[i]} $< z \leq $ {zbins[i+1]}")
        plt.text(x=np.min(massbins)+0.8, y=0.4, \
                s="N of FRBs = " + f"{len(frbdata_mstar[idx]):.0f}\nWeighted by {weight}\nsfms={sfr_ref}\nmfunc={mfunc_ref}", \
                fontsize=10, color='gray')
        plt.xlabel("$\mathrm{\log M_\star ~ [M_\odot]}$"); plt.xlim(6.9,12.1); plt.ylim(0.,1.0)
        #if i>0 : plt.gca().set_yticklabels([])

        # run ks-test
        if (ks_test) and (plot_cdf_ridge) : 
            """
            Notes related to hypothesis testing can be found here: 
            https://github.com/astrostatistics-in-crete/2024_summer_school/blob/main/02_Hypothesis/Hypothesis.ipynb
            """
            pvalue = ks_2samp(np.log10(frbdata_mstar[idx]), sample[ind_magcut], alternative='two-sided', method='auto').pvalue
            print(f"The ks-test has returned a p-value equal to {pvalue:.2e}.")
            plt.text(x=np.max(massbins)-2.3, y=0.06, s=f"p-value={pvalue:.1e}", fontsize=10, color='blue')




        # generate samples from the posterior distribution and plot them on top of the ridge lines
        if (posterior) and (mode!='nn'): # it is currently broken for spatial distributions different than delta dirac
            nlines = n_realizations
            alpha = transparency

            for n in tqdm(range(nlines), desc=f"Sampling posterior for {zbins[i]} $<$ z $\leq$ {zbins[i+1]}") : 
                if space_dist == 'delta':
                    post_sample = Gen_mock_gal(Nsample=Nsample, z_arr=zgal[i], plot_cdf=False, weight=weight, mfunc_ref=mfunc_ref, mfunc_slope=mfunc_slope, mfunc_mstar0=mfunc_mstar0, \
                                                        sfr_ref=sfr_ref, mode=mode, posterior=True, space_dist=space_dist, z_min=z_min, z_max=z_max)[0]
                
                elif space_dist in ['uniform-z', 'uniform-vol']:
                    post_sample = []; post_z_values = []
                    Nsubsample = int(Nsample/nz_bins)

                    for zij in zij_bins[i] :
                        post_subsample = Gen_mock_gal(Nsample=Nsubsample, z_arr=zij , plot_cdf=False, weight=weight, mfunc_ref=mfunc_ref, mfunc_slope=mfunc_slope, mfunc_mstar0=mfunc_mstar0, \
                                                        sfr_ref=sfr_ref, mode=mode, posterior=True, space_dist='delta', z_min=z_min, z_max=z_max)[0]
                        post_sample.append(post_subsample)
                        post_z_values.append([zij]*Nsubsample)
                    # merge subsumples to generate one realization for given redshift range
                    post_sample = np.array(post_sample).flatten()
                    post_z_values = np.array(post_z_values).flatten()
                
                # add realization to the grand sample for given redshift range
                logm_samples.append(post_sample)
                if space_dist=='delta' : post_z_values = None
                ml_sample, ind_magcut = magnitude_cut(log_mstar=post_sample, z=zgal[i], rmag_cut=23.5, plot=False, z_values=post_z_values,  space_dist=space_dist)
                MtoL_samples.append(ml_sample); magcut_ind_flags.append(ind_magcut)
                plt.hist(post_sample, cumulative=True, density=True, histtype='step', ls='--', color='red', bins=massbins, lw=1., alpha=alpha, zorder=0)
                plt.hist(post_sample[ind_magcut], cumulative=True, density=True, histtype='step', ls='-', color='red', bins=massbins, lw=1., alpha=alpha, zorder=0)
                try :
                    ks_pvalues[i, n] = utls.run_kstest(post_sample[ind_magcut], np.log10(frbdata_mstar[idx]))
                except :
                    pass

                try :
                    redshift_samples.append(post_z_values)
                except :
                    pass

        elif (posterior) and (mode=='nn') :
            
            nlines = n_realizations
            alpha = transparency

            # load a posterior interpolator to use in the generate star_forming_gal to model sfr-mstar-redshift  
            # it works for all space-dists as below z<0.2 there is no well defined posterior and we use the one at z=0.25 which is within the redshift range of the real data                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
            if completeness_handling=='sharma-like' and (zgal[i] <= 0.2) : posterior_interp = utls.interpolate_prob(prob_arr=p_prob_arr, z_arr=p_z_arr, logm_arr=p_logm_arr, logsfr_arr=p_logsfr_arr, ztarget=0.25, plot=False)
            elif space_dist=='delta' : posterior_interp = utls.interpolate_prob(prob_arr=p_prob_arr, z_arr=p_z_arr, logm_arr=p_logm_arr, logsfr_arr=p_logsfr_arr, ztarget=zgal[i], plot=False)

            for n in tqdm(range(nlines), desc=f"Sampling posterior for {zbins[i]} $<$ z $\leq$ {zbins[i+1]}") : 
                if not sfr_sampling :
                    if space_dist == 'delta' :
                        post_sample = Gen_mock_gal(Nsample=Nsample, z_arr=zgal[i], plot_cdf=False, weight=weight, mfunc_ref=mfunc_ref, mfunc_slope=mfunc_slope, mfunc_mstar0=mfunc_mstar0, \
                                                        sfr_ref=sfr_ref, mode=mode, sfr_sampling=sfr_sampling, completeness_handling=completeness_handling, \
                                                             sigma_norm=sigma_norm, posterior=True, posterior_interp=posterior_interp, \
                                                                lgsfr_grid=p_logsfr_arr, space_dist=space_dist, z_min=z_min, z_max=z_max)[0]
                    
                    else :
                        post_sample = []; post_z_values = []
                        Nsubsample = int(Nsample/nz_bins)

                        for zij in zij_bins[i] :
                            posterior_interp = utls.interpolate_prob(prob_arr=p_prob_arr, z_arr=p_z_arr, logm_arr=p_logm_arr, logsfr_arr=p_logsfr_arr, ztarget=zij, plot=False)
                            post_subsample = Gen_mock_gal(Nsample=Nsubsample, z_arr=zgal[i], plot_cdf=False, weight=weight, mfunc_ref=mfunc_ref, mfunc_slope=mfunc_slope, mfunc_mstar0=mfunc_mstar0, \
                                                        sfr_ref=sfr_ref, mode=mode, sfr_sampling=sfr_sampling, completeness_handling=completeness_handling, \
                                                             sigma_norm=sigma_norm, posterior=True, posterior_interp=posterior_interp, \
                                                                lgsfr_grid=p_logsfr_arr, space_dist='delta', z_min=z_min, z_max=z_max)[0]
  
                            post_sample.append(post_subsample)
                            post_z_values.append([zij]*Nsubsample)
                        # merge subsumples to generate one realization for given redshift range
                        post_sample = np.array(post_sample).flatten()
                        post_z_values = np.array(post_z_values).flatten()
                    
                    log_sfr, log_sfr_mode = [], []
                
                    if (ml_sampling=='advanced') : ml_sampling='prescribed'

                    Kr_correction=False   # if log_sfr is empty then k-correction can not be applied since the algorithm cannot sample a color, which is a variable of the K-correction formula  
                
                else : # sample sfr and color etc.
                    if space_dist == 'delta' :
                        post_sample, log_sfr, log_sfr_mode = Gen_mock_gal(Nsample=Nsample, z_arr=zgal[i], plot_cdf=False, weight=weight, mfunc_ref=mfunc_ref, mfunc_slope=mfunc_slope, mfunc_mstar0=mfunc_mstar0, \
                                                        sfr_ref=sfr_ref, mode=mode, sfr_sampling=sfr_sampling, completeness_handling=completeness_handling, \
                                                             sigma_norm=sigma_norm, posterior=True, posterior_interp=posterior_interp, lgsfr_grid=p_logsfr_arr, \
                                                                space_dist=space_dist, z_min=z_min, z_max=z_max)
                    else : 
                        post_sample = []; post_z_values = []; log_sfr = []; log_sfr_mode = []
                        Nsubsample = int(Nsample/nz_bins)

                        for zij in zij_bins[i] :
                            posterior_interp = utls.interpolate_prob(prob_arr=p_prob_arr, z_arr=p_z_arr, logm_arr=p_logm_arr, logsfr_arr=p_logsfr_arr, ztarget=zij, plot=False)
                            post_subsample, sublog_sfr, sublog_sfr_mode = Gen_mock_gal(Nsample=Nsubsample, z_arr=zij, plot_cdf=False, weight=weight, mfunc_ref=mfunc_ref, mfunc_slope=mfunc_slope, mfunc_mstar0=mfunc_mstar0, \
                                                        sfr_ref=sfr_ref, mode=mode, sfr_sampling=sfr_sampling, completeness_handling=completeness_handling, \
                                                             sigma_norm=sigma_norm, posterior=True, posterior_interp=posterior_interp, lgsfr_grid=p_logsfr_arr, \
                                                                space_dist='delta', z_min=z_min, z_max=z_max)
                            
                            post_sample.append(post_subsample)
                            log_sfr.append(sublog_sfr)
                            log_sfr_mode.append(sublog_sfr_mode)
                            post_z_values.append([zij]*Nsubsample)
                        # merge subsumples to generate one realization for given redshift range
                        post_sample = np.array(post_sample).flatten()
                        log_sfr = np.array(log_sfr).flatten()
                        lof_sfr_mode = np.array(log_sfr_mode).flatten()
                        post_z_values = np.array(post_z_values).flatten()

                logm_samples.append(post_sample)
                if space_dist == 'delta' : post_z_values = None
                try :    
                    color_sample, Kcorr_sample, ml_sample, mlg_rest_sample, ind_magcut = magnitude_cut(log_mstar=post_sample, log_sfr=log_sfr, log_sfr_mode=log_sfr_mode, z=zgal[i], z_values=post_z_values, rmag_cut=23.5, plot=False, space_dist=space_dist, ml_sampling=ml_sampling, density_sfr_color=density_sfr_color, sfr_grid=sfr_grid, color_gr_grid=color_gr_grid, Kr_correction=Kr_correction)
                    MtoL_samples.append(ml_sample)
                    MtoLg_rest_samples.append(mlg_rest_sample)
                    color_samples.append(color_sample)
                    Kcorr_samples.append(Kcorr_sample)
                except :
                    ml_sample, ind_magcut = magnitude_cut(log_mstar=post_sample, log_sfr=log_sfr, log_sfr_mode=log_sfr_mode, z=zgal[i], z_values = post_z_values, rmag_cut=23.5, plot=False, space_dist=space_dist, ml_sampling=ml_sampling, density_sfr_color=density_sfr_color, sfr_grid=sfr_grid, color_gr_grid=color_gr_grid, Kr_correction=Kr_correction)
                    MtoL_samples.append(ml_sample)
                
                try :
                    lgsfr_samples.append(log_sfr)
                    lgsfr_mode_samples.append(log_sfr_mode)
                except : pass
                
                magcut_ind_flags.append(ind_magcut)
                try :
                    redshift_samples.append(post_z_values)
                except :
                    pass
                plt.hist(post_sample, cumulative=True, density=True, histtype='step', ls='--', color='red', bins=massbins, lw=1., alpha=alpha, zorder=0)
                plt.hist(post_sample[ind_magcut], cumulative=True, density=True, histtype='step', ls='-', color='red', bins=massbins, lw=1., alpha=alpha, zorder=0)

                try :
                    ks_pvalues[i, n] = utls.run_kstest(post_sample[ind_magcut], np.log10(frbdata_mstar[idx]))
                except :
                    pass
        
        del ind_magcut

    #plt.xlim(1e-2,1e2)
    plt.legend(fontsize=10)
    plt.tight_layout()
    if store_output : plt.savefig(folder_path+'prediction_vs_frb_cdf_mass.png', format='png')
    plt.show()



    # post processing data and plotting


    if store_output: 
        # Open a new HDF5 file and write data to it
        with h5py.File(folder_path + 'dataset.h5', 'w') as hf:
            # Store the data
            hf.create_dataset('logm_samples', data=logm_samples)
            hf.create_dataset('MtoL_samples', data=MtoL_samples)
            hf.create_dataset('magcut_ind_flags', data=magcut_ind_flags)
            hf.create_dataset('lgsfr_samples', data=lgsfr_samples)
            hf.create_dataset('lgsfr_mode_samples', data=lgsfr_mode_samples)
            hf.create_dataset('MtoLg_rest_samples', data=MtoLg_rest_samples)
            hf.create_dataset('color_samples', data=color_samples)
            hf.create_dataset('Kcorr_samples', data=Kcorr_samples)
            hf.create_dataset('density_sfr_color', data=density_sfr_color)
            hf.create_dataset('sfr_grid', data=sfr_grid)
            hf.create_dataset('color_gr_grid', data=color_gr_grid)
            if space_dist != 'delta' : hf.create_dataset('redshift_samples', data=redshift_samples)
            # Store input params (metadata) as attributes
            hf.attrs['zbins'] = zbins
            hf.attrs['zgal'] = zgal
            hf.attrs['nz_bins'] = nz_bins
            hf.attrs['Nsample'] = Nsample 
            hf.attrs['weight'] = weight 
            hf.attrs['mfunc_ref'] = mfunc_ref 
            hf.attrs['mfunc_slope'] = mfunc_slope
            hf.attrs['mfunc_mstar0'] = mfunc_mstar0
            hf.attrs['sfr_ref'] = sfr_ref
            hf.attrs['mode'] = mode
            hf.attrs['posterior'] = posterior 
            hf.attrs['completeness_handling'] = completeness_handling
            hf.attrs['sigma_norm'] = sigma_norm
            hf.attrs['n_realizations'] = n_realizations
            hf.attrs['data_source'] = data_source 
            hf.attrs['sfr_sampling'] = sfr_sampling
            hf.attrs['space_dist'] = space_dist
            if z_min is not None : hf.attrs['z_min'] = z_min
            if z_max is not None : hf.attrs['z_max'] = z_max
            hf.attrs['ml_sampling'] = ml_sampling
            #hf.attrs['prescribed_ml_func'] = prescribed_ml_func
            hf.attrs['Kr_correction'] = Kr_correction
            # Store the description (phrase) as an attribute
            hf.attrs['description'] = f"Output was created on {date.today()}."
            for key in p_dens_params.keys() :
                hf.attrs[f'{key}'] = p_dens_params[key]

    # plot p-values distribution
    if ks_test :
        # plot ks_values for each redshift bin
        plt.subplots(1,len(zgal), sharey=True, figsize=(11,4), gridspec_kw={'wspace': 0, 'hspace': 0})
        plt.subplot(101+10*len(zgal))
        plt.ylabel('probability')
        for i in range(len(zgal)) :
            plt.subplot(131+i)
            plt.hist(np.log10(np.clip(ks_pvalues[i], 1e-12, None)), bins=20, histtype='step', lw=2, density=True, ls='-', color='black')
            plt.xlabel('$\mathrm{\log(p-value)}$')
            plt.title('$\mathrm{P(p-value ~ | ~}$' + f'${zbins[i]}~$'+'$\mathrm{<z\leq}$' + f'$~{zbins[i+1]:.1f}$)')
        plt.tight_layout()
        if store_output : plt.savefig(folder_path+'ks_distribution.png', format='png')
        plt.show()


    # plot stellar mass samples 
    plt.subplots(1,3, sharey=True, figsize=(11,4), gridspec_kw={'wspace': 0, 'hspace': 0}) 
    plt.subplot(131)
    plt.ylabel("$\mathrm{P(M_\star,~z) ~ [arb.]}$")
    clrs = ['black', 'red', 'green']
    for counter in range(len(zgal)) :
        plt.subplot(131+counter)
        plt.hist(np.array(logm_samples[counter*n_realizations:(counter+1)*n_realizations]).ravel(), bins=50, lw=2, density=True, ls='-', histtype='step', color=clrs[counter], label=f'z={zgal[counter]:.2f}')
        plt.xlabel("$\mathrm{\log M_{\star} ~ [M_\odot]}$")
        plt.legend()
        plt.title(f"Stellar masses for z={zgal[counter]:.2f}")
        plt.yscale('log'); plt.xlim(6.51,13)
    
    plt.tight_layout()
    if store_output : plt.savefig(folder_path+'logm_distribution.png', format='png')
    plt.show()


    # plot stellar mass vs SFR samples
    try :
        if len(log_sfr) != 0 :
            plt.subplots(1,3, sharey=True, figsize=(11,4), gridspec_kw={'wspace': 0, 'hspace': 0}) 
            plt.subplot(131)
            plt.ylabel("$\mathrm{\log{SFR_{SED}} ~ [M_\odot ~ yr^{-1}]}$")
            #idx_sorted = np.argsort(post_sample)

            #post_sample = post_sample[idx_sorted]
            #log_sfr_
            panel = 0; clrs = ['black', 'red', 'green']
            for counter in range(len(logm_samples)) :
                plt.subplot(131+panel)
                plt.scatter(logm_samples[counter], lgsfr_samples[counter], marker='.', color=clrs[panel], s=1, alpha=0.1)#'SFR sample')
                plt.scatter(logm_samples[counter], lgsfr_mode_samples[counter], marker='.', color='blue', s=1, alpha=0.1)
            
                if (counter+1)%n_realizations == 0 :
                    plt.xlabel("$\mathrm{\log M_{\star} ~ [M_\odot]}$")
                    plt.legend(['SFR samples', 'SFR ridge-line'], markerscale=6., framealpha=1)
                    plt.title(f"SFR samples for z={zgal[panel]:.2f}")
                    panel+=1
                
            plt.tight_layout()
            if store_output : plt.savefig(folder_path+'sfr_mass_distribution.png', format='png')
            plt.show()
    except : pass

    # plot redshift distribution
    try :
        if space_dist in ['uniform-z', 'uniform-vol'] :
            plt.subplots(1,3, sharey=True, figsize=(11,4), gridspec_kw={'wspace': 0, 'hspace': 0}) 
            plt.subplot(131)
            plt.ylabel("density")

            clrs = ['black', 'red', 'green']
            for counter in range(len(zgal)) :
                plt.subplot(131+counter)
                plt.hist(np.array(redshift_samples[counter*n_realizations:(counter+1)*n_realizations]).ravel(), bins=20, lw=2, density=True, ls='-', histtype='step', color=clrs[counter])
                plt.xlabel("redshift"); plt.xlim(zbins[counter], zbins[counter+1])
                plt.title(f"Galaxy sample in ${zbins[counter]:.1f}<z\leq{zbins[counter+1]:.1f}$")
                
            plt.tight_layout()
            if store_output : plt.savefig(folder_path+'redshift_distribution.png', format='png')
            plt.show()
    except : pass

    # plot comoving distance distribution
    try :
        if space_dist in ['uniform-z', 'uniform-vol'] :
            plt.subplots(1,3, sharey=True, figsize=(11,4), gridspec_kw={'wspace': 0, 'hspace': 0}) 
            plt.subplot(131)
            plt.ylabel("$\mathrm{p(\chi)}$")

            clrs = ['black', 'red', 'green']
            for counter in range(len(zgal)) :
                plt.subplot(131+counter)
                #print('a')
                #print(cosmo.comoving_distance(np.array(redshift_samples[counter*n_realizations:(counter+1)*n_realizations]).ravel()).value / 1e3)
                x = np.linspace(cosmo.comoving_distance(zbins[counter]).value, cosmo.comoving_distance(zbins[counter+1]).value, 20) / 1e3
                plt.plot(x, 3*(x*x)/(x[-1]**3)/(1-(x[0]/x[-1])**3), ls='--', marker='', color='k', label="$p(\chi) ~\propto ~ \chi^2$")
                plt.hist(cosmo.comoving_distance(np.array(redshift_samples[counter*n_realizations:(counter+1)*n_realizations]).ravel()).value / 1e3, bins=20, lw=2, density=True, ls='-', histtype='step', color=clrs[counter])
                plt.xlabel("$\mathrm{\chi~~ [Gpc]}$"); plt.xlim(cosmo.comoving_distance(zbins[counter]).value / 1e3, cosmo.comoving_distance(zbins[counter+1]).value / 1e3)
                plt.title(f"Galaxy sample in ${zbins[counter]:.1f}<z\leq{zbins[counter+1]:.1f}$")
                
            plt.tight_layout()
            if store_output : plt.savefig(folder_path+'comoving_distance_distribution.png', format='png')
            plt.show()
    except : pass


    # plot mass-to-light ratio, Kr-correction, and color realizations
    if plot_M_L and MtoL_samples is not None : #(assume that len(zgal) = 3)
        plt.subplots(3,2, figsize=(8,12))#, sharey='row', gridspec_kw={'wspace': 0})#, 'hspace': 1})
        line_color='black'; tag_i = None #f'z = {zgal[0]}'
        
        for counter in range(len(MtoL_samples)) :
            plt.subplot(326)
            #plt.hist(MtoL_samples[counter], histtype='step', bins=80, color=line_color, alpha=0.1, cumulative=False, density=True, label=tag_i)
            try:
                plt.subplot(321)
            #    plt.hist(lgsfr_samples[counter], histtype='step', bins=40, color=line_color, alpha=0.1, cumulative=False, density=True, label=tag_i)
            except :
                pass
            try :
                plt.subplot(322)
            #    plt.hist(color_samples[counter], histtype='step', bins=40, color=line_color, alpha=0.1, cumulative=False, density=True, label=tag_i)
                plt.subplot(325)
            #    plt.hist(Kcorr_samples[counter], histtype='step', bins=40, color=line_color, alpha=0.1, cumulative=False, density=True, label=tag_i)
                plt.subplot(324)
            #    plt.hist(MtoL_samples[counter] + (Kcorr_samples[counter]/2.5), histtype='step', bins=80, color=line_color, alpha=0.1, cumulative=False, density=True, label=tag_i)
                plt.subplot(323)
            #    plt.hist(MtoLg_rest_samples[counter], histtype='step', bins=80, color=line_color, alpha=0.1, cumulative=False, density=True, label=tag_i)
            except: 
                pass
            counter += 1
            tag_i = None
            if counter > (n_realizations - 1) and counter < (n_realizations + 1) : 
                tag_i = f'{zbins[0]} $<$ z $\leq$ {zbins[1]}'
                plt.subplot(326)
                plt.hist(np.array(MtoL_samples[:(n_realizations)]).ravel(), histtype='step', bins=80, color=line_color, lw=2., cumulative=False, density=True, label=tag_i)
                try:
                    plt.subplot(321)
                    plt.hist(np.array(lgsfr_samples[:n_realizations]).ravel(), histtype='step', bins=40, color=line_color, lw=2., cumulative=False, density=True, label=tag_i)
                except:
                    pass
                try :
                    plt.subplot(322)
                    plt.hist(np.array(color_samples[:n_realizations]).ravel(), histtype='step', bins=40, color=line_color, lw=2., cumulative=False, density=True, label=tag_i)
                    plt.subplot(325)
                    plt.hist(np.array(Kcorr_samples[:n_realizations]).ravel(), histtype='step', bins=40, color=line_color, lw=2., cumulative=False, density=True, label=tag_i)
                    plt.subplot(324)
                    plt.hist(np.array(MtoL_samples[:n_realizations]).ravel() + (np.array(Kcorr_samples[:n_realizations]).ravel()/2.5), histtype='step', bins=80, color=line_color, lw=2., cumulative=False, density=True, label=tag_i)
                    plt.subplot(323)
                    plt.hist(np.array(MtoLg_rest_samples[:n_realizations]).ravel(), histtype='step', bins=80, color=line_color, lw=2., cumulative=False, density=True, label=tag_i)
                except: 
                    pass
                
                line_color = 'red'; tag_i = None

            if counter > (2*n_realizations - 1) and counter < (2*n_realizations + 1) :
                tag_i = f'{zbins[1]} $<$ z $\leq$ {zbins[2]}'#f'z = {zgal[1]}' 
                plt.subplot(326)
                plt.hist(np.array(MtoL_samples[n_realizations:2*n_realizations]).ravel(), histtype='step', bins=80, color=line_color, lw=2., cumulative=False, density=True, label=tag_i)
                try :
                    plt.subplot(321)
                    plt.hist(np.array(lgsfr_samples[n_realizations:2*n_realizations]).ravel(), histtype='step', bins=40, color=line_color, lw=2., cumulative=False, density=True, label=tag_i)
                except :
                    pass
                try :
                    plt.subplot(322)
                    plt.hist(np.array(color_samples[n_realizations:2*n_realizations]).ravel(), histtype='step', bins=40, color=line_color, lw=2., cumulative=False, density=True, label=tag_i)
                    plt.subplot(325)
                    plt.hist(np.array(Kcorr_samples[n_realizations:2*n_realizations]).ravel(), histtype='step', bins=40, color=line_color, lw=2., cumulative=False, density=True, label=tag_i)
                    plt.subplot(324)
                    plt.hist(np.array(MtoL_samples[n_realizations:2*n_realizations]).ravel() + (np.array(Kcorr_samples[n_realizations:2*n_realizations]).ravel()/2.5), histtype='step', bins=80, color=line_color, lw=2., cumulative=False, density=True, label=tag_i)
                    plt.subplot(323)
                    plt.hist(np.array(MtoLg_rest_samples[n_realizations:2*n_realizations]).ravel(), histtype='step', bins=80, color=line_color, lw=2., cumulative=False, density=True, label=tag_i)
                except: 
                    pass
                line_color = 'green'; tag_i = None
        

        tag_i = f'{zbins[2]} $<$ z $\leq$ {zbins[3]}'#f'z = {zgal[2]}' 
        plt.subplot(326)
        plt.hist(np.array(MtoL_samples[2*n_realizations:]).ravel(), histtype='step', bins=80, color=line_color, lw=2., cumulative=False, density=True, label=tag_i)
        try :
            plt.subplot(321)
            plt.hist(np.array(lgsfr_samples[2*n_realizations:]).ravel(), histtype='step', bins=40, color=line_color, lw=2., cumulative=False, density=True, label=tag_i)
        except:
            pass
        try :
            plt.subplot(322)
            plt.hist(np.array(color_samples[2*n_realizations:]).ravel(), histtype='step', bins=40, color=line_color, lw=2., cumulative=False, density=True, label=tag_i)
            plt.subplot(325)
            plt.hist(np.array(Kcorr_samples[2*n_realizations:]).ravel(), histtype='step', bins=40, color=line_color, lw=2., cumulative=False, density=True, label=tag_i)
            plt.subplot(324)
            plt.hist(np.array(MtoL_samples[2*n_realizations:]).ravel() + (np.array(Kcorr_samples[2*n_realizations:]).ravel()/2.5), histtype='step', bins=80, color=line_color, lw=2., cumulative=False, density=True, label=tag_i)
            plt.subplot(323)
            plt.hist(np.array(MtoLg_rest_samples[2*n_realizations:]).ravel(), histtype='step', bins=80, color=line_color, lw=2., cumulative=False, density=True, label=tag_i)
        except: 
            pass
        tag_i = None
        
        plt.subplot(326)
        plt.legend(fontsize=10)
        plt.ylabel('density')
        plt.xlabel("$\mathrm{\log(M_\star/L_{r})_{obs} ~~[M_\odot / L_\odot]}$")
        plt.title("observed $\mathrm{M_\star/L_r}$ realizations")
        plt.subplot(324)
        plt.legend(fontsize=10)
        plt.ylabel('density')
        plt.xlabel("$\mathrm{\log(M_\star/L_{r})_{rest} ~~[M_\odot / L_\odot]}$")
        plt.title("rest-frame $\mathrm{M_\star/L_r}$ realizations")
        plt.subplot(323)
        plt.legend(fontsize=10)
        plt.ylabel('density')
        plt.xlabel("$\mathrm{\log(M_\star/L_{g})_{rest} ~~[M_\odot / L_\odot]}$")
        plt.title("rest-frame $\mathrm{M_\star/L_g}$ realizations")
        plt.subplot(321)
        plt.legend(fontsize=10)
        plt.ylabel('density')
        plt.xlabel("$\mathrm{\log(SFR) ~~[M_\odot ~yr^{-1}]}$")
        plt.title("$\mathrm{SFR}$ realizations")
        plt.subplot(325)
        plt.legend(fontsize=10)
        plt.ylabel('density')
        plt.xlabel("$\mathrm{K_r}$")
        plt.text(x=0.25, y=0.7,s="$\mathrm{K_r = m_r^{rest} - m_r^{obs}}$", transform=plt.gca().transAxes, ha='center', va='center')
        plt.title("$\mathrm{K_r}$-correction realizations")
        plt.subplot(322)
        plt.legend(fontsize=10)
        plt.xlabel("$\mathrm{(g-r)_{rest}}$")
        plt.title("rest-frame color g-r realizations")
        plt.ylabel('density')
        plt.tight_layout()
        if store_output : plt.savefig(folder_path+'multi_panel_fig.png', format='png')
        plt.show()


    return 






