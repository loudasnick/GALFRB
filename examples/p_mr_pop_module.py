import numpy as np
import matplotlib.pyplot as plt
import h5py
import seaborn as sns


def load_p_mr_distributions(fname: str = 'p_mr_distributions_dz0.01_z_in_0_1.2.h5', output_dir: str = 'p_mr_dists/') -> tuple:
    """
    Load the p(mr|z) distributions from an HDF5 file.
    Args:
        fname (str): Input filename.
        output_dir (str): Directory where the file is stored.
    Returns:
        zbins (np.array): Redshift bin edges.
        rmag_centers (np.array): Centers of r-band magnitude bins.
        p_mr_sfr (np.array): p(mr|z) for SFR-weighted population. Shape: (len(zbins) - 1, rmag_resolution). rmag_resolution(=len(rmag_centers)) is fixed across redshift bins.
        p_mr_mass (np.array): p(mr|z) for Mass-weighted population. Shape: (len(zbins) - 1, rmag_resolution). rmag_resolution(=len(rmag_centers)) is fixed across redshift bins.
    Note:
        The PDF in m_r within a given redshift bin [z1,z2] has been computed at the right edge of the bin (z = z2).
    """

    with h5py.File(output_dir + fname, 'r') as hf:
        zbins = np.array(hf['zbins'])
        rmag_centers = np.array(hf['rmag_centers'])
        p_mr_sfr = np.array(hf['p_mr_sfr'])
        p_mr_mass = np.array(hf['p_mr_mass'])

    print(f"p(mr|z) distributions loaded successfully from 'p_mr_dists/{fname}'")
    n_redshift_bins = len(zbins) - 1

    def give_p_mr_mass(z: float):
        """
        Function to return p(mr|z) for mass-weighted population.
        Args:
            z (float): Redshift value.
        Returns:
            np.array: p(mr|z) values.
        Note:
            This function assumes that the redshift bins are defined in the `massweighted_population` data.
            Given the fine discretization of redshift bins, it uses the nearest bin for the provided redshift value.
            rmag_centers and p_mr_mass are defined in the outer scope of this function.
        """
        # Find the appropriate redshift bin index
        idx = np.clip(np.searchsorted(zbins, z) - 1, 0,  n_redshift_bins - 1)
        return p_mr_mass[idx]
    
    def give_p_mr_sfr(z: float):
        """
        Function to return p(mr|z) for SFR-weighted population.
        Args:
            z (float): Redshift value.
        Returns:
            np.array: p(mr|z) values.
        Note:
            This function assumes that the redshift bins are defined in the `sfrweighted_population` data.
            Given the fine discretization of redshift bins, it uses the nearest bin for the provided redshift value.
            rmag_centers and p_mr_sfr are defined in the outer scope of this function.
        """
        # Find the appropriate redshift bin index
        idx = np.clip(np.searchsorted(zbins, z) - 1, 0,  n_redshift_bins - 1)
        return p_mr_sfr[idx]

    return zbins, rmag_centers, p_mr_sfr, p_mr_mass, give_p_mr_sfr, give_p_mr_mass

zbins, rmag_centers, p_mr_sfr, p_mr_mass, give_p_mr_sfr, give_p_mr_mass = load_p_mr_distributions(fname = 'p_mr_distributions_dz0.01_z_in_0_1.2.h5', output_dir = 'p_mr_dists/')