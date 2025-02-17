# standard regular-use python packages
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# standard astropy packages
from astropy.cosmology import Planck13, z_at_value
import astropy.units as u
from astropy.cosmology import WMAP9 as cosmo
import astropy
from astropy.io import ascii
from astropy import coordinates
from astropy.table import Table, unique
from astropy.coordinates import SkyCoord, Galactic, Galactocentric

import os
rltv_pth = os.path.dirname(os.path.abspath(__file__)) + '/../src/galfrb/data/'

def get_ra_dec_errs(frb_name):
    """
    Function for extracting the localizations for DSA-110 FRBs from discovery sheets.

    Parameters:
    ----------
    frb_name  : str
        Name of the FRB to retrieve localization errors for.

    Returns:
    -------
    position0 : SkyCoord
        SkyCoord object representing the voltage localization.
    e_ra      : float
        Error in right ascension (RA).
    e_dec     : float
        Error in declination (Dec).
    """
    # Read the FRBs data sheet
    dsa_radio_sheet = pd.read_excel(rltv_pth + 'frbs_data/DSA110-FRBs.xlsx', 
                                    sheet_name='frb_params', engine='openpyxl', 
                                    keep_default_na=False, na_values='')
    
    # Drop the first row (assuming it contains headers or irrelevant data)
    dsa_radio_sheet = dsa_radio_sheet.drop(0)
    
    # Filter the data by FRB name
    df = dsa_radio_sheet[dsa_radio_sheet["Nickname"]==frb_name].reset_index()
    
    # Convert DataFrame to astropy Table
    tbl = Table.from_pandas(df)
    tbl.rename_column('Nickname', 'Name')
    
    # Extract relevant columns
    nicknames = tbl['Name'].tolist()
    position0 = [coordinates.SkyCoord(tbl[i]['Voltage localization'].replace('\u200b', '', 2))
                 for i in range(len(tbl))]
    
    # Ensure consistency between nicknames and positions
    assert len(nicknames) == len(position0)
    
    # Extract position errors
    epos = tbl['Position error (1sigma)']
    for row in epos:
        if isinstance(row, str):
            er, ed = row.split(',')
            e_ra = float(er)
            if ed == ' 1.4 (nominal)':
                e_dec = 1.4
            else:
                e_dec = float(ed)
        else:
            e_ra = 1
            e_dec = 0.7
    
    return position0, e_ra, e_dec



def get_DMexgal(frb_name):
    """
    Get the extragalactic DM value for a given FRB name.
    
    Parameters:
    frb_name : str
        The nickname of the Fast Radio Burst (FRB).
    
    Returns:
    DMexgal  : float
        The extragalactic Dispersion Measure (DM_exgal) of the FRB.
    """
    # Read the FRBs data sheet
    dsa_radio_sheet = pd.read_excel(rltv_pth + 'frbs_data/DSA110-FRBs.xlsx', 
                                    sheet_name='frb_params', 
                                    engine='openpyxl', 
                                    keep_default_na=False, 
                                    na_values='')

    # Drop the first row (assuming it contains headers or irrelevant data)
    dsa_radio_sheet = dsa_radio_sheet.drop(0)

    # Filter the data by FRB name
    df = dsa_radio_sheet[dsa_radio_sheet["Nickname"] == frb_name].reset_index()
    DMexgal = df.DM_exgal[0]
    return DMexgal



def read_dsa_data(only_gold=True):
    """
    Reads DSA FRBs data published in Sharma+2024, 
    filters for 'sed_done?' == 'Y' and 'sample' == 'Gold', 
    and returns the filtered DataFrame.

    Parameters:
    ----------
    only_gold : bool, optional
        If True, filters for FRBs labeled as 'Gold' in addition to 
        'sed_done?' == 'Y' (default is True).

    Returns:
    -------
    dsa_frb s : DataFrame
        Filtered DataFrame containing DSA FRBs data.

    """
    # Read CSV file
    dsa_frbs = pd.read_csv(rltv_pth + "frbs_data/dsa_frbs.csv")
    
    if only_gold:
        dsa_frbs = dsa_frbs[np.logical_and(dsa_frbs["sed_done?"] == "Y", dsa_frbs["sample"] == "Gold")]
    else:
        dsa_frbs = dsa_frbs[dsa_frbs["sed_done?"] == "Y"]
    
    # Compute logarithmic errors for ssfr
    dsa_frbs["ssfr_errl"] = np.log10(dsa_frbs["ssfr"] + dsa_frbs["ssfr_errl"]) - np.log10(dsa_frbs["ssfr"])
    dsa_frbs["ssfr_erru"] = np.log10(dsa_frbs["ssfr"] + dsa_frbs["ssfr_erru"]) - np.log10(dsa_frbs["ssfr"])
    
    # Compute logarithm of ssfr
    dsa_frbs["ssfr"] = np.log10(dsa_frbs["ssfr"])
    
    return dsa_frbs



def read_chime_data():
    """
    Reads CHIME FRBs data from Bhardwaj+2023 and returns the DataFrame.

    Returns:
    -------
    chime_frbs : DataFrame
        DataFrame containing CHIME FRBs data.
    """
    chime_frbs = pd.read_csv(rltv_pth + "frbs_data/bhardwaj_et_al.csv")
    return chime_frbs



def fix_gordon_et_al_data_format(df):
    """
    Fixes the format of Gordon+2023 data by extracting numerical values
    from LaTeX-formatted strings and assigning them to appropriate 
    columns.
    """
    logMstar, logMstar_errl, logMstar_erru = [], [], []
    sfr_100Myr, sfr_100Myr_errl, sfr_100Myr_erru = [], [], []
    tm, tm_errl, tm_erru = [], [], []
    logzsol, logzsol_errl, logzsol_erru = [], [], []
    Av, Av_errl, Av_erru = [], [], []
    
    for i in range(len(df)):
        logMstar.append(float(str(df["log(M _*/M _sun)"][i])
                              .split("$")[1].split("_")[0].split("{")[1]
                              .split("}")[0]))
        logMstar_errl.append(float(str(df["log(M _*/M _sun)"][i])
                                  .split("$")[1].split("_")[1].split("{")[1]
                                  .split("}")[0]))
        logMstar_erru.append(float(str(df["log(M _*/M _sun)"][i])
                                  .split("$")[1].split("_")[1].split("{")[2]
                                  .split("}")[0]))
        sfr_100Myr.append(float(str(df["SFR_0-100 Myr"][i])
                                .split("$")[1].split("_")[0].split("{")[1]
                                .split("}")[0]))
        sfr_100Myr_errl.append(float(str(df["SFR_0-100 Myr"][i])
                                    .split("$")[1].split("_")[1].split("{")[1]
                                    .split("}")[0]))
        sfr_100Myr_erru.append(float(str(df["SFR_0-100 Myr"][i])
                                    .split("$")[1].split("_")[1].split("{")[2]
                                    .split("}")[0]))
        tm.append(float(str(df["t _m"][i])
                        .split("$")[1].split("_")[0].split("{")[1].split("}")[0]))
        tm_errl.append(float(str(df["t _m"][i])
                            .split("$")[1].split("_")[1].split("{")[1].split("}")[0]))
        tm_erru.append(float(str(df["t _m"][i])
                            .split("$")[1].split("_")[1].split("{")[2].split("}")[0])) 
        logzsol.append(float([df["log(Z _*/Z _sun)"][i][1] if df["log(Z _*/Z _sun)"][i][1]=="-" else "+"][0]
                             +str(df["log(Z _*/Z _sun)"][i])
                             .split("$")[1].split("_")[0].split("{")[1].split("}")[0]))
        logzsol_errl.append(float(str(df["log(Z _*/Z _sun)"][i])
                                 .split("$")[1].split("_")[1].split("{")[1]
                                 .split("}")[0]))
        logzsol_erru.append(float(str(df["log(Z _*/Z _sun)"][i])
                                 .split("$")[1].split("_")[1].split("{")[2]
                                 .split("}")[0])) 
        Av.append(float(str(df["A _V,old"][i])
                        .split("$")[1].split("_")[0].split("{")[1].split("}")[0]))
        Av_errl.append(float(str(df["A _V,old"][i])
                            .split("$")[1].split("_")[1].split("{")[1].split("}")[0]))
        Av_erru.append(float(str(df["A _V,old"][i])
                            .split("$")[1].split("_")[1].split("{")[2].split("}")[0])) 
    
    df["logMstar"] = logMstar
    df["logMstar_errl"] = logMstar_errl
    df["logMstar_erru"] = logMstar_erru
    df["sfr_100Myr"] = sfr_100Myr
    df["sfr_100Myr_errl"] = sfr_100Myr_errl
    df["sfr_100Myr_erru"] = sfr_100Myr_erru
    df["t_m"] = tm
    df["t_m_errl"] = tm_errl
    df["t_m_erru"] = tm_erru
    df["Av_old"] = Av
    df["Av_old_errl"] = Av_errl
    df["Av_old_erru"] = Av_erru
    df["logzsol"] = logzsol
    df["logzsol_errl"] = logzsol_errl
    df["logzsol_erru"] = logzsol_erru
    df["ssfr"] = np.log10((sfr_100Myr/(10**np.array(logMstar)))*(1e9)) # per Gyr
    df["ssfr_errl"] = np.log10(((np.array(sfr_100Myr)+np.array(sfr_100Myr_errl))/(10**(np.array(logMstar)+np.array(logMstar_erru))))*(1e9)) - np.array(df["ssfr"]) # per Gyr
    df["ssfr_erru"] = np.log10(((np.array(sfr_100Myr)+np.array(sfr_100Myr_erru))/(10**(np.array(logMstar)+np.array(logMstar_errl))))*(1e9)) - np.array(df["ssfr"]) # per Gyr

    return df



def read_askap_frbs():
    """
    Read and return Gordon+2023 FRB host galaxies data.

    Returns:
    -------
    askap_frbs : DataFrame
        A pandas DataFrame containing the ASKAP FRBs data with additional 
        rmag and rmag_err columns, excluding FRB 20121102A.
    """
    # Read the ASKAP FRBs data from a text file
    askap_frbs = pd.read_csv(rltv_pth + "frbs_data/gordon_et_al.txt", sep="\t")
    
    # Fix the data format using a custom function
    askap_frbs = fix_gordon_et_al_data_format(askap_frbs)

    # List of r-band magnitudes
    rmags = [23.73, 21.21, 16.17, 20.33, 21.68, 21.87, 22.16, 
             17.41, 22.15, 23.54, 20.34, 18.36, 21.05, 19.95, 
             17.86, 22.97, 19.47, 20.65, 17.17, 14.96, 19.64, 
             16.44, 21.19]
    
    # List of r-band magnitude errors
    rmag_errs = [0.14, 0.06, 0.03, 0.01, 0.05, 0.10, 0.06, 
                 0.002, 0.15, 0.15, 0.03, 0.003, 0.02, 0.01,
                 0.03, 0.04, 0.02, 0.03, 0.01, 0.01, 0.03, 
                 0.01, 0.08]
    
    # Add the rmag and rmag_err columns to the DataFrame
    askap_frbs["rmag"] = rmags
    askap_frbs["rmag_err"] = rmag_errs

    # Exclude FRB 20121102A and reset the index
    askap_frbs = askap_frbs[askap_frbs["FRB"] != "20121102A"].reset_index()

    return askap_frbs
    


def read_askap_nr_frbs():
    """
    Read and return Gordon+2023 non-repeating FRB host galaxies data.

    Returns:
    -------
    askap_nr_frbs : DataFrame
        A pandas DataFrame containing the filtered non-repeating FRBs data, 
        excluding FRB 20121102A.
    """
    # Read ASKAP FRBs data
    askap_frbs = read_askap_frbs()
    
    # Read ASKAP repeating FRBs data
    askap_r_frbs = read_askap_r_frbs()
    
    # Identify non-repeating FRBs
    repeaters = list(set(askap_frbs.FRB) - set(askap_r_frbs.FRB))
    rep_mask = []
    
    for i in range(len(askap_frbs)):
        if askap_frbs.FRB[i] in repeaters:
            rep_mask.append(True)
        else:
            rep_mask.append(False)
    
    # Filter the non-repeating FRBs
    askap_frbs = askap_frbs[np.array(rep_mask)]
    
    # Exclude FRB 20121102A and reset the index
    askap_frbs = askap_frbs[askap_frbs["FRB"] != "20121102A"].reset_index()
    
    # Drop the index column
    askap_frbs = askap_frbs.drop(labels=['index'], axis=1)
    
    return askap_frbs



def read_askap_r_frbs():
    """
    Read and return Gordon+2023 repeating FRB host galaxies data.

    Returns:
    -------
    askap_frbs_rep : DataFrame
        A pandas DataFrame containing the repeating FRBs data, 
        excluding FRB 20121102A.
    """
    # Read the ASKAP repeating FRBs data from a text file
    askap_frbs_rep = pd.read_csv(rltv_pth + "frbs_data/gordon_et_al_repeaters.txt", 
                                 sep="\t")
    
    # Fix the data format using a custom function
    askap_frbs_rep = fix_gordon_et_al_data_format(askap_frbs_rep)
    
    # Exclude FRB 20121102A and reset the index
    askap_frbs_rep = askap_frbs_rep[askap_frbs_rep["FRB"] != "20121102A"] \
                                .reset_index()
    
    # Drop the index column
    askap_frbs_rep = askap_frbs_rep.drop(labels=['index'], axis=1)
    
    return askap_frbs_rep



def read_frbs_hosts_data():
    """
    Read and process FRBs hosts data from multiple sources.
    
    Returns:
    -------
    frb_df : DataFrame
        A pandas DataFrame containing combined FRB hosts data 
        including logM, logSFR, and redshift values along with 
        their errors.
    """
    # Read data from various sources
    dsa_frbs   = read_dsa_data()
    askap_frbs = read_askap_frbs()
    chime_frbs = read_chime_data()

    # Combine logMstar data from different sources
    frb_logM = (list(dsa_frbs.logMstar) + 
                list(askap_frbs.logMstar) + 
                list(chime_frbs.logMstar))
    
    # Combine logMstar lower error data from different sources
    frb_logM_errl = (list(dsa_frbs.logMstar_errl) + 
                     list(askap_frbs.logMstar_errl) + 
                     list(chime_frbs.logMstar_errl))
    
    # Combine logMstar upper error data from different sources
    frb_logM_erru = (list(dsa_frbs.logMstar_erru) + 
                     list(askap_frbs.logMstar_erru) + 
                     list(chime_frbs.logMstar_erru))

    # Combine and calculate logSFR data from different sources
    frb_logSFR = np.log10(list(dsa_frbs.sfr_100Myr) + 
                          list(askap_frbs.sfr_100Myr) + 
                          list(chime_frbs.sfr_100Myr))
    
    # Combine and calculate logSFR lower error data from different sources
    frb_logSFR_errl = np.abs(
        np.log10(list(np.array(dsa_frbs.sfr_100Myr) + 
                      np.array(dsa_frbs.sfr_100Myr_errl)) +
                 list(np.array(askap_frbs.sfr_100Myr) + 
                      np.array(askap_frbs.sfr_100Myr_errl)) +
                 list(np.array(chime_frbs.sfr_100Myr) + 
                      np.array(chime_frbs.sfr_100Myr_errl))) - frb_logSFR)
    
    # Combine and calculate logSFR upper error data from different sources
    frb_logSFR_erru = (np.log10(list(dsa_frbs.sfr_100Myr + 
                                     dsa_frbs.sfr_100Myr_erru) +
                                list(askap_frbs.sfr_100Myr + 
                                     askap_frbs.sfr_100Myr_erru) +
                                list(chime_frbs.sfr_100Myr + 
                                     chime_frbs.sfr_100Myr_erru)) - frb_logSFR)
    
    # Combine redshift data from different sources
    frb_z = (list(dsa_frbs.redshift) + 
             list(askap_frbs.z) + 
             list(chime_frbs.z))

    # Create a DataFrame with the combined data
    frb_df = pd.DataFrame({
        "logM":        frb_logM,
        "logM_errl":   np.abs(frb_logM_errl),
        "logM_erru":   frb_logM_erru,
        "logSFR":      frb_logSFR,
        "logSFR_errl": frb_logSFR_errl,
        "logSFR_erru": frb_logSFR_erru,
        "z":           frb_z
    })
    
    return frb_df



def read_frbs_offsets_literature_data():
    """
    Read and process FRBs offsets data from Manning+2021 and Woodland+2023.
    
    Returns:
    -------
    frb_offset : list
        List of FRB offsets.
    frb_err : list
        List of FRB offset errors.
    """
    # Read Mannings et al. data from text files
    mannings_data1 = pd.read_csv(rltv_pth + "frbs_data/mannings_et_al.txt", sep="\t")
    mannings_data2 = pd.read_csv(rltv_pth + "frbs_data/mannings_et_al_redshifts.txt", 
                                 sep="\t")
    
    # Concatenate the Mannings data along columns
    mannings_data = pd.concat([mannings_data1, mannings_data2], 
                              axis=1, join="inner")

    # Extract Mannings offset and error values
    mannings_offset = [float(mannings_data['delta R'][i].split(" +or- ")[0]) 
                       for i in range(len(mannings_data))]
    mannings_err = [float(mannings_data['delta R'][i].split(" +or- ")[1]) 
                    for i in range(len(mannings_data))]

    # Woodland offset and error values
    woodland_offset = [11.91, 2.07, 2.84]
    woodland_err    = [0.95, 0.38, 1.0]

    # Combine offsets and errors from both sources
    frb_offset = mannings_offset + woodland_offset
    frb_err    = mannings_err + woodland_err
    
    return frb_offset, frb_err


def read_frbs_normalized_offsets_literature_data():
    """
    Read and process normalized FRB offsets data from Manning+2021 and 
    Woodland+2023.
    
    Returns:
    -------
    frb_offset_norm : list
        List of normalized FRB offsets.
    frb_err_norm : list
        List of normalized FRB offset errors.
    """
    # Read Mannings et al. data from text files
    mannings_data1 = pd.read_csv(rltv_pth + "frbs_data/mannings_et_al.txt", sep="\t")
    mannings_data2 = pd.read_csv(rltv_pth + "frbs_data/mannings_et_al_redshifts.txt", 
                                 sep="\t")
    
    # Concatenate the Mannings data along columns
    mannings_data = pd.concat([mannings_data1, mannings_data2], 
                              axis=1, join="inner")

    # Extract Mannings normalized offset and error values
    mannings_offset_norm = [
        float(mannings_data['delta R/r_e'][i].split(" +or- ")[0]) 
        for i in range(len(mannings_data))
    ]
    mannings_err_norm = [
        float(mannings_data['delta R/r_e'][i].split(" +or- ")[1]) 
        for i in range(len(mannings_data))
    ]

    # Woodland normalized offset and error values
    woodland_offset_norm = [4.32, 0.74, 1.20]
    woodland_err_norm    = [0.34, 0.12, 0.42]

    # Combine normalized offsets and errors from both sources
    frb_offset_norm = mannings_offset_norm + woodland_offset_norm
    frb_err_norm    = mannings_err_norm + woodland_err_norm

    return frb_offset_norm, frb_err_norm


def read_TypeIaSNe_hosts_data():
    """
    Read and process data related to Type Ia supernova host galaxies from 
    Lampeitl+2010 and Chilress+2013.

    Returns:
    -------
    TypeIaSN_df : DataFrame
        DataFrame containing logSFR, logSFR errors (lower and upper), logM, 
        logM errors (lower and upper), and redshift (z) information.
    """
    TypeIaSN_IAU = []
    TypeIaSN_logM, TypeIaSN_logM_erru, TypeIaSN_logM_errl = [], [], []
    TypeIaSN_logSFR, TypeIaSN_logSFR_erru, TypeIaSN_logSFR_errl = [], [], []

    line_num = -1
    with open('other_transients_data/lampeitl_et_al.txt') as f:
        for line in f.readlines():
            line_num += 1
            if line_num>38:
                try:
                    (line[6:12], float(line[55:60]), float(line[39:44]))
                    TypeIaSN_IAU.append((line[6:12]))
                    TypeIaSN_logSFR.append(float(line[55:60]))
                    TypeIaSN_logSFR_erru.append(float(line[61:65]))
                    TypeIaSN_logSFR_errl.append(float(line[66:70]))
                    TypeIaSN_logM.append(float(line[39:44]))
                    TypeIaSN_logM_erru.append(float(line[45:49]))
                    TypeIaSN_logM_errl.append(float(line[50:54]))
                except:
                    continue

    df1 = pd.DataFrame({"IAU": TypeIaSN_IAU, 
                        "logsfr": TypeIaSN_logSFR, 
                        "logsfr_errl": TypeIaSN_logSFR_errl, 
                        "logsfr_erru": TypeIaSN_logSFR_erru, 
                        "logmass": TypeIaSN_logM, 
                        "logmass_errl": TypeIaSN_logM_errl, 
                        "logmass_erru": TypeIaSN_logM_erru})

    TypeIaSN_IAU_ = []
    TypeIaSN_z = []

    TypeIaSN_table=ascii.read('other_transients_data/sdssIIsnsurvey.dat', 
                              delimiter=r'&')
    for i in range(len(TypeIaSN_table)):
        try:
            TypeIaSN_z.append(float(TypeIaSN_table[i][0].split()[5]))
            TypeIaSN_IAU_.append(TypeIaSN_table[i][0].split()[2])
        except:
            continue

    df2 = pd.DataFrame({"IAU": TypeIaSN_IAU_, "z": TypeIaSN_z})
    df = pd.merge(df1, df2, on="IAU", how="inner") 
    # joining dataframes to get redshift information from SDSS survey


    TypeIaSN_IAU = []
    TypeIaSN_logM, TypeIaSN_logM_erru, TypeIaSN_logM_errl = [], [], []
    TypeIaSN_logSFR, TypeIaSN_logSFR_erru, TypeIaSN_logSFR_errl = [], [], []

    line_num = -1
    with open('other_transients_data/childress_et_al.txt') as f:
        for line in f.readlines():
            line_num += 1
            if line_num>28:
                try:
                    TypeIaSN_IAU.append((line[0:15]))
                    TypeIaSN_logSFR.append(
                        float(line[33:39])+float(line[16:21]))
                    TypeIaSN_logSFR_erru.append(
                        float(line[40:44])+float(line[22:26]))
                    TypeIaSN_logSFR_errl.append(
                        float(line[45:49])+float(line[27:31]))
                    TypeIaSN_logM.append(float(line[16:21]))
                    TypeIaSN_logM_erru.append(float(line[22:26]))
                    TypeIaSN_logM_errl.append(float(line[27:31]))
                
                except:
                    continue
                    
    df1 = pd.DataFrame({"IAU": TypeIaSN_IAU, 
                        "logsfr": TypeIaSN_logSFR, 
                        "logsfr_errl": TypeIaSN_logSFR_errl, 
                        "logsfr_erru": TypeIaSN_logSFR_erru, 
                        "logmass": TypeIaSN_logM, 
                        "logmass_errl": TypeIaSN_logM_errl, 
                        "logmass_erru": TypeIaSN_logM_erru})

    TypeIaSN_IAU_ = []
    TypeIaSN_z = []
    line_num = -1

    with open('other_transients_data/childress_et_al_redshifts.txt') as f:
        for line in f.readlines():
            line_num += 1
            if line_num>45:
                try:
                    _ = (line[0:15], float(line[40:47]))
                    TypeIaSN_IAU_.append((line[0:15]))
                    TypeIaSN_z.append(float(line[40:47]))
                except:
                    continue

    df2 = pd.DataFrame({"IAU": TypeIaSN_IAU_, "z": TypeIaSN_z})
    df_ = pd.merge(df1, df2, on="IAU", how="inner")


    TypeIaSN_logM = np.array(list(df["logmass"])+list(df_["logmass"]))
    TypeIaSN_logSFR = np.array(list(df["logsfr"])+list(df_["logsfr"]))
    TypeIaSN_logM_errl = np.array(list(df["logmass_errl"])+\
                                  list(df_["logmass_errl"]))
    TypeIaSN_logSFR_errl = np.array(list(df["logsfr_errl"])+\
                                    list(df_["logsfr_errl"]))
    TypeIaSN_logM_erru = np.array(list(df["logmass_erru"])+\
                                  list(df_["logmass_erru"]))
    TypeIaSN_logSFR_erru = np.array(list(df["logsfr_erru"])+\
                                    list(df_["logsfr_erru"]))
    TypeIaSN_z = np.array(list(df["z"])+list(df_["z"]))

    TypeIaSN_df = pd.DataFrame({"logSFR": TypeIaSN_logSFR, 
                                "logSFR_errl": TypeIaSN_logSFR_errl, 
                                "logSFR_erru": TypeIaSN_logSFR_erru, 
                                "logM": TypeIaSN_logM, 
                                "logM_errl": TypeIaSN_logM_errl,
                                "logM_erru": TypeIaSN_logM_erru, 
                                "z": TypeIaSN_z})

    return TypeIaSN_df



def read_TypeIaSNe_offsets():
    """
    Read and process Type Ia Supernovae offsets data from Uddin+2020.

    Returns:
    -------
    typeIa_offset : list
        List of Type Ia Supernovae projected distances.
    typeIa_err    : list
        List of Type Ia Supernovae projected distance errors.
    """
    # Read Type Ia Supernovae data from a text file
    typeIa_data = pd.read_csv("other_transients_data/uddin_et_al.txt", 
                              sep="\t")
    
    # Extract Type Ia Supernovae projected distances
    typeIa_offset = typeIa_data["Projected distance"]
    
    # Calculate error in projected distances (assuming 10% error)
    typeIa_err = typeIa_offset * 0.1

    return list(typeIa_offset), list(typeIa_err)



def read_TypeIaSNe_normalized_offsets():
    """
    Read Type Ia supernova data from Uddin+2020, calculate normalized offsets,
    and return the host_norm_offset and its error.
    
    Returns:
    -------
    host_norm_offset : float
        Host-normalized offsets.
    error_host_norm_offset : float
        Errors in host-normalized offsets.
    """
    typeIa_data = pd.read_csv("other_transients_data/uddin_et_al.txt", sep="\t")
    
    logA = 0.86
    alpha = 0.25
    typeIa_data["half_light_radii"] = 10**(logA + alpha * \
                                    (typeIa_data["M_best"] - np.log10(5e10)))  # kpc
    host_norm_offset = typeIa_data["Projected distance"] / \
          typeIa_data["half_light_radii"]

    # Calculating error in host_norm_offset (assuming 10% error)
    error_host_norm_offset = 0.1 * np.array(host_norm_offset)

    return host_norm_offset, error_host_norm_offset



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



def read_ULXsources_offsets():
    """
    Read and process ULX sources offsets data from Kovlakas+2020.
    
    Returns:
    -------
    ULX_offset : list
        List of ULX sources offsets in kpc.
    ULX_err : list
        List of ULX sources offset errors in kpc.
    """
    # Read ULX hosts and sources data from FITS files
    ULX_data1 = Table.read('other_transients_data/kovlakas_et_al_hosts.fits')
    ULX_data1["pgc"] = ULX_data1["PGC"]
    ULX_data2 = Table.read('other_transients_data/kovlakas_et_al_sources.fits')
    
    # Join the data on 'pgc' key
    ULX_data = astropy.table.join(ULX_data1, ULX_data2, keys="pgc", 
                                  join_type='inner')

    # Filter the data based on reliability, luminosity, and other criteria
    ULX_data = ULX_data[~ULX_data["unreliable"]]
    ULX_data = ULX_data[ULX_data["LX"] > 1e39]
    ULX_data = ULX_data[~ULX_data["nuclear"]]
    ULX_data = ULX_data[ULX_data["logM"] > 1]
    ULX_data = ULX_data[ULX_data["logSFR"] > -5]
    
    # Calculate redshift based on distance
    ULX_z = z_at_value(Planck13.luminosity_distance, ULX_data["D"])
    
    ULX_offset = []
    for i in range(len(ULX_data)):
        pos1 = SkyCoord(ULX_data["RA"][i], ULX_data["DEC"][i], 
                        unit=(u.deg, u.deg), frame='icrs')
        pos2 = SkyCoord(ULX_data["source_ra"][i], ULX_data["source_dec"][i], 
                        unit=(u.deg, u.deg), frame='icrs')
        ang_dist = cosmo.angular_diameter_distance(ULX_z[i])
        ULX_offset.append(
            (pos1.separation(pos2).arcsec / 3600) * (3.14 / 180) * 
            ang_dist.value * 1e3)

    # Convert offset list to a numpy array
    ULX_offset = np.array(ULX_offset)
    ULX_err = ULX_offset * 0.1
    
    return ULX_offset, ULX_err



def read_CCSNe_hosts_data(SNclass="SN II"):
    """
    Read Core-Collapse Supernovae (CCSNe) host data from Schulze+2021, filter 
    based on SN class, calculate necessary values, and return a DataFrame with 
    relevant columns.

    Parameters:
    ----------
    SNclass : str, optional
        Type of CCSNe to filter for (default is "SN II").

    Returns:
    -------
    CCSN_df : DataFrame
        DataFrame with columns 'logM', 'logM_errl', 'logM_erru',
        'logSFR', 'logSFR_errl', 'logSFR_erru', 'SN_type', 'z',
        'filter', 'rmag', 'ID'.
    """
    file_path = 'other_transients_data/schulze_et_al_host_mags.txt'
    column_names = ['ID', 'Tel/Sur', 'Inst', 'Filter', 'mag', 'e_mag', 'Ref']
    df1 = pd.read_fwf(file_path, skiprows=30, names=column_names)
    selected_columns = ['ID', 'Filter', 'mag', 'e_mag']
    df1 = df1[selected_columns]

    file_path = 'other_transients_data/schulze_et_al.txt'
    column_names = ['ID', 'Type', 'z', 'chi2', 'nof', 'E(B-V)', 'E_E(B-V)', 
                    'e_E(B-V)', 'FUVMag', 'E_FUVMag', 'e_FUVMag', 'BMag', 
                    'E_BMag', 'e_BMag', 'KsMag', 'E_KsMag', 'e_KsMag',
                    'logSFR', 'E_logSFR', 'e_logSFR', 'logM', 'E_logM', 
                    'e_logM', 'logsSFR', 'E_logsSFR', 'e_logsSFR', 
                    'Age', 'E_Age', 'e_Age']
    df2 = pd.read_fwf(file_path, skiprows=58, names=column_names)

    CCSN_df = pd.merge(df1, df2, on='ID', how='inner')
    CCSN_df = CCSN_df[CCSN_df["Filter"] == "r"]

    CCSN_df = pd.DataFrame({"logM": CCSN_df.logM, "logM_errl": CCSN_df.e_logM, 
                            "logM_erru": CCSN_df.E_logM,
                            "logSFR": CCSN_df.logSFR, 
                            "logSFR_errl": CCSN_df.e_logsSFR, 
                            "logSFR_erru": CCSN_df.E_logsSFR,
                            "SN_type": CCSN_df.Type, "z": CCSN_df.z, 
                            "filter": CCSN_df.Filter, "rmag": CCSN_df.mag, 
                            'ID': CCSN_df.ID})
    if SNclass != "all":
        CCSN_df = CCSN_df[CCSN_df["SN_type"] == SNclass]
    CCSN_df = CCSN_df.drop_duplicates(subset=['ID']).reset_index()
    CCSN_df = CCSN_df.drop(labels=['index'], axis=1)
    
    return CCSN_df



def read_CCSNe_offsets(SNclass="SNII"):
    """
    Read and process CCSNe offsets data from Schulze+2021.

    Parameters:
    ----------
    SNclass : str, optional
        The class of supernovae to filter by. Default is "SNII". 
        Use "all" to include all classes.

    Returns:
    -------
    ccsn_offset : array_like
        Array of CCSNe offsets.
    ccsn_err : array_like
        Array of CCSNe offset errors.
    """
    # File path to the Schulze et al. data
    file_path = "other_transients_data/schulze_et_al_offsets.txt"
    
    # Define the column names based on the header
    colnames = ['ID', 'Type', 'z', 'f_z', 'IAU', 'RAh', 'RAm', 'RAs', 'DE-', 
                'DEd', 'DEm', 'DEs', 'hRAh', 'hRAm', 'hRAs', 'hDE-', 'hDEd', 
                'hDEm', 'hDEs', 'AOffset', 'e_AOffset', 'POffset', 'e_POffset', 
                'E(B-V)']

    # Read the data into a pandas DataFrame
    CCSN_df = pd.read_fwf(file_path, skiprows=51, names=colnames)
    
    # Filter data based on the specified supernova class
    if SNclass != "all":
        CCSN_df = CCSN_df[CCSN_df["Type"] == SNclass]
    
    # Remove rows with non-positive offset errors
    CCSN_df = CCSN_df[CCSN_df["e_AOffset"] > 0].reset_index(drop=True)
    
    # Extract offsets and errors as numpy arrays
    ccsn_offset = np.array(CCSN_df.AOffset)
    ccsn_err = np.array(CCSN_df.e_AOffset)
    
    return ccsn_offset, ccsn_err



def read_CCSNe_normalized_offsets():
    """
    Read Core-Collapse Supernovae (CCSNe) normalized offsets from Kelly & 
    Kirshner (2012), filter for type 'II' CCSNe, and return a DataFrame with 
    relevant columns.

    Returns:
    -------
    ccsn_offset_norm : array_like
        Array of CCSNe offsets.
    ccsn_err_norm : array_like
        Array of CCSNe offset errors.
    """
    file_path = "other_transients_data/kelly_and_kirshner.txt"
    col_widths = [8, 8, 4, 4, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4]
    column_names = ["SN", "Type", "r_Type", "Offset", "LogM", "e_LogM", 
                    "E_LogM", "FOffSSFR", "SSFR", "e_SSFR", "E_SSFR", "FOffOH", 
                    "T04", "e_T04", "E_T04", "PP04", "e_PP04", "AV", "e_AV", 
                    "u-z"]
    data = pd.read_fwf(file_path, widths=col_widths, 
                       skiprows=41, names=column_names)
    selected_columns = ['Type', 'Offset']
    df = data[selected_columns]
    mask = []
    for i in range(len(df)):
        if "II" in df["Type"][i]:
            mask.append(True)
        else:
            mask.append(False)
    df = df[mask].reset_index()
    ccsn_offset_norm, ccsn_err_norm = df["Offset"], 0.1 * df["Offset"]
    return ccsn_offset_norm, ccsn_err_norm



def read_SLSNe_hosts_data():
    """
    Read data for Superluminous Supernovae (SLSNe) hosts from Schulze+2021 and
    Taggart & Perley 2021, merge them into a DataFrame, and return it.

    Returns:
    -------
    SLSN_df : DataFrame
        DataFrame containing columns 'logM', 'logM_errl', 'logM_erru',
        'logSFR', 'logSFR_errl', 'logSFR_erru', 'z' for SLSNe hosts.
    """
    SLSN_logM, SLSN_logM_errl, SLSN_logM_erru = [], [], []
    SLSN_logSFR, SLSN_logSFR_errl, SLSN_logSFR_erru = [], [], []
    SLSN_z = []

    # Read SLSN-I data from CCSNe hosts data
    CCSN_df = read_CCSNe_hosts_data("SLSN-I")
    for i in range(len(CCSN_df)):
        SLSN_logM.append(CCSN_df["logM"][i])
        SLSN_logM_erru.append(CCSN_df["logM_erru"][i])
        SLSN_logM_errl.append(CCSN_df["logM_errl"][i])
        SLSN_logSFR.append(CCSN_df["logSFR"][i])
        SLSN_logSFR_errl.append(CCSN_df["logSFR_errl"][i])
        SLSN_logSFR_erru.append(CCSN_df["logSFR_erru"][i])
        SLSN_z.append(CCSN_df["z"][i])

    # Read SLSN-IIn data from CCSNe hosts data
    CCSN_df = read_CCSNe_hosts_data("SLSN-IIn")
    for i in range(len(CCSN_df)):
        SLSN_logM.append(CCSN_df["logM"][i])
        SLSN_logM_erru.append(CCSN_df["logM_erru"][i])
        SLSN_logM_errl.append(CCSN_df["logM_errl"][i])
        SLSN_logSFR.append(CCSN_df["logSFR"][i])
        SLSN_logSFR_errl.append(CCSN_df["logSFR_errl"][i])
        SLSN_logSFR_erru.append(CCSN_df["logSFR_erru"][i])
        SLSN_z.append(CCSN_df["z"][i])

    # Read additional SLSN host parameters
    slsn_table = ascii.read('other_transients_data/slsne_host_params.dat', delimiter=r'&')
    slsn_table["sfr_plus_er"] = np.log10(slsn_table["sfr"] + \
                                         slsn_table["sfr_plus_er"])
    slsn_table["sfr_minus_er"] = np.log10(slsn_table["sfr"] + \
                                          slsn_table["sfr_minus_er"])
    slsn_table["sfr"] = np.log10(slsn_table["sfr"])
    slsn_table["sfr_plus_er"] -= slsn_table["sfr"]
    slsn_table["sfr_minus_er"] -= slsn_table["sfr"]

    # Append additional SLSN host data to lists
    SLSN_logM = np.array(list(SLSN_logM) + list(slsn_table["mass"]))
    SLSN_logM_errl = np.array(list(SLSN_logM_errl) + \
                             list(slsn_table["mass_minus_er"]))
    SLSN_logM_erru = np.array(list(SLSN_logM_erru) + \
                              list(slsn_table["mass_plus_er"]))
    SLSN_logSFR = np.array(list(SLSN_logSFR) + list(slsn_table["sfr"]))
    SLSN_logSFR_errl = np.array(list(SLSN_logSFR_errl) + \
                                list(slsn_table["sfr_minus_er"]))
    SLSN_logSFR_erru = np.array(list(SLSN_logSFR_erru) + \
                                list(slsn_table["sfr_plus_er"]))
    SLSN_z = np.array(list(SLSN_z) + list(slsn_table["z"]))

    # Create and return DataFrame for SLSN hosts
    SLSN_df = pd.DataFrame({
        "logM": SLSN_logM,
        "logM_errl": SLSN_logM_errl,
        "logM_erru": SLSN_logM_erru,
        "logSFR": SLSN_logSFR,
        "logSFR_errl": SLSN_logSFR_errl,
        "logSFR_erru": SLSN_logSFR_erru,
        "z": SLSN_z
    })

    return SLSN_df



def read_SLSNe_offsets():
    """
    Read and process SLSNe offsets data from Lunnan+2015.
    
    Returns:
    -------
    slsn_offset : array_like
        Array of SLSNe projected offsets.
    slsn_err : array_like
        Array of SLSNe projected offset errors.
    """
    file_path = "other_transients_data/lunnan_et_al.txt"
    
    # Read the file into lines
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Define the columns explicitly
    columns = ["Object", "sigma_tie", "sigma_SN", "sigma_gal", "r_50", "r_80", 
               "log(Sigma_SFR)", "ProjectedOffset", "NormalizedOffset", 
               "LightFraction"]

    # Initialize an empty list to store rows
    rows = []

    # Process each line
    for line in lines[1:]:
        # Split the line by whitespace and handle multiple spaces
        parts = line.strip().split()
        
        # Ensure we have the correct number of columns by checking for 
        # extra values
        if len(parts) == 11:
            # Handle the case with an extra split causing 11 parts
            parts[3] = parts[3] + ' ' + parts[4]
            parts.pop(4)

        # Adjust for 'cdots' placeholders and '<0.4' value
        row = []
        for part in parts:
            if part == 'cdots':
                row.append(0)
            elif part.startswith('<'):
                row.append(part[1:])  # Strip the '<' sign
            else:
                row.append(part)
        
        # Append the row to the list of rows
        rows.append(row)

    # Convert the list of rows to a DataFrame
    df = pd.DataFrame(rows, columns=columns)

    # Convert appropriate columns to numeric, setting errors='coerce' to 
    # handle 'NaN'
    numeric_columns = columns[1:]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, 
                                                    errors='coerce')
    
    # Filter out zero offsets and assume 10% uncertainties
    slsn_offset = np.array(df.ProjectedOffset)[np.array(df.ProjectedOffset)!=0]
    slsn_err = 0.1 * slsn_offset  # assuming 10% uncertainties
    
    return slsn_offset, slsn_err



def read_SLSNe_normalized_offsets():
    """
    Read and process SLSNe offsets data from Lunnan+2015.
    
    Returns:
    -------
    slsn_offset : array_like
        Array of normalized SLSNe offsets.
    slsn_err    : array_like
        Array of normalized SLSNe offset errors.
    """
    file_path = "other_transients_data/lunnan_et_al.txt"
    
    # Read the file into lines
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Define the columns explicitly
    columns = ["Object", "sigma_tie", "sigma_SN", "sigma_gal", "r_50", "r_80", 
               "log(Sigma_SFR)", "ProjectedOffset", "NormalizedOffset", 
               "LightFraction"]

    # Initialize an empty list to store rows
    rows = []

    # Process each line
    for line in lines[1:]:
        # Split the line by whitespace and handle multiple spaces
        parts = line.strip().split()
        
        # Ensure we have the correct number of columns by checking 
        # for extra values
        if len(parts) == 11:
            # Handle the case with an extra split causing 11 parts
            parts[3] = parts[3] + ' ' + parts[4]
            parts.pop(4)

        # Adjust for 'cdots' placeholders and '<0.4' value
        row = []
        for part in parts:
            if part == 'cdots':
                row.append(0)
            elif part.startswith('<'):
                row.append(part[1:])  # Strip the '<' sign
            else:
                row.append(part)
        
        # Append the row to the list of rows
        rows.append(row)

    # Convert the list of rows to a DataFrame
    df = pd.DataFrame(rows, columns=columns)

    # Convert appropriate columns to numeric, setting errors='coerce' to
    # handle 'NaN'
    numeric_columns = columns[1:]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, 
                                                    errors='coerce')
    
    # Filter out zero normalized offsets and assume 10% uncertainties
    slsn_offset = np.array(df.NormalizedOffset)[
        np.array(df.NormalizedOffset) != 0]
    slsn_err = 0.1 * slsn_offset  # assuming 10% uncertainties
    
    return slsn_offset, slsn_err



def read_sGRB_hosts_data():
    """
    Read data for short Gamma-Ray Burst (sGRB) hosts from Nugent+2022,
    process the data, and return a DataFrame.

    Returns:
    -------
    sgrb_df : DataFrame
        DataFrame containing columns 'logM', 'logM_errl', 'logM_erru',
        'logSFR', 'logSFR_errl', 'logSFR_erru', 'z' for sGRB hosts.
    """
    sgrb_logM, sgrb_logM_errl, sgrb_logM_erru = [], [], [] 
    sgrb_logSFR, sgrb_logSFR_errl, sgrb_logSFR_erru = [], [], [] 
    sgrb_z = []

    line_num = -1
    with open('other_transients_data/nugent_et_al.txt') as f:
        for line in f.readlines():
            line_num += 1
            if line_num > 67:
                try:
                    if line[0:4] == "GOLD":
                        z = float(line[19:23])
                        mass = float(line[55:60])
                        sfr = float(line[72:77])
                        sfr_errl = float(line[87:92])
                        sfr_erru = float(line[79:85])
                        
                        if sfr == 0:
                            sfr = 1e-10
                            sfr_errl = 1e-10
                            sfr_erru = 1e-10
                        
                        sgrb_logM.append(mass)
                        sgrb_logM_errl.append(float(line[66:70]))
                        sgrb_logM_erru.append(float(line[61:65]))
                        sgrb_logSFR.append(np.log10(sfr))
                        sgrb_logSFR_errl.append(np.log10(sfr) - \
                                                np.log10(sfr + sfr_errl))
                        sgrb_logSFR_erru.append(-np.log10(sfr) + \
                                                np.log10(sfr + sfr_erru))
                        sgrb_z.append(z)
                    else:
                        continue
                except:
                    continue

    sgrb_df = pd.DataFrame({
        "logM": sgrb_logM,
        "logM_errl": sgrb_logM_errl,
        "logM_erru": sgrb_logM_erru,
        "logSFR": sgrb_logSFR,
        "logSFR_errl": np.abs(sgrb_logSFR_errl),
        "logSFR_erru": sgrb_logSFR_erru,
        "z": sgrb_z
    })
    
    return sgrb_df



def read_sGRB_offsets(zcut=10):
    """
    Read and process sGRB offsets data from Fong+2022.
    
    Parameters:
    ----------
    zcut : float, optional
        Redshift cutoff for filtering sGRBs. Default is 10.

    Returns:
    -------
    sgrb_offset : Series
        Series of sGRB offsets.
    sgrb_err : Series
        Series of sGRB offset errors.
    """
    # Read the sGRB data from Fong et al.
    sgrb_data = pd.read_csv("other_transients_data/fong_et_al.txt", sep="\t")
    
    # Filter data based on redshift cutoff and remove rows with 'cdots' 
    # placeholders
    sgrb_data = sgrb_data[
        np.logical_and(sgrb_data["z"] <= zcut, 
                       sgrb_data["Offset.1"] != "cdots")
    ].reset_index(drop=True)
    
    # Extract sGRB offsets and errors
    sgrb_offset = sgrb_data["Offset ^b"]
    sgrb_err = sgrb_data["sigma.1"]
    
    return sgrb_offset, sgrb_err



def read_sGRB_normalized_offsets(zcut=10):
    """
    Read normalized offsets for sGRBs from Fong+2022
    and filter based on redshift cutoff (zcut).

    Parameters:
    ----------
    zcut : float, optional
        Redshift cutoff value. Default is 10.

    Returns:
    -------
    sgrb_offset_ : Series
        Series of sGRB offsets.
    sgrb_err_ : Series
        Series of sGRB offset errors.
    """
    sgrb_data = pd.read_csv("other_transients_data/fong_et_al.txt", sep="\t")
    sgrb_data = sgrb_data[np.logical_and(sgrb_data["z"] <= zcut, 
                                         sgrb_data["Offset.1"] != "cdots")
                                         ].reset_index()
    sgrb_offset_ = [float(sgrb_data["Offset.1"][i]) for i in range(
        len(sgrb_data))]
    sgrb_err_ = [float(sgrb_data["sigma.2"][i]) for i in range(len(sgrb_data))]
    return sgrb_offset_, sgrb_err_



def read_lGRB_hosts_data():
    """
    Reads host galaxy data for lGRBs from Vergani+2015 and Taggart & Perley 
    2021, combines and processes the data, and returns a DataFrame.

    Returns:
    -------
    lgrb_df : DataFrame
        DataFrame containing columns for logM (stellar mass), 
        logM_errl (lower stellar mass error), logM_erru (upper stellar mass 
        error), logSFR (logarithm of star formation rate), logSFR_errl (lower 
        logSFR error), logSFR_erru (upper logSFR error), and z (redshift).
    """
    lGRB_data = pd.read_csv("other_transients_data/vergani_et_al.txt", sep="\t")
    lGRB_data["logM"] = lGRB_data["Log(M⋆)"]
    lGRB_data["logM_errl"] = lGRB_data["Log(M⋆)"] - lGRB_data["Log(M⋆ inf)"]
    lGRB_data["logM_erru"] = lGRB_data["Log(M⋆ sup)"] - lGRB_data["Log(M⋆)"]
    lGRB_data["logSFR"] = [float(lGRB_data["Log(SFR)"][i]) for i in range(
        len(lGRB_data))]
    lGRB_data["logSFR_errl"] = lGRB_data["logSFR"] - \
          np.array([float(lGRB_data["Log(SFRinf)"][i]) for \
                     i in range(len(lGRB_data))])
    lGRB_data["logSFR_erru"] = np.array([float(lGRB_data["Log(SFRsup)"][i]) \
                                          for i in range(len(lGRB_data))]) - \
                                            lGRB_data["logSFR"]

    lgrb_table = ascii.read(
        '../sed_fits_v1/results_compilation/literature_data/grb_host_params.dat', 
        delimiter=r'&')
    lgrb_table["sfr_plus_er"] = np.log10(lgrb_table["sfr"] + \
                                         lgrb_table["sfr_plus_er"])
    lgrb_table["sfr_minus_er"] = np.log10(lgrb_table["sfr"] + \
                                          lgrb_table["sfr_minus_er"])
    lgrb_table["sfr"] = np.log10(lgrb_table["sfr"])
    lgrb_table["sfr_plus_er"] -= lgrb_table["sfr"]
    lgrb_table["sfr_minus_er"] -= lgrb_table["sfr"]

    lgrb_logM = np.array(list(lgrb_table["mass"]) + list(lGRB_data["logM"]))
    lgrb_logM_errl = np.array(list(lgrb_table["sfr_minus_er"]) + \
                              list(lGRB_data["logM_errl"]))
    lgrb_logM_erru = np.array(list(lgrb_table["sfr_plus_er"]) + \
                              list(lGRB_data["logM_erru"]))
    lgrb_logSFR = np.array(list(lgrb_table["sfr"]) + \
                           list(lGRB_data["logSFR"]))
    lgrb_logSFR_errl = np.array(list(lgrb_table["sfr_minus_er"]) + \
                                list(lGRB_data["logSFR_errl"]))
    lgrb_logSFR_erru = np.array(list(lgrb_table["sfr_plus_er"]) + \
                                list(lGRB_data["logSFR_erru"]))
    lgrb_z = np.array(list(lgrb_table["z"]) + list(lGRB_data["z"]))

    lgrb_df = pd.DataFrame({"logM": lgrb_logM, 
                            "logM_errl": lgrb_logM_errl, 
                            "logM_erru": lgrb_logM_erru,
                            "logSFR": lgrb_logSFR, 
                            "logSFR_errl": lgrb_logSFR_errl, 
                            "logSFR_erru": lgrb_logSFR_erru,
                            "z": lgrb_z})

    return lgrb_df



def read_lGRB_offsets():
    """
    Read and process lGRB offsets data from Blanchard+2016.
    
    Returns:
    -------
    lgrb_offset : list
        List of lGRB physical offsets.
    lgrb_err : list
        List of lGRB physical offset errors.
    """
    # Read the lGRB data from Blanchard et al.
    lgrb_data = pd.read_csv("other_transients_data/blanchard_et_al.txt", 
                            sep="\t")
    
    # Extract lGRB physical offsets and errors
    lgrb_offset = [float(lgrb_data['Rphys'][i].split(" +or- ")[0]) \
                   for i in range(len(lgrb_data))]
    lgrb_err = [float(lgrb_data['Rphys'][i].split(" +or- ")[1]) \
                for i in range(len(lgrb_data))]
    
    return lgrb_offset, lgrb_err



def read_lGRB_normalized_offsets():
    """
    Read and process normalized lGRBs offsets data from Blanchard+2016.
    
    Returns:
    -------
    lgrb_offset_norm : list
        List of normalized lGRB offsets.
    lgrb_err_norm : list
        List of normalized lGRB offset errors.
    """
    # Read the lGRB data from Blanchard et al.
    lgrb_data = pd.read_csv("other_transients_data/blanchard_et_al.txt", 
                            sep="\t")
    
    # Extract normalized lGRB offsets and errors
    lgrb_offset_norm = [float(lgrb_data['Rnorm'][i].split(" +or- ")[0]) \
                        for i in range(len(lgrb_data))]
    lgrb_err_norm = [float(lgrb_data['Rnorm'][i].split(" +or- ")[1]) \
                     for i in range(len(lgrb_data))]
    
    return lgrb_offset_norm, lgrb_err_norm



def read_MW_sattelites_offsets():
    """
    Reads Milky Way satellite galaxy offsets from Drlica-Wagner+2020, 
    calculates galactocentric distances, and returns a list of 
    absolute distances.

    Returns:
    -------
    mw_satellites_offsets : list
        List of absolute galactocentric distances of Milky Way 
        satellite galaxies.
    """
    filename = "other_transients_data/MW_sattelites.txt"
    mw_satellites_offsets = []
    
    with open(filename, 'r') as file:
        for line in file:
            ra = float(line[29:37]) * u.deg
            dec = float(line[38:46]) * u.deg
            helio_offset = float(line[66:69]) * u.kpc
            
            # Create SkyCoord object for equatorial coordinates
            equatorial_coord = SkyCoord(ra=ra, dec=dec, distance=helio_offset, 
                                        frame='icrs')
            
            # Transform to Galactic coordinates
            galactic_coord = equatorial_coord.transform_to(Galactic)
            
            # Transform to Galactocentric coordinates
            galactocentric_coord = galactic_coord.transform_to(Galactocentric)
            
            # Extract Milky Way centric distance (absolute value)
            milky_way_centric_distance = galactocentric_coord.z
            mw_satellites_offsets.append(
                np.abs(milky_way_centric_distance.value))
    
    return mw_satellites_offsets



def read_MW_sattelites_normalized_offsets():
    """
    Reads Milky Way satellite galaxy offsets from Drlica-Wagner+2020, 
    calculates galactocentric distances, and returns normalized offsets.

    Returns:
    -------
    normalized_offsets : numpy.array
        Normalized offsets of Milky Way satellite galaxies.
    """
    mw_sattelites_offsets = read_MW_sattelites_offsets()
    mw_half_light_rad = 3.43  # Half-light radius in kpc (Lian et al. 2024)

    normalized_offsets = np.array(mw_sattelites_offsets) / mw_half_light_rad
    return normalized_offsets



def read_MW_GCs_offsets():
    """
    Reads Milky Way globular cluster galactocentric distances from Harris+1996.
    https://vizier.cds.unistra.fr/viz-bin/VizieR-3

    Returns:
    -------
    mw_gc_offsets : list
        List of galactocentric distances of Milky Way globular clusters (in kpc).
    """
    filename = "other_transients_data/mw_gcs.tsv"
    mw_gc = pd.read_csv(filename, delimiter="|")
    mw_gc_offsets = []
    
    for i in range(len(mw_gc)):
        try:
            mw_gc_offsets.append(float(mw_gc["Rgc"][i]))  # Append the galactocentric distance (Rgc) in kpc
        except:
            continue
    
    return mw_gc_offsets



def read_MW_GCs_normalized_offsets():
    """
    Reads Milky Way globular cluster galactocentric distances from Harris+1996, 
    normalizes them by specified half-light radius, and returns the normalized 
    offsets.

    Returns:
    -------
    normalized_offsets : list
        Normalized offsets of Milky Way globular clusters based on half-light 
        radius.
    """
    mw_gc_offsets = read_MW_GCs_offsets()  # Assuming read_MW_GCs_offsets() 
                            # function reads and returns MW GC offsets in kpc
    mw_half_light_rad = 3.43  # Half-light radius in kpc

    normalized_offsets = np.array(mw_gc_offsets) / mw_half_light_rad
    
    return normalized_offsets



def read_cosmos_3dhst_data(apply_cuts=False):
    """
    Read galaxies data from Prospector catalogs - Laigle et al. (2016), 
    Skelton et al. (2014), Leja et al. (2019), Leja et al. (2020), 
    Leja et al. (2022), Leja et al. (2020), Leja et al. (2022)
    
    Parameters:
    -----------
    apply_cuts : bool, optional
        If True, apply redshift cuts (z < 1). Default is False.

    Returns:
    --------
    cosmos_data : DataFrame
        Merged dataframe containing COSMOS galaxies data with columns:
        - "SFR": Star formation rate.
        - "Mass": Stellar mass.
        - "Z": Redshift.
    """
    # Read and convert COSMOS 1 data
    cosmos_data1 = ascii.read('galaxies_data/prospector_cosmos_catalog.dat')
    cosmos_data1 = cosmos_data1.to_pandas()
    
    # Apply redshift cut if specified
    if apply_cuts:
        cosmos_data1 = cosmos_data1[cosmos_data1["z"] < 1].reset_index(
            drop=True)
    
    # Rename and select relevant columns
    cosmos_data1["SFR"] = cosmos_data1["logsfr_median"]
    cosmos_data1["Mass"] = cosmos_data1["logm_median"]
    cosmos_data1["Z"] = cosmos_data1["z"]
    
    # Read and convert COSMOS 2 data
    cosmos_data2 = ascii.read('galaxies_data/prospector_3dhst_catalog.dat')
    cosmos_data2 = cosmos_data2.to_pandas()
    
    # Apply redshift cut if specified
    if apply_cuts:
        cosmos_data2 = cosmos_data2[cosmos_data2["z"] < 1].reset_index(
            drop=True)
    
    # Concatenate both datasets
    cosmos_data = pd.concat([cosmos_data1, cosmos_data2]).reset_index(
        drop=True)
    
    return cosmos_data



def read_cosmos_data(apply_cuts=False):
    """
    Read galaxies data from Prospector catalogs - Laigle et al. (2016), 
    Leja et al. (2020), Leja et al. (2022), Leja et al. (2020), 
    Leja et al. (2022)

    Parameters:
    -----------
    apply_cuts : bool, optional
        If True, applies additional cuts and merges with photometric data.
        Default is False.

    Returns:
    --------
    cosmos_data : DataFrame
        Processed COSMOS galaxy data with columns:
        - "logMstar": Stellar mass.
        - "logMstar_erru", "logMstar_errl": Errors in stellar mass.
        - "sfr_100Myr": Star formation rate over 100 Myr.
        - "sfr_100Myr_erru", "sfr_100Myr_errl": Errors in SFR over 100 Myr.
        - "logzsol": Metallicity.
        - "logzsol_erru", "logzsol_errl": Errors in metallicity.
        - "ssfr": Specific star formation rate.
        - "ssfr_erru", "ssfr_errl": Errors in specific star formation rate.
        - "Av_old": Old dust attenuation.
        - "Av_old_erru", "Av_old_errl": Errors in old dust attenuation.
        - "t_m": Average age.
        - "t_m_erru", "t_m_errl": Errors in average age.
        - "Z": Redshift.
        - "rmag": Photometric magnitude.

    Notes:
    The function assumes columns and data structures from the prospector_cosmos_catalog.dat file.
    """
    # Read COSMOS data from file
    cosmos_data = ascii.read('galaxies_data/prospector_cosmos_catalog.dat').to_pandas()
    
    # Apply redshift cut if specified
    if apply_cuts:
        cosmos_data = cosmos_data[cosmos_data["z"] < 1].reset_index(drop=True)
    
    # Rename columns for clarity
    cosmos_data = cosmos_data.rename(columns={
        "logm_median": "logMstar",
        "logm_errup": "logMstar_erru",
        "logm_errdown": "logMstar_errl",
        "logsfr_median": "sfr_100Myr",
        "logsfr_errup": "sfr_100Myr_erru",
        "logsfr_errdown": "sfr_100Myr_errl",
        "massmet_2_median": "logzsol",
        "massmet_2_errup": "logzsol_erru",
        "massmet_2_errdown": "logzsol_errl",
        "ssfr_100_median": "ssfr",
        "ssfr_100_errup": "ssfr_erru",
        "ssfr_100_errdown": "ssfr_errl",
        "dust2_median": "Av_old",
        "dust2_errdown": "Av_old_errl",
        "dust2_errup": "Av_old_erru",
        "avg_age_median": "t_m",
        "avg_age_errdown": "t_m_errl",
        "avg_age_errup": "t_m_erru"
    })
    
    # Perform data transformations
    cosmos_data["logMstar_erru"] -= cosmos_data["logMstar"]
    cosmos_data["logMstar_errl"] -= cosmos_data["logMstar"]
    cosmos_data["sfr_100Myr"] = 10**cosmos_data["sfr_100Myr"]
    cosmos_data["sfr_100Myr_erru"] = 10**cosmos_data["sfr_100Myr_erru"] - cosmos_data["sfr_100Myr"]
    cosmos_data["sfr_100Myr_errl"] = 10**cosmos_data["sfr_100Myr_errl"] - cosmos_data["sfr_100Myr"]
    cosmos_data["logzsol_erru"] -= cosmos_data["logzsol"]
    cosmos_data["logzsol_errl"] -= cosmos_data["logzsol"]
    cosmos_data["ssfr_erru"] -= cosmos_data["ssfr"]
    cosmos_data["ssfr_errl"] -= cosmos_data["ssfr"]
    cosmos_data["ssfr"] += 9  # Adding 9 to ssfr
    cosmos_data["Av_old_erru"] = 1.086 * (cosmos_data["Av_old_erru"] - cosmos_data["Av_old"])
    cosmos_data["Av_old_errl"] = 1.086 * (cosmos_data["Av_old_errl"] - cosmos_data["Av_old"])
    cosmos_data["Av_old"] = 1.086 * cosmos_data["Av_old"]
    cosmos_data["t_m_erru"] -= cosmos_data["t_m"]
    cosmos_data["t_m_errl"] -= cosmos_data["t_m"]
    
    # Merge with photometric data if apply_cuts is True
    if apply_cuts:
        with open("galaxies_data/cosmos_photometry_catalog.txt", 'r') as file:
            lines = file.readlines()

        IDs, rmags1, rmags2 = [], [], []

        for line in lines[4:]:
            parts = line.split()
            IDs.append(int(parts[0]))
            rmags1.append(float(parts[20]))
            rmags2.append(float(parts[34]))

        cosmos_phot_cat = pd.DataFrame({"objname": IDs, "rmag1": rmags1, "rmag2": rmags2})

        # Calculate average rmag
        rmags = []
        for rmag1, rmag2 in zip(rmags1, rmags2):
            if rmag1 == -99 and rmag2 != 99:
                rmags.append(rmag2)
            elif rmag1 != -99 and rmag2 == 99:
                rmags.append(rmag1)
            elif rmag1 != -99 and rmag2 != 99:
                rmags.append((rmag1 + rmag2) / 2)
            else:
                rmags.append(99)

        cosmos_phot_cat["rmag"] = rmags
        cosmos_data = pd.merge(cosmos_data, cosmos_phot_cat, on="objname", how='inner')

    return cosmos_data

