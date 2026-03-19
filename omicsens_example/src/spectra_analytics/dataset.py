"""
Helper module for loading the spectra from .npy files

"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import lttb
from matplotlib import pyplot as plt
import math
import os
import pywt
from joblib import Parallel, delayed
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils import check_random_state
from sklearn.model_selection import cross_val_score


def acid_data_loader(acid_type, acids, unknown_acids, path=None):
    """Load aminoacids data."""
    if path is None:
        path = "./"
    _type, subtype = acid_type
    if _type == "gas" and subtype in ("neutral", "zwit"):
        frame = Data(path).load_gas_acids(subtype)
    elif _type == "water" and 0 <= int(subtype) <= 10:
        frame = Data(path).load_water_acids(subtype)
    else:
        raise ValueError(f"invalid acid type: {acid_type}")

    if unknown_acids is None:
        unknown_acids = []
    element_spectra = frame[acids]

    unknown_element_spectra = frame[unknown_acids]
    return element_spectra, unknown_element_spectra

def peptide_data_loader(peptide_type, peptides, unknown_peptides, path=None,data_source='simu'):
    """Load peptide data.
     peptide_type: tuple (solvent_type, condition, index)
        solvent_type: "H2O" or "D2O"
        condition: "water" or "no_water"
        index: "avg" or 0-10
    """
    if path is None:
        path = "./"
    solvent_type, condition, index = peptide_type
    data = Data(path)
    frame = data.load_peptides(condition=condition, index=index, solvent_type=solvent_type,data_source=data_source)

    if unknown_peptides is None:
        unknown_peptides = []

    element_spectra = frame[peptides]
    unknown_element_spectra = frame[unknown_peptides]
    return element_spectra, unknown_element_spectra


def plot_solution_comparison(solution_cnn, solution_bilstm, title="CNN vs BiLSTM spectra"):
    """
    Plot the generated spectrum vs CNN and BiLSTM regression solutions
    in the same style as your original plot_solution.
    """
    # X-axis: just the index of the spectrum
    x = np.arange(len(solution_cnn.spectrum))

    # Generated spectrum
    spectrum = solution_cnn.spectrum

    # Predicted spectra
    pred_cnn = solution_cnn.predict(solution_cnn.solve())
    pred_bilstm = solution_bilstm.predict(solution_bilstm.solve())

    plt.figure(figsize=(8, 6))
    plt.plot(x, spectrum, label="spectrum", linewidth=2,color="blue")
    plt.plot(x, pred_cnn, linestyle="--", label="CNN regression", linewidth=2,color='lime')
    plt.plot(x, pred_bilstm, linestyle="--", label="BiLSTM regression", linewidth=2, color='magenta')

    plt.xlabel("Frequency vector index", fontsize=16)
    plt.ylabel("Intensity", fontsize=16)
    plt.title(title, fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_sampling_distribution_comparison(solution_cnn, solution_bilstm, true_weights=None, num_samples=2000):
    peptides = solution_cnn.element_weights.index.tolist()

    # Get the sampling distributions
    frame_cnn = solution_cnn.sampling_distribution(num_samples)
    frame_bilstm = solution_bilstm.sampling_distribution(num_samples)

    n_peptides = len(peptides)
    fig, axes = plt.subplots(1, n_peptides, figsize=(4 * n_peptides, 4), sharey=True)
    if n_peptides == 1:
        axes = [axes]

    for i, peptide in enumerate(peptides):
        ax = axes[i]
        # CNN histogram
        frame_cnn[peptide].hist(bins=20, alpha=0.8, color="magenta", ax=ax, label="CNN")
        # BiLSTM histogram
        frame_bilstm[peptide].hist(bins=20, alpha=0.8, color="lime", ax=ax, label="BiLSTM")

        # True weight line
        if true_weights is not None:
            w_true = true_weights[peptide]
            ax.axvline(w_true, color="k", linestyle="--", label="True weight")

        ax.set_title(peptide)
        ax.set_xlabel("Weight")
        ax.set_ylabel("Count")
        ax.legend()

    plt.suptitle("Sampling distribution of predicted peptide weights (CNN vs BiLSTM)")
    plt.tight_layout()
    plt.show()


def load_multiple_peptide_configs(peptide_config_map,data_source='simu',downsampling = 'None',path='./'):
    combined = {}


    for peptide, configuration in peptide_config_map.items():
        spectra, _ = peptide_data_loader(
            peptide_type=configuration,
            peptides=[peptide],
            unknown_peptides=None,
            path = path,
            data_source=data_source
        )
        combined[peptide] = spectra[peptide]  # store the single column

    # Turn into DataFrame with each peptide as a column
    element_spectra = pd.DataFrame(combined)
    peptides = list(peptide_config_map.keys())

    if downsampling == 'H2O_simu_water':
        selected_wavelengths = [3095.4, 3167.9, 3201.1000000000004,
                                3214.2000000000003, 3226.6000000000004, 3262.4,
                                3275.6000000000004, 3282.2000000000003, 3296.2000000000003,
                                3317.6000000000004, 3330.4, 3356.6000000000004,
                                3365.9, 3378.2000000000003, 3397.6000000000004,
                                3405.6000000000004, 3415.4, 3415.5,
                                3420.2000000000003, 3426.2000000000003, 3434.2000000000003,
                                3447.4, 3460.7000000000003, 3463.6000000000004,
                                3468.1000000000004, 3491.2000000000003, 3492.1000000000004,
                                3505.9, 3531.2000000000003, 3537.6000000000004,
                                3537.7000000000003, 3538.1000000000004, 3552.7000000000003,
                                3582.2000000000003, 3612.1000000000004, 3667.1000000000004]
        element_spectra = element_spectra.loc[selected_wavelengths]

    elif downsampling == 'H2O_simu_no_water':
        selected_wavelengths = [1579.1000000000001, 1638.1000000000001, 1639.1000000000001, 1658.1000000000001,
                                1669.1000000000001, 1671.8000000000002, 1672.8000000000002, 1690.3000000000002,
                                1706.3000000000002, 1714.6000000000001, 1721.4, 1729.4, 2734.1000000000004,
                                2741.0,
                                2770.8, 2927.9, 2943.1000000000004, 2976.7000000000003, 3098.6000000000004,
                                3121.2000000000003,
                                3133.2000000000003, 3152.3, 3167.6000000000004, 3184.7000000000003,
                                3231.6000000000004,
                                3244.7000000000003, 3276.0, 3288.6000000000004, 3305.4, 3324.6000000000004,
                                3371.2000000000003, 3381.7000000000003, 3409.7000000000003, 3425.4,
                                3434.6000000000004, 3516.2000000000003]  # no water

        element_spectra = element_spectra.loc[selected_wavelengths]

    elif downsampling == "D2O_exp":
         selected_wavelengths =[ 450.0,  452.5,  453.0,  453.5,  454.0,  480.5,  666.5,  667.0,  667.5,
        668.0, 1455.5, 1456.0, 1670.0, 1670.5, 1671.0, 1978.0, 2322.5, 2323.0,
       2323.5, 2324.0, 2324.5, 2325.0, 2325.5, 2326.0, 2326.5, 2327.0, 2327.5,
       2328.0, 2328.5, 2329.0, 3372.0, 3376.0, 3383.0, 3383.5, 3384.0, 3384.5] #The selected wavelengths in D2O exp

         element_spectra = element_spectra.loc[selected_wavelengths]

    elif downsampling == 'H2O_exp':
        selected_wavelenghts = [ 451.0,  451.5,  452.0,  452.5,  453.0,  453.5,  454.0,  454.5,  459.0,
        465.0,  465.5,  466.0,  529.0, 1045.5, 1046.0, 1046.5, 1047.5, 1194.5,
        1195.5, 1196.0, 1196.5, 1197.5, 1198.0, 1198.5, 1199.0, 1199.5, 1200.0,
        1200.5, 1201.0, 1201.5, 1202.0, 1202.5, 1203.0, 1205.5, 1208.5, 2918.0] # The selected wavelengths in H2O exp

        element_spectra = element_spectra.loc[selected_wavelenghts]

    elif downsampling == 'D2O_simu_no_water':
        selected_wavelengths = [1462.1000000000001, 1514.1000000000001, 1639.6000000000001,
         1651.8000000000002, 1660.8000000000002, 1666.5,
         1670.1000000000001, 1670.3000000000002, 1670.8000000000002,
         1670.9, 1679.8000000000002, 1687.5,
         1690.9, 1697.8000000000002, 1704.9,
         1710.3000000000002, 1716.8000000000002, 1725.3000000000002,
         1740.6000000000001, 2016.4, 2181.7000000000003,
         2197.2000000000003, 2276.4, 2319.2000000000003,
         2346.5, 2367.2000000000003, 2377.2000000000003,
         2393.2000000000003, 2397.9, 2425.9,
         2436.7000000000003, 2445.7000000000003, 2484.4,
         2496.1000000000004, 2507.6000000000004, 2562.1000000000004]  # no water

        element_spectra = element_spectra.loc[selected_wavelengths]


    elif downsampling == "diff_D2O_exp":
        selected_wavelengths = [ 450.0,  452.5,  453.0,  453.5,  454.0,  480.5,  666.5,  667.0,  667.5,
        668.0, 1455.5, 1456.0, 1670.0, 1670.5, 1671.0, 1978.0, 2322.5, 2323.0,
       2323.5, 2324.0, 2324.5, 2325.0, 2325.5, 2326.0, 2326.5, 2327.0, 2327.5,
       2328.0, 2328.5, 2329.0, 3372.0, 3376.0, 3383.0, 3383.5, 3384.0, 3384.5]  # The selected wavelengths in D2O exp

        element_spectra = element_spectra.loc[selected_wavelengths]

    elif downsampling == 'diff_H2O_exp':
        selected_wavelenghts = [ 451.5,  452.0,  452.5,  453.0,  453.5,  454.0,  454.5,  457.5,  459.0,
        465.0,  465.5,  466.0,  484.0,  501.5, 1044.5, 1045.0, 1045.5, 1046.0,
       1047.0, 1090.0, 1191.5, 1196.5, 1197.0, 1198.0, 1199.0, 1200.0, 1200.5,
       1201.0, 1201.5, 1202.0, 1204.0, 1204.5, 1549.0, 2159.5, 2915.5, 2917.0]  # The selected wavelengths in H2O exp

        element_spectra = element_spectra.loc[selected_wavelenghts]

    elif downsampling == 'diff_D2O_simu_no_water':
        selected_wavelengths = [1509.1000000000001, 1635.6000000000001, 1639.3000000000002,
       1651.8000000000002, 1660.8000000000002, 1666.6000000000001,
                   1669.9,             1670.0, 1670.3000000000002,
                   1670.4, 1679.6000000000001, 1687.1000000000001,
       1690.6000000000001, 1697.8000000000002, 1704.8000000000002,
       1710.1000000000001, 1716.6000000000001, 1725.1000000000001,
       1736.1000000000001, 1740.6000000000001, 1767.8000000000002,
                   2016.4, 2029.1000000000001, 2181.7000000000003,
                   2211.0, 2231.7000000000003,             2276.4,
       2319.2000000000003, 2397.7000000000003,             2425.9,
                   2436.4, 2445.7000000000003,             2484.4,
                   2495.4,             2508.9, 2562.1000000000004]

        element_spectra = element_spectra.loc[selected_wavelengths]

    elif downsampling == 'diff_H2O_simu_no_water':
        selected_wavelengths = [1639.6000000000001, 1656.6000000000001, 1663.3000000000002,
                   1669.0, 1669.1000000000001, 1669.6000000000001,
       1684.8000000000002, 1690.8000000000002,             1690.9,
                   1699.5, 1706.1000000000001, 1715.8000000000002,
       1722.1000000000001,             1729.4, 2734.1000000000004,
                   2741.0,             2770.8,             2927.9,
       2943.7000000000003,             2976.9, 3007.1000000000004,
       3022.2000000000003, 3032.2000000000003, 3121.2000000000003,
                   3133.4, 3168.2000000000003, 3183.1000000000004,
                   3244.8,             3276.0,             3288.5,
                   3305.5,             3324.4,             3380.3,
       3409.6000000000004,             3425.4, 3516.1000000000004] # no water

        element_spectra = element_spectra.loc[selected_wavelengths]

    elif downsampling == 'range_H2O_exp':
        selected_wavelengths = [3171.0, 3177.5, 3178.5, 3202.0, 3206.0, 3210.5, 3228.0, 3228.5, 3229.5,
       3232.5, 3234.5, 3235.0, 3235.5, 3236.5, 3237.0, 3294.5, 3295.0, 3310.5,
       3372.5, 3472.5, 3500.5, 3501.0, 3501.5, 3502.0, 3509.0, 3509.5, 3510.0,
       3516.5, 3517.0, 3527.0, 3527.5, 3528.0, 3690.0, 3735.0, 3742.0, 3854.5]

        element_spectra = element_spectra.loc[selected_wavelengths]


    elif downsampling == 'range_D2O_exp':
        selected_wavelengths = [3006.0, 3006.5, 3332.5, 3335.0, 3342.5, 3354.5, 3363.5, 3372.5, 3373.0,
       3375.0, 3382.0, 3382.5, 3383.0, 3383.5, 3384.0, 3384.5, 3385.5, 3386.0,
       3389.0, 3389.5, 3390.0, 3395.0, 3399.5, 3400.0, 3400.5, 3401.0, 3402.5,
       3403.0, 3403.5, 3408.5, 3411.5, 3420.0, 3420.5, 3438.0, 3443.0, 3475.0]

        element_spectra = element_spectra.loc[selected_wavelengths]

    elif downsampling == 'range_H2O_simu':
        selected_wavelengths = [3171.0, 3177.5, 3178.5, 3202.0, 3206.0, 3210.5, 3228.0, 3228.5, 3229.5,
       3232.5, 3234.5, 3235.0, 3235.5, 3236.5, 3237.0, 3294.5, 3295.0, 3310.5,
       3372.5, 3472.5, 3500.5, 3501.0, 3501.5, 3502.0, 3509.0, 3509.5, 3510.0,
       3516.5, 3517.0, 3527.0, 3527.5, 3528.0, 3690.0, 3735.0, 3742.0, 3854.5]

        element_spectra = element_spectra.loc[selected_wavelengths]


    elif downsampling == 'range_D2O_simu':
        selected_wavelengths = [3000.9,             3004.4, 3009.1000000000004,
       3009.2000000000003,             3009.3,             3015.8,
       3020.7000000000003,             3028.9, 3035.7000000000003,
                   3039.0, 3043.1000000000004, 3047.7000000000003,
                   3047.9, 3052.7000000000003, 3057.6000000000004,
                   3064.9, 3070.1000000000004,             3076.5,
       3082.6000000000004,             3087.5, 3097.7000000000003,
       3105.1000000000004, 3106.6000000000004, 3110.2000000000003,
       3116.6000000000004, 3123.7000000000003, 3124.6000000000004,
       3128.6000000000004,             3136.4, 3141.1000000000004,
       3146.2000000000003, 3151.1000000000004, 3160.7000000000003,
                   3166.9, 3183.2000000000003,             3198.4]

        element_spectra = element_spectra.loc[selected_wavelengths]


    else:
        return element_spectra, peptides

    return element_spectra, peptides


def plot_amino_acid_spectra(element_spectra, title='Amino Acid Spectra'):
    """
    Plot spectra of individual amino of element_spectra

    Parameters:
    - element_spectra: pd.DataFrame
        Index = wavelengths (x-axis)
        Columns = amino acid names
        Values = intensity at each wavelength
    - title: str
        Title of the plot.
    """


    amino_acids_names = element_spectra.columns.tolist()
    num_acids = len(amino_acids_names)
    total_plots = num_acids + 1  # including the sum plot

    # to chose how many subplots
    cols = 4
    rows = math.ceil(total_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4), sharex=True, sharey=True)
    axes = axes.flatten()

    wavelengths = element_spectra.index.values

    for i, aa in enumerate(amino_acids_names):
        axes[i].plot(wavelengths, element_spectra[aa], label=aa)
        axes[i].set_title(f'{aa}',fontsize=15)
        axes[i].grid(True)
        axes[i].tick_params(axis='both', labelsize=15)

    # simple sum
    # sum_spectrum = element_spectra.sum(axis=1)
    sum_spectrum = element_spectra.iloc[:, :2].sum(axis=1)
    axes[num_acids].plot(wavelengths, sum_spectrum, label='Sum')
    axes[num_acids].set_title('Sum of both peptides',fontsize=15)
    axes[num_acids].tick_params(axis='both', labelsize=15)
    axes[num_acids].grid(True)


    # fig.suptitle(title, fontsize=16,y=0.90)
    fig.supxlabel('Wavelength',fontsize=16,y=0.08)
    fig.supylabel('Intensity',fontsize=16,x=0.03)
    plt.tight_layout(rect=[0, 0, 1, 0.98]) #to adjust spacing

def plot_all_amino_acids(element_spectra, title='all_amino_acids'):
    """
    Plot all the amino acids of element spectra in the same figure

    Parameters:
    - element_spectra: pd.DataFrame
        Index = wavelengths (x-axis)
        Columns = amino acid names
        Values = intensity at each wavelength
    - title: str
        Title of the plot.
    """

    wavelengths = element_spectra.index.values
    cmap = plt.cm.get_cmap('tab20')  # built-in qualitative 20-color map

    colors = {aa: cmap(i) for i, aa in enumerate(element_spectra.columns.tolist())}

    plt.figure(figsize=(10,6))
    for aa in element_spectra.columns:
        color = None
        if colors is not None and aa in colors:
            color = colors[aa]
        plt.plot(wavelengths, element_spectra[aa], label=aa, color=color)

    plt.title(title)
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.grid(True)
    plt.legend(loc='upper right', fontsize='small', ncol=4)
    plt.show()

class Data:
    """SPecific class for the amino acid data set."""

    acids = "ACDEFGHIKLMNPQRSTVWY"
    peptides = ['DGDVI','ESRVS','complex']

    types = ("neutral", "zwit","avg")
    indexes = list(range(11))

    def __init__(self, path):
        self.root = Path(path)
        self.X = np.load(self.root / "X.npy")
        self.Xpowder = np.load(self.root / "Xpowder.npy")

    def gas_acid(self, acid, _type):
        """Load amino acid in gas."""
        assert acid in self.acids, f"invalid acid name: {acid}"
        assert _type in ("neutral", "zwit"), f"wrong type: {type}"
        return np.load(self.root / "gas_acids" / f"{acid}_{_type}.npy")

    def dimer(self, acid1, acid2):
        """Load dimer in gas."""
        assert acid1 in self.acids and acid2 in self.acids
        return np.load(self.root / "gas_dimers" / f"{acid1}{acid2}.npy")

    def water_acid(self, acid, index):
        """Load amino acid in water."""
        assert acid in self.acids and 0 <= index <= 10
        return np.load(self.root / "water_acids" / f"{acid}_{index}.npy")

    # def water_peptide(self, peptide, index):
    #     """Load amino acid in water."""
    #     # assert peptide in self.peptides and 0 <= index <= 10
    #     # Special case: if index is "avg", load precomputed file
    #     if index == "avg":
    #         return np.load(self.root / "water_peptides" / f"{peptide}_avg.npy")
    #
    #     # Convert string digits to int if needed
    #     if isinstance(index, str):
    #         index = int(index)
    #     return np.load(self.root / "water_peptides" / f"{peptide}_{index}.npy")
    #
    # def no_water_peptide(self, peptide, index):
    #     """Load amino acid in water."""
    #     # assert peptide in self.peptides and 0 <= index <= 10
    #     # Special case: if index is "avg", load precomputed file
    #     if index == "avg":
    #         return np.load(self.root / "no_water_peptides" / f"{peptide}_avg.npy")
    #
    #     # Convert string digits to int if needed
    #     if isinstance(index, str):
    #         index = int(index)
    #     return np.load(self.root / "no_water_peptides" / f"{peptide}_{index}.npy")
    #
    # def powder_exp_peptide(self, peptide):
    #     """Load peptide in powder."""
    #     return np.load(self.root / "powder_exp_peptides" / f"{peptide}.npy")
    def load_gas_acids(self, _type):
        """Load amino acid in gas and make a dataframe"""
        names = self.acids
        acids = [self.gas_acid(name, _type) for name in names]
        frame = pd.DataFrame(np.column_stack(acids), columns=list(names), index=self.X)
        return frame

    def load_water_acids(self, index):
        """Load amino acid in water and make a dataframe"""
        assert 0 <= index <= 10
        names = self.acids
        acids = [self.water_acid(name, index) for name in names]
        frame = pd.DataFrame(np.column_stack(acids), columns=list(names), index=self.X)
        return frame

    def load_dimers(self):
        """Load dimers in gas and make a dataframe"""
        pairs = [a + b for a in self.acids for b in self.acids]
        acids = [self.dimer(*pair) for pair in pairs]
        frame = pd.DataFrame(np.column_stack(acids), columns=pairs, index=self.X)
        return frame
    # def load_water_peptides(self, index):
    #     """Load peptides in water and make a dataframe"""
    #     if isinstance(index, str) and index != "avg":
    #         index = int(index)
    #
    #     if isinstance(index, int):
    #         assert 0 <= index <= 10
    #     names = self.peptides
    #     peptides = [self.water_peptide(name, index) for name in names]
    #     frame = pd.DataFrame(np.column_stack(peptides), columns=list(names), index=self.X)
    #     return frame

    # def load_powder_exp_peptides(self):
    #     """Load experimental powder and make a dataframe"""
    #     self.peptides = [p for p in self.peptides if p != "complex"] #because no data is 'complex'
    #     names = self.peptides
    #     peptides = [self.powder_exp_peptide(name) for name in names]
    #     frame = pd.DataFrame(np.column_stack(peptides), columns=list(names), index=self.Xpowder)
    #     return frame
    # def load_no_water_peptides(self, index):
    #     """Load peptides in water and make a dataframe"""
    #
    #     if isinstance(index, str) and index != "avg":
    #         index = int(index)
    #
    #     if isinstance(index, int):
    #         assert 0 <= index <= 10
    #     names = self.peptides
    #     peptides = [self.no_water_peptide(name, index) for name in names]
    #     frame = pd.DataFrame(np.column_stack(peptides), columns=list(names), index=self.X)
    #     return frame

    def load_peptides(self,condition,index,solvent_type,data_source="simu"):

        """general loader for petitdes
        condition : "water" or "no_water"
        index : 0-10 or "avg"
        solvent_type: "H2O" or "D2O" """

        if isinstance(index, str) and index != "avg":
           index = int(index)

        if isinstance(index, int):
            assert 0 <= index <= 10
        if data_source == 'simu':
            folder_name = f"{data_source}_{solvent_type}_{condition}_peptides"
            X = self.X # simulated value index

        elif data_source == 'exp':
            folder_name = f"{data_source}_{solvent_type}_peptides"
            X = self.Xpowder  # experimental value index
        else:
            raise ValueError(f"Unknown data_source: {data_source}")

        peptides = [np.load(self.root / folder_name / (f"{p}_{index}.npy" if index != "avg" else f"{p}_avg.npy"))
           for p in self.peptides]

        frame = pd.DataFrame(np.column_stack(peptides), columns=self.peptides, index=X)
        return frame
