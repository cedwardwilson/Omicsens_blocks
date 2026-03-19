"""This class can be used to downsample the spectra into target size"""


import math
import scipy
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils import check_random_state
import lttb


class DownsampleSpectra:
    """This class can use used to downsampled spectral data using various methods
    with classical methods or feature selection methods (VIP, SPA, CARS)"""

    def __init__(
        self,
        target=36,
        method='resample',
        wavelength_range=None,
        n_components=3,
        random_state=42,
        first_selection='max',
    ):
        """
        Parameters
        :param self:
        :param target: int
            Number of wavelength to keep
        :param method: str
            Downsampling method
        :param range: tuple
            the range of element spectra where to downsample
        :param n_components: int
            Number of PLS components (For VIP and CARS)
        :param random_state:
            Random seed (for CARS) and to keeep the same random everytime
        :return:
            Spectra with only -target- value
        """

        self.target = target
        self.method = method
        self.wavelength_range = wavelength_range
        self.n_components = n_components
        self.random_state = random_state
        self.first_selection = first_selection

    def downsample(self, element_spectra):
        "downsample the method according to the chosen method. Return reduces dataframe downsampled"

        # Filter by wavelength range if specified
        if self.wavelength_range is not None:
            min_wl, max_wl = self.wavelength_range
            element_spectra = element_spectra.loc[
                (element_spectra.index >= min_wl) & (element_spectra.index <= max_wl)
            ]

        method = getattr(self, f"_{self.method}", None)
        if not method:
            raise ValueError(f"Unknown resampling method: {self.method}")

        return method(element_spectra)

    def _resample(self, element_spectra):
        columns = element_spectra.columns
        resampled = scipy.signal.resample(element_spectra, self.target, axis=0)
        resampled_df = pd.DataFrame(resampled, columns=columns)
        return resampled_df

    def _resample_poly(self, element_spectra):
        """this methods is supposed to work better than 'resample' on data that are not periodic"""
        current_points = element_spectra.shape[0]

        gcd = math.gcd(
            current_points, self.target
        )  # to find the Greatest Common Divisor (used in resample_poly)
        up = self.target // gcd
        down = current_points // gcd

        # Apply polyphase resampling
        resampled = scipy.signal.resample_poly(element_spectra, up, down, axis=0)

        # Adjust column names if needed
        resampled_df = pd.DataFrame(resampled, columns=element_spectra.columns)
        return resampled_df

    def _decimate(self, element_spectra):
        """other method to downsampling in scipy"""
        columns = element_spectra.columns
        downsampling_factor = element_spectra.shape[0] // self.target
        resampled = scipy.signal.decimate(element_spectra, downsampling_factor, axis=0)
        resampled_df = pd.DataFrame(resampled, columns=columns)
        return resampled_df

    def _lttb(self, element_spectra):
        """Largest-Triangle-Three-Buckets (LTTB)
        method developped by https://pypi.org/project/lttb/
        """
        # lttb only works with data of shape N,2 (N being 38000  here)

        wavelengths = element_spectra.index.to_numpy()
        downsampled_data = {}

        for aa in element_spectra.columns:  # for all amino acids
            intensity = element_spectra[aa].to_numpy()
            data = np.column_stack((wavelengths, intensity))  # for one amino acid
            reduced = lttb.downsample(data, n_out=self.target)
            downsampled_data[aa] = reduced[:, 1]  # just stored the downsampled intensity

        # Use lttb only for the wavelength columns only for
        # the first amino acid (bias : maybe it is not the same for all?)
        wavelengths_downsampled = lttb.downsample(
            np.column_stack((wavelengths, element_spectra.iloc[:, 0])), n_out=self.target
        )[:, 0]

        return pd.DataFrame(downsampled_data, index=wavelengths_downsampled)

    def _average(self, element_spectra):
        wavelengths = element_spectra.index.values  # wavelength axis
        intensities = element_spectra.values  # intensity values

        bins = np.array_split(np.arange(len(wavelengths)), self.target)

        # Average over bins
        binned_wavelengths = [wavelengths[idx].mean() for idx in bins]
        binned_intensities = [intensities[idx].max(axis=0) for idx in bins]

        resampled_df = pd.DataFrame(
            binned_intensities, columns=element_spectra.columns, index=binned_wavelengths
        )
        return resampled_df

    def _max(self, element_spectra):
        wavelengths = element_spectra.index.values  # wavelength axis
        intensities = element_spectra.values  # intensity values

        bins = np.array_split(np.arange(len(wavelengths)), self.target)

        # max over bins_intensities and average over bins_wavelengths
        binned_wavelengths = [wavelengths[idx].mean() for idx in bins]
        binned_intensities = [intensities[idx].max(axis=0) for idx in bins]

        resampled_df = pd.DataFrame(
            binned_intensities, columns=element_spectra.columns, index=binned_wavelengths
        )
        return resampled_df

    def _regular(self, element_spectra):
        wavelengths = element_spectra.index.values  # wavelength axis
        intensities = element_spectra.values  # intensity values

        selected_wavelengths = []
        selected_intensities = []

        bins = np.array_split(np.arange(len(wavelengths)), self.target)
        for idx in bins:
            mid_idx = idx[len(idx) // 2]  # middle index in this bin
            selected_wavelengths.append(wavelengths[mid_idx])
            selected_intensities.append(intensities[mid_idx])

        resampled_df = pd.DataFrame(
            selected_intensities, columns=element_spectra.columns, index=selected_wavelengths
        )
        return resampled_df

    def _vip(self, element_spectra):
        """Compute Variable Importance in Projection (VIP) scores using PLS regression.
        Select the top N most important wavelengths
        based on the paper of Chong et al.
        https://www-sciencedirect-com.lama.univ-amu.fr/science/article/pii/S0169743905000031?via%3Dihub
        input:
        element_spectra : pd.DataFrame (38 000,20)
            spectra data of each amino acids with wavelengths as index
        n_components : int
            Number of PLS components to use for dimensionality reduction
        target : number of wavelengths to be selected based on highest VIP scores (36 by default)

        returns:
        reduced_df_sorted : pd.DataFrame
            subset of the original element_spectra with only
            the top N selected wavelengths sorted by wavelength
        selected_wavelengths : The selected wavelengths
        """

        X = element_spectra.T.values  # shape : (20,38000)
        wavelengths = element_spectra.index
        y = np.arange(X.shape[0]).reshape(-1, 1)  # the 20 amino acids
        pls = PLSRegression(n_components=self.n_components)
        pls.fit(X, y)  # To do classification

        # VIP score computation
        t = pls.x_scores_
        w = pls.x_weights_
        b = pls.y_loadings_
        p, h = w.shape

        s = np.diag(t.T @ t @ b.T @ b).reshape(h, -1)
        total_s = s.sum()
        vip_scores = np.sqrt(p * (w**2 @ s).flatten() / total_s)

        # Select the wavelength with the highest VIP score meaning
        # the highest differenece between amino acids
        top_indices = np.argsort(vip_scores)[-self.target :]  # return the indices of the top value
        selected_wavelengths = wavelengths[
            top_indices
        ]  # array containing all the wavelengths value

        # create the updated reduced dataframe (with only N wavelengths)
        reduced_df = element_spectra.loc[selected_wavelengths]
        reduced_df_sorted = reduced_df.sort_index()  # to sorted
        return reduced_df_sorted

    def _spa(self, element_spectra):
        """
        Successive Projection Algorithm for wavelength selection from a spectra DataFrame.
        https://www.sciencedirect.com/science/article/abs/pii/S0169743901001198
        Parameters:
            element_spectra: pd.DataFrame, shape (wavelengths, amino acids)
                spectra data of each amino acids with wavelengths as index
            target: int
                Number of wavelengths to select.

        Returns:
            reduced_df: reduced dataframe with the selected wavelength
            selected_wavelengths: list of selected wavelength values sorted (from index)
        """

        # Transpose to shape (samples, features): (20 amino acids, N wavelengths)
        X = element_spectra.T.values  # shape: (20, N)
        wavelengths = element_spectra.index

        # initialization
        _, n_features = X.shape
        selected = []
        remaining = list(range(n_features))

        # choosing the first wavelength with the highest spectra (highest norm)
        if self.first_selection == 'max':
            norms = np.linalg.norm(X, axis=0)
            first = np.argmax(norms)
        # choosing the highest difference between sum and complex
        elif self.first_selection == 'diff':
            diff = np.abs((X[0] + X[1]) - X[2])
            first = np.argmax(diff)
        else:
            raise ValueError

        selected.append(first)
        remaining.remove(first)

        # projections
        for _ in range(1, self.target):
            proj_errors = []  # to store the nord of the projection errors (xj - xproj)
            for j in remaining:  # loop on all remaining variables
                xj = X[:, j].reshape(-1, 1)
                X_selected = X[:, selected]
                proj = X_selected @ np.linalg.pinv(X_selected) @ xj
                error = np.linalg.norm(xj - proj)
                proj_errors.append(error)
            next_idx = remaining[np.argmax(proj_errors)]  # select the most 'orthogonal' wavelength
            selected.append(next_idx)
            remaining.remove(next_idx)

        # Get selected wavelengths
        selected_wavelengths = wavelengths[selected]
        selected_wavelengths = selected_wavelengths.sort_values()
        print("selected wavelengths : ", selected_wavelengths)
        reduced_df = element_spectra.loc[selected_wavelengths]
        return reduced_df

    def _cars(self, element_spectra, n_sample_runs=100, fit_samples_ratio=0.9):
        """Competitive Adaptive Reweighted Sampling (CARS) for
            feature selection based on the principle
             'survival of the fittest' give the variable that give the smallest prediciton error.
             based on the code of
             https://www.sciencedirect.com/science/article/abs/pii/S0003267009008332
        Parameters:
        -----------
        element_spectra : pd.DataFrame
            Spectral data (wavelengths and amino acids)
        n_components : int
            Number of PLS components.
        target : int
            Target number of wavelengths to select.
        n_sample_runs : int
            Number of iterations (100 by default).
        fit_samples_ratio : float
            Ratio of samples used in model fitting.
        random_state : int
            For reproductivity with randomness

        Returns:
        --------
        reduced_df_sorted : pd.DataFrame
            Reduced spectral dataframe with selected wavelengths.
        selected_wavelengths : Selected wavelengths sorted by size
        """

        X = element_spectra.T.values  # shape: (20, 38 000)
        wavelengths = element_spectra.index
        y = np.arange(X.shape[0]).reshape(-1, 1)  # classification

        rng = check_random_state(self.random_state)
        n_samples, n_features = X.shape
        comp = np.min([self.n_components, n_samples - 1])
        n_fit_samples = int(n_samples * fit_samples_ratio)

        # EDF is utilized to remove the wavelengths which are of relatively
        # small absolute regression coefficients by force.
        a = (n_features / 2) ** (1 / (n_sample_runs - 1))
        k = np.log(n_features / 2) / (n_sample_runs - 1)
        edf = (a * np.exp(-k * (np.arange(n_sample_runs) + 1)) * n_features).astype(int)

        current_wavelengths = np.arange(n_features)

        for i in range(n_sample_runs):

            # Randomly select sample (amino acids 18 here)
            fit_idx = rng.choice(n_samples, n_fit_samples, replace=False)
            X_fit = X[fit_idx][:, current_wavelengths]
            y_fit = y[fit_idx]

            # PLS regression and compute coeff weights for each wavelengths
            pls = PLSRegression(n_components=comp)
            pls.fit(X_fit, y_fit)
            coef = np.abs(pls.coef_).flatten()
            coef /= coef.sum()  # normalize weights

            weights = np.zeros(n_features)
            weights[current_wavelengths] = coef

            # how many wavelengths to keep at this iteration depending of the computation of edf
            n_keep = max(edf[i], self.target)

            # keeping n_keep highest weights
            top_k_indices = np.argsort(-weights)[:n_keep]

            # ARS: to introduce randomness
            probs = weights[top_k_indices]
            probs /= probs.sum()  # to normalize again

            new_size = int(n_keep * fit_samples_ratio)
            new_size = max(new_size, self.target)
            selection = rng.choice(top_k_indices, size=new_size, replace=False, p=probs)

            # update of current wavelength
            current_wavelengths = np.unique(selection)

            # early stop if target is reached
            if len(current_wavelengths) <= self.target:
                break

        selected_wavelengths = wavelengths[current_wavelengths]
        reduced_df = element_spectra.loc[selected_wavelengths]
        reduced_df_sorted = reduced_df.sort_index()

        return reduced_df_sorted
