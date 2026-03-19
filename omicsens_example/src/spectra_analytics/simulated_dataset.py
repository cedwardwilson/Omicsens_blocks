"""Sampling data sets for element spectra to be used for training."""

# pylint: disable=too-many-locals

import numpy as np
import pandas as pd


class SimulatedSpectraDataset:
    """This class generate a dataset based on the simulated spectra of each amino acid
    -works for a single configuration (old version) when config_groups = None
    -works for multi configuration mode when config_groups is provided
    """

    def __init__(
        self,
        element_spectra,
        element_noise,
        config_groups: dict = None,  # single mode or multi mode
        unknown_element_spectra=None,
        unknown_element_noise=None,
    ):
        self.element_spectra = element_spectra
        # self.element_names = element_spectra.columns.tolist()
        self.element_noise = element_noise

        self.config_groups = config_groups
        self.multi_config = config_groups is not None

        if self.multi_config:
            self.element_names = list(config_groups.keys())
        else:
            # old behavior
            self.element_names = element_spectra.columns.tolist()

        self.unknown_element_spectra = unknown_element_spectra
        if unknown_element_spectra is None:
            self.unknown_element_spectra = None
            self.unknown_element_names = None
        else:
            self.unknown_element_spectra = unknown_element_spectra
            self.unknown_element_names = unknown_element_spectra.columns.tolist()
        if unknown_element_noise is None:
            unknown_element_noise = 0
        self.unknown_element_noise = unknown_element_noise

    def generate_sample(self, p_zero=0.3):
        '''Generate one spectrum (one sample).
        Return the spectrum and the weight used to generate the spectrum.
        '''
        element_standard_noise = pd.DataFrame(
            np.random.randn(*self.element_spectra.shape),
            columns=self.element_spectra.columns,
            index=self.element_spectra.index,
        )
        # Initialize all weights and all noises to zero
        weights = pd.Series(
            np.random.uniform(0.0, 1.0, size=len(self.element_names)), index=self.element_names
        ).round(2)

        # to add sparsity
        mask = np.random.rand(len(weights)) < p_zero
        weights[mask] = 0.0

        spectrum = pd.Series(0.0, index=self.element_spectra.index)

        if not self.multi_config:
            noise = pd.Series(
                np.random.uniform(0.0, self.element_noise, size=len(self.element_names)),
                index=self.element_names,
            )

            spectrum = (weights * (self.element_spectra + noise * element_standard_noise)).sum(
                axis=1
            )

        # When there are multiple configurations
        else:
            for element in self.element_names:
                w = weights[element]
                if w == 0.0:
                    continue

                configs = self.config_groups[element]
                chosen_config = np.random.choice(configs)

                base_spec = self.element_spectra[chosen_config]
                noise_scale = np.random.uniform(0, self.element_noise)
                noise_vec = noise_scale * element_standard_noise[chosen_config]

                spectrum += w * (base_spec + noise_vec)

        if self.unknown_element_spectra is not None:
            assert len(self.unknown_element_noise) == self.unknown_element_spectra.shape[1]

            unknown_standard_noise = pd.DataFrame(
                np.random.randn(*self.unknown_element_spectra.shape),
                columns=self.unknown_element_spectra.columns,
                index=self.unknown_element_spectra.index,
            )
            # Initialize all weights and all noises to zero
            unknown_weights = pd.Series(
                np.random.uniform(0.0, 1.0, size=len(self.unknown_element_names)),
                index=self.unknown_element_names,
            )

            spectrum += (
                unknown_weights
                * (
                    self.unknown_element_spectra
                    + self.unknown_element_noise * unknown_standard_noise
                )
            ).sum(axis=1)

        return spectrum, weights

    def generate(self, n_samples):
        '''The function to generate all spectra (n_samples spectrum) with the associated weights'''
        spectra = []
        weights = []

        for _ in range(n_samples):
            s, w = self.generate_sample()
            spectra.append(s)
            weights.append(w.values)

        return (
            pd.DataFrame(spectra, columns=self.element_spectra.index),
            pd.DataFrame(weights, columns=self.element_names),
        )

    def save(self, spectra, weights, path="synthetic_dataset.npz"):
        '''To save spectra the generated simulated spectra
        and the associated weight used to generate the spectra
        '''
        np.savez(
            path,
            spectra=spectra.values,
            weights=weights.values,
            spectra_columns=spectra.columns.to_numpy(),
            weight_columns=weights.columns.to_numpy(),
        )
