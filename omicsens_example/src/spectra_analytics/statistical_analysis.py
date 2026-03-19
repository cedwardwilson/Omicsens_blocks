"""Analysis of regression of spectra in order to know
how many experiments are required for statistical significance.
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import scipy
import spectra_analytics.spectrum_expansion as se
from matplotlib.ticker import FormatStrFormatter


class ParametricSpectrumAnalysis(se.SpectrumExpansion):
    """Statistical analysis for required number of experiments related to a given spectrum,
    basis spectra, noise levels and unknown spectra.
    """

    def __init__(
        self,
        element_spectra,
        element_weights,
        element_noise,
        regression_solver,
        unknown_element_spectra=None,
        unknown_element_weights=None,
        unknown_element_noise=None,
    ):
        """
        element_weights: spectrum weight coefficients
        element_noise: random noise coefficients

        """
        self.spectrum = None
        self.element_spectra = element_spectra
        self.element_weights = element_weights
        self.element_noise = element_noise
        if unknown_element_spectra is not None:
            if unknown_element_weights is None:
                unknown_element_weights = [0 for _ in range(unknown_element_spectra.shape[1])]
            if unknown_element_noise is None:
                unknown_element_noise = [0 for _ in range(unknown_element_spectra.shape[1])]
        self.unknown_element_spectra = unknown_element_spectra
        self.unknown_element_weights = unknown_element_weights
        self.unknown_element_noise = unknown_element_noise
        self.remix()
        super().__init__(self.spectrum, self.element_spectra, regression_solver)

    def remix(self):
        """
        Re-draw gaussian random noise and re-estimate the spectrum of the solution
        """
        element_standard_noise = pd.DataFrame(
            np.random.randn(*self.element_spectra.shape),
            columns=self.element_spectra.columns,
            index=self.element_spectra.index,
        )

        self.spectrum = (
            self.element_weights
            * (self.element_spectra + self.element_noise * element_standard_noise)
        ).sum(axis=1)

        if self.unknown_element_spectra is not None:
            assert len(self.unknown_element_noise) == self.unknown_element_spectra.shape[1]

            unknown_standard_noise = pd.DataFrame(
                np.random.randn(*self.unknown_element_spectra.shape),
                columns=self.unknown_element_spectra.columns,
                index=self.unknown_element_spectra.index,
            )
            self.spectrum += (
                self.unknown_element_weights
                * (
                    self.unknown_element_spectra
                    + self.unknown_element_noise * unknown_standard_noise
                )
            ).sum(axis=1)

    def confidence_intervals(self, num_samples, level=0.95):
        """
        confidence intervals of estimated coefficients of known acids
        the process is repeated `num_samples` times
        the confidence interval decreases on average with each measurement,
        and converges to zero at infinity
        """
        frame = self.sampling_distribution(num_samples)
        intervals = frame.expanding().apply(lambda x: self.calculate_confidence_interval(x, level))
        return intervals

    def sampling_distribution(self, num_samples):
        """
        estimated coefficients of Y as linear combination of known acids
        the process is repeated `num_samples` times
        """
        W_hats = []

        for _ in range(num_samples):
            self.remix()
            W_hats.append(self.solve())

        frame = pd.DataFrame(W_hats)
        return frame

    def R2(self):
        """
        coefficient of determination
        returns the percentage of variance of the spectrum that is explained as
        linear combination of known acids
        the residual is due to unknown acids and random noise
        the function re-draws noise every time it's called, so it will return a different value
        """
        self.remix()
        W_hat = self.solve()
        Y_hat = self.predict(W_hat)
        return 1 - ((Y_hat - self.spectrum) ** 2).mean() / self.spectrum.var()

    def R2_stats(self, num_samples):
        """
        Compute mean and std of R² across multiple samples.
        """
        r2_values = [self.R2() for _ in range(num_samples)]
        r2_mean = np.mean(r2_values)
        r2_std = np.std(r2_values)
        return r2_mean, r2_std

    @staticmethod
    def calculate_confidence_interval(X, level=0.95):
        """
        accepts a list-like of measurements `X`
        and calculates the size of confidence interval at `level` confidence.
        ("level" is the complemetary of p-value, that is, p = 1-level)
        """
        N = len(X)
        c = 1 - (1 - level) / 2
        c = scipy.stats.t.ppf(c, N - 1)
        half_interval = c * (np.var(X) / N) ** 0.5
        return 2 * half_interval

    def plot_sampling_distribution(self, num_samples, num_ticks=5):
        """Plot sampling distribution for each element's weights
        as a solution for the given spectrum regression.
        The higher the noise then wider the distribution will be.
        """
        frame = self.sampling_distribution(num_samples)
        # axes = frame.hist(bins=20, layout=(1, frame.shape[1]))

        # Number of histograms (columns)
        n_cols = frame.shape[1]

        # Force a single row and control the figure size
        axes = frame.hist(
            bins=20,
            layout=(1, n_cols),
            figsize=(4.5 * n_cols, 1.5 * n_cols)  # width, height in inches
        )

        for ax in axes.ravel():
            acid = ax.title.get_text()
            if acid:
                w = self.element_weights[acid]
                ax.axvline(w, color="k", linewidth=2)
                y_min, y_max = ax.get_ylim()
                ax.text(
                    w,
                    y_min + 0.05 * (y_max - y_min),
                    f"W = {w:.2f}",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7},
                )
                x_min, x_max = ax.get_xlim()
                ticks = np.linspace(x_min, x_max, num_ticks)
                ax.set_xticks(ticks)

                # Increase tick label and axis label font sizes
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                ax.tick_params(axis='x', labelsize=16)  # x-axis tick labels
                ax.tick_params(axis='y', labelsize=12)  # y-axis tick labels

        plt.suptitle(
            "Sampling distribution of element weights \n from multiple regressions of the spectrum"
        )
        plt.tight_layout()
        plt.show()

    def plot_confidence_intervals(self, num_samples, level=0.95):
        """Plot confidence intervals up to a given number of samples of each of
        the weights computed from the sampling distribution.
        """
        frame = self.confidence_intervals(num_samples, level)
        frame.plot()
        plt.title(f"Confidence interval at {level} \n as a function of number of experiments")
        plt.xlabel("Number of experiments")
        plt.ylabel("Confidence interval of weights")
        plt.show()
