"""Module for analyzing the spectrum of a solution for the presence of specific analytes."""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch


class SpectrumExpansion:
    """Expand a spectrum in a basis of known spectra using least squares."""

    def __init__(self, spectrum, element_spectra, regression_solver):
        """Obtain a spectrum of the solution and expands it using least squares."""
        assert spectrum.shape[0] == element_spectra.shape[0]
        self.spectrum = spectrum
        self.element_spectra = element_spectra
        self.regression_solver = regression_solver

    def solve(self):
        """Uses regression solver callable."""
        return self.regression_solver(self.spectrum)

    def predict(self, W):
        """
        estimate the spectrum that would result from the
        linear combination of acids with coefficients W
        """
        assert isinstance(W, pd.Series)
        return (W * self.element_spectra).sum(axis=1)

    def R2(self):
        """
        coefficient of determination
        returns the percentage of variance of the spectrum that is explained as
        linear combination of known acids
        the residual is due to unknown acids and random noise
        the function re-draws noise every time it's called, so it will return a different value
        """
        W_hat = self.solve()
        Y_hat = self.predict(W_hat)
        return 1 - ((Y_hat - self.spectrum) ** 2).mean() / self.spectrum.var()

    def plot_solution(self):
        """Plot input spectrum together with the regression solution."""
        plt.figure(figsize=(8, 6))
        plt.plot(self.spectrum, "C1", label="spectrum")
        plt.plot(self.predict(self.solve()), "C0", label="regression")
        plt.xlabel("Frequency vector index", fontsize=16)
        plt.title("Spectrum and Regression Comparison", fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.tick_params(axis='both', which='minor', labelsize=20)
        plt.legend(fontsize=14)
        plt.tight_layout()

        plt.show()


def statistical_regression_solver(element_spectra, spectrum):
    """
    estimate the linear combination of known acids that
    would minimize least squares to spectrum of the solution
    """
    W_hat, *_ = np.linalg.lstsq(element_spectra, spectrum)
    return pd.Series(W_hat, element_spectra.columns)


def general_regression_solver(model, spectrum):
    """
    predict amino acid weights using trained regression model
    """
    spectrum_input = spectrum.values.reshape(1, -1)  # to have the correct input
    full_w_hat = model.predict(spectrum_input).flatten()  # for all amino acids
    full_series = pd.Series(full_w_hat, index=model.weights_names)
    return full_series


def neural_network_regression_solver(model, spectrum):
    """
    predict amino acid weights using trained neural network regression model
    """
    with torch.no_grad():
        model.eval()  # to be in evaluation mode
        spectrum_input = torch.tensor(spectrum.values, dtype=torch.float32).unsqueeze(
            0
        )  # to have the correct input
        full_w_hat = model(spectrum_input)
        full_series = pd.Series(full_w_hat[0], index=model.weights_names)
    return full_series


'''

def statistical_regression_solver_PLSR(spectrum, element_spectra):
    """
    estimate the linear combination of known acids that
    would minimize least squares to spectrum of the solution
    """
    # X = element_spectra
    # y = pd.Series(spectrum)

    n_components = len(element_spectra.columns)
    # Fit PLS
    pls = PLSRegression(n_components=n_components)
    pls.fit(element_spectra, spectrum)

    # Return weights as a Series
    W_hat = pls.coef_.flatten()
    return pd.Series(W_hat, element_spectra.columns)
'''
