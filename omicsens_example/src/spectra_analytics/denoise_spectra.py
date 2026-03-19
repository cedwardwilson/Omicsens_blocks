"""This class can be used to downsample the spectra into target size"""

# pylint: disable=too-few-public-methods

import numpy as np
import pandas as pd
import scipy
import pywt


class DenoiseSpectra:
    """This class can use used to denoise a spectra with different methods:
    Wavelet, Fourier Transform (FT), Savitsky-golday"""

    def __init__(self, method='wavelet', **kwargs):
        """
        Parameters
        :param self:
        :param method:
            Method to be used : wavelet, fourier_transform, savitzky_golay
        :param kwargs: method-specific parameters
            wavelet: wavelet='db4', level=5
            fourier_transform: cutoff ratio = 0.1
            savitzky_golay: window_length=11, polyorder=3
        """

        self.method = method
        self.params = kwargs

    def denoise(self, element_spectra):
        "denoise the spectra according to the chosen method. Return denoise dataframe"

        method = getattr(self, f"_{self.method}", None)
        if not method:
            raise ValueError(f"Unknown resampling method: {self.method}")

        return method(element_spectra)

    def _wavelet(self, element_spectra):
        wavelet = self.params.get('wavelet', 'db4')
        level = self.params.get('level', 2)

        print("element spectra shape : ", element_spectra.shape)
        denoised = pd.DataFrame(index=element_spectra.index)

        for col in element_spectra.columns:
            signal = element_spectra[col].values

            # Auto-safe level
            max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
            use_level = min(level, max_level) if level else max_level // 2
            coeffs = pywt.wavedec(signal, wavelet, level=use_level)

            coeffs[-1] = np.zeros_like(coeffs[-1])
            coeffs[-2] = np.zeros_like(coeffs[-2])

            reconstructed = pywt.waverec(coeffs, wavelet)
            denoised[col] = reconstructed[: len(signal)]

        return denoised

    def _fourier(self, element_spectra):

        cutoff_ratio = self.params.get('cutoff_ratio', 0.1)
        denoised = pd.DataFrame(index=element_spectra.index)

        for col in element_spectra.columns:
            signal = element_spectra[col].values
            fft = np.fft.rfft(signal)
            freqs = np.fft.rfftfreq(len(signal))

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot(freqs, np.abs(fft))
            # plt.xlabel('Frequency')
            # plt.ylabel('Amplitude')
            # plt.title(f'Spectrum in Frequency Domain - {col}')
            # plt.show()

            fft[freqs > cutoff_ratio] = 0
            filtered = np.fft.irfft(fft, n=len(signal))

            denoised[col] = filtered

        return denoised

    def _savitzky_golay(self, element_spectra):

        window = self.params.get('window_length', 11)
        poly = self.params.get('polyorder', 3)

        denoised = pd.DataFrame(index=element_spectra.index)

        for col in element_spectra.columns:
            signal = element_spectra[col].values
            smooth = scipy.signal.savgol_filter(signal, window, poly)
            denoised[col] = smooth

        return denoised
