import sys
import os
import torch
import numpy as np
import pandas as pd
import joblib
from functools import partial
from matplotlib import pyplot as plt
import spectra_analytics.ML.model as mo
from spectra_analytics.dataset import load_multiple_peptide_configs
from spectra_analytics.statistical_analysis import ParametricSpectrumAnalysis
from spectra_analytics.spectrum_expansion import neural_network_regression_solver


# 3. THE INFERENCE FUNCTION
def run_spectrum_inference(
    model_path,
    peptide_config,
    true_weights,
    data_source="exp",
    downsampling="D2O_exp",
    noise_level=0.01,
    num_stats_samples=1000,
    device="cpu",
):
    """
    Runs spectral inference using a trained model and specific peptide configurations.
    """
    # Load the spectral data based on the provided config
    element_spectra, peptides = load_multiple_peptide_configs(
        peptide_config,
        data_source=data_source,
        downsampling=downsampling,
    )

    # Load the Model Checkpoint (Mapped to CPU for Mac compatibility)
    checkpoint = torch.load(model_path, map_location=torch.device(device))

    # Initialize Model from checkpoint config
    model = mo.CNN2Layer(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.weights_names = peptides

    # Prepare the solver
    regression_solver = partial(neural_network_regression_solver, model)

    # Perform Analysis
    solution = ParametricSpectrumAnalysis(
        element_spectra=element_spectra,
        element_weights=pd.Series(true_weights, index=peptides),
        element_noise=pd.Series([noise_level for _ in peptides], index=peptides),
        regression_solver=regression_solver,
    )

    # Output Results
    print("-" * 30)
    print(f"Model: {os.path.basename(model_path)}")
    print(f"R2 coefficient is : {solution.R2():.4f}")

    mean, std = solution.R2_stats(num_samples=num_stats_samples)
    print(f"Stats (n={num_stats_samples}) -> Mean: {mean:.4f}, Std: {std:.4f}")
    print("-" * 30)

    # Visualizations
    solution.plot_solution()
    solution.plot_sampling_distribution(num_samples=2000)
    solution.plot_confidence_intervals(num_samples=50)

    return solution


# 4. EXECUTION BLOCK
if __name__ == "__main__":
    # Define Inputs
    MY_MODEL = os.path.join(
        "trained_models/relu_down_exp_D2O_0_1_100000_CNN2Layer_16_32.pth.tar"
    )

    MY_PEPTIDES = {
        "DGDVI": ("D2O", "no_water", "1"),
        "ESRVS": ("D2O", "no_water", "1"),
        "complex": ("D2O", "no_water", "1"),
    }

    MY_WEIGHTS = [0.5, 0.5, 0.5]

    # Run the function
    analysis_result = run_spectrum_inference(
        model_path=MY_MODEL,
        peptide_config=MY_PEPTIDES,
        true_weights=MY_WEIGHTS,
        device="cpu",  # Force CPU to avoid CUDA errors on Mac
    )
