#!/usr/bin/env python

"""
Implements the constants for the research project.
"""
### IMPORTS ###
import os
from pathlib import Path

### DIRECTORIES ###
ROOT_DIR = Path(__file__).resolve().parents[0].parents[0]
MODELS_DIR = ROOT_DIR.joinpath("01_CNN").joinpath("models")
DATA_DIR = ROOT_DIR.joinpath("datasets")
LOGS_DIR = ROOT_DIR.joinpath("01_CNN").joinpath("runs")
SETTINGS_DIR = ROOT_DIR.joinpath("01_CNN").joinpath("model_settings")
STORAGE_DIR = ROOT_DIR.joinpath("01_CNN").joinpath("storage")
MODEL_STORAGE_DIR = STORAGE_DIR.joinpath("01_CNN").joinpath("models")
PLOT_DIR = STORAGE_DIR.joinpath("01_CNN").joinpath("plots")
RESULTS_DIR = STORAGE_DIR.joinpath("01_CNN").joinpath("results")

### CONSTANTS ###
SEED = 42
