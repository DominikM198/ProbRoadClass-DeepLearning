#!/usr/bin/env python

"""
Implements the training and testing pipeline for deep learning models.
"""

### IMPORTS ###
import const
from model_settings.settings_calibration import interpret_json
import const
import utils
from training_evaluation.train_validate import train_evaluate_segmentation_model


### Main ###
if __name__ == "__main__":
    """
    Function for training and/or evaluating models.
    """
    path = "c:\\Road_segmentation\\road-seg-hist-maps\\01_CNN\\model_settings\\Road_classification_final\\Siegfried_settings_classification_42_lr_tuning.json"

    # Set the random seed
    utils.set_seed(const.SEED)
    
    # Set the path to the experiment settings
    settings_path = const.SETTINGS_DIR.joinpath(path)

    print(settings_path)

    # Get the experiment settings
    settings = interpret_json(path = settings_path)

    # Train and evaluate the model
    train_evaluate_segmentation_model(settings)



