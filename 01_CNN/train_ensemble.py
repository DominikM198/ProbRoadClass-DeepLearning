#!/usr/bin/env python

"""
Trains sequentially ensemble with 30 members
"""

### IMPORTS ###
import const
from model_settings.settings_calibration import interpret_json
import const
import utils
from training_evaluation.train_validate import train_evaluate_segmentation_model

### CONSTANTS ###
n_members = 30

### Main ###
if __name__ == "__main__":
    """
    Function for training and/or evaluating ensemble models.
    """
    
    for i in range(1, n_members +1):

        # Set the random seed
        utils.set_seed(i)

        # Set the path to the experiment settings
        settings_path = const.SETTINGS_DIR.joinpath("c:\\Road_segmentation\\road-seg-hist-maps\\01_CNN\\model_settings\\Road_classification_final\\Siegfried_settings_classification_{}.json".format(i))

        # Get the experiment settings
        settings = interpret_json(path = settings_path)

        # Train and validate the ensemble model
        train_evaluate_segmentation_model(settings)

