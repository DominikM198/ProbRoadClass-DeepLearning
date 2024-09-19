#!/usr/bin/env python

"""
Preprocess the predictions for the road classification for evaluation.
"""

import geopandas as gpd

tiles = ["199_1941", "385_1941"]
name_predictions = "_road_geoms_analytical_breakpoints_minline_150m_seg_10m_6m.shp"
local_path_predictions = "analytical_breakpoints_minline_150m_seg_10m_6m\\"
local_path_processed = "road_classification\\"
subfolder = "\\Predictions\\"
road_classes = [1, 2, 3, 4, 5]

for tile in tiles:
    # read the predictions
    predictions = gpd.read_file(local_path_predictions + tile[:3] + name_predictions)
    predictions.set_crs(epsg=21781, inplace=True)

    # rename column "road_class" to "class"
    predictions.rename(columns={"road_cat": "class"}, inplace=True)

    for road_class in road_classes:
        # filter the predictions by road class
        predictions_class = predictions[predictions["class"] == road_class]


        # paths to save the predictions
        path_save = local_path_processed + tile + subfolder + "class_{}_PRED".format(str(road_class)) + ".shp"
        path_save_buffer = local_path_processed + tile + subfolder + "class_{}_buffer_PRED".format(str(road_class)) + ".shp"
        print(path_save)

        # save the predictions
        predictions_class.to_file(path_save)

        # create a buffer around the predictions
        predictions_class["geometry"] = predictions_class.buffer(5)

        # save the buffered predictions
        predictions_class.to_file(path_save_buffer)