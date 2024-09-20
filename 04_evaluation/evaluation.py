#!/usr/bin/env python

"""
Evaluate the road classification.
"""

import geopandas as gpd

tiles = ["199_1941", "385_1941"]
local_path_processed = "road_classification\\"
subfolder_predictions = "\\Predictions\\"
subfolder_GT = "\\GT\\"
name_GT = "_GT"
name_predictions = "_PRED"
road_classes = [1, 2, 3, 4, 5]

completness_list = []
correctness_list = []
quality_list = []
length_class_list = []
length_predicted_class_list = []
for road_class in road_classes:
    completeness = 0
    correctness = 0
    length_GT_witin_vectorized_line_buffer = 0
    length_GT = 0
    length_vectorized_line_within_GT_bufer = 0
    length_vectorized_lines = 0

    for tile in tiles:
        # read the predictions
        predictions = gpd.read_file(local_path_processed + tile + subfolder_predictions + "class_{}{}.shp".format(str(road_class), name_predictions)).dissolve()
        predictions_buffer = gpd.read_file(local_path_processed + tile + subfolder_predictions + "class_{}_buffer{}.shp".format(str(road_class), name_predictions)).dissolve()

        # read the ground truth
        ground_truth = gpd.read_file(local_path_processed + tile + subfolder_GT + "class_{}{}.shp".format(str(road_class), name_GT)).dissolve()
        ground_truth_buffer = gpd.read_file(local_path_processed + tile + subfolder_GT + "class_{}_buffer{}.shp".format(str(road_class), name_GT)).dissolve()

        # calculate length of predictions within the buffer of the ground truth
        gt_within_prediction_buffer = gpd.overlay(predictions_buffer, ground_truth, how="intersection", keep_geom_type=False)
        length_GT_witin_vectorized_line_buffer = length_GT_witin_vectorized_line_buffer + gt_within_prediction_buffer.length.sum()
        length_GT = length_GT + ground_truth.length.sum()

        # calculate length of ground truth within the buffer of the predictions
        predictions_within__GT_buffer = gpd.overlay(ground_truth_buffer, predictions, how="intersection", keep_geom_type=False)
        length_vectorized_line_within_GT_bufer = length_vectorized_line_within_GT_bufer + predictions_within__GT_buffer.length.sum()
        length_vectorized_lines = length_vectorized_lines + predictions.length.sum()

    # calculate correctness and completeness
    completeness = length_GT_witin_vectorized_line_buffer / length_GT
    correctness = length_vectorized_line_within_GT_bufer / length_vectorized_lines
    quality =   (completeness * correctness) / (completeness - completeness* correctness + correctness)

    # append to list
    length_class_list.append(length_GT)
    length_predicted_class_list.append(length_vectorized_lines)
    completness_list.append(completeness)
    correctness_list.append(correctness)
    quality_list.append(quality)

    print("Class {} Scores:".format(road_class))
    print("Completeness: {}".format(round(completeness, 4)))
    print("Correctness: {}".format(round(correctness, 4)))
    print("Quality: {}".format(round(quality, 4)))
    print("")

weighted_completeness = sum([completness_list[i] * length_class_list[i] for i in range(len(completness_list))]) / sum(length_class_list)    
weighted_correctness = sum([correctness_list[i] * length_predicted_class_list[i] for i in range(len(length_predicted_class_list))]) / sum(length_predicted_class_list)
print("Weighted Scores:")
print("Completeness: {}".format(round(weighted_completeness, 4)))
print("Correctness: {}".format(round(weighted_correctness, 4)))



