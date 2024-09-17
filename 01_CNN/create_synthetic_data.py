#!/usr/bin/env python
### IMPORTS ###
import datetime

import numpy as np
import const
import utils
import os
from PIL import Image
import cv2
from skimage.util import view_as_windows
import fiona
from shapely.geometry import shape, mapping, LineString
from shapely import offset_curve
from osgeo import gdal
import shutil
from tqdm import tqdm
gdal.DontUseExceptions()

# Constants
Image.MAX_IMAGE_PIXELS = 140000000
# Buffer size for the road labels
# 10 for binary segmentation
# 13 for road classification
BUFFERSIZE_LABELS = 10
BUFFERSIZE_BACKGROUND_OVERPAINTING = 17
BREAKPOINT_BUFFER_RADIUS = 8

# Define the standard deviation for the variability of the symbol painting
PAINTING_STD = {
    'offset': 0.2,  # meter
    'dash': 0.5,  # meter
    'thickness': 0.35  # pixel
}

# Define the params for the road class symbolization
SYMBOLIZATION = {
    5: {
        'offset': 8.25,  # meter
        'thickness_line_1': 3,  # pixel
        'thickness_line_2': 1.5  # pixel
    },
    4: {
        'offset': 5.6,  # meter
        'thickness': 1.5  # pixel
    },
    3: {
        'offset': 4.7,  # meter
        'gap_length': 7.5,  # meter
        'dash_length': 11,  # meter
        'thickness_line_dashed': 1.75,  # pixel
        'thickness_line_solid': 1.75  # pixel
    },
    2: {
        'thickness': 2,  # pixel
    },
    1: {
        'gap_length': 12.5,  # meter
        'dash_length': 17.5,  # meter
        'thickness': 2,  # pixel
    }
}


### UTILS ###
def _generate_tiling(image_path, w_size, mode='rgb'):
    """
    Generate tiling images

    Args:
    image_path: str, the path to the image
    w_size: int, the window size
    mode: str, the mode of the image

    Returns:
    np.array, the tiling images
    """

    # Generate tiling images
    win_size = w_size
    pad_px = win_size // 2

    # Read image
    in_img = np.array(Image.open(image_path))

    # Padding image
    if mode == 'rgb':
        img_pad = np.pad(in_img, [(pad_px, pad_px), (pad_px, pad_px), (0, 0)], 'constant')
        tiles = view_as_windows(img_pad, (win_size, win_size, 3), step=pad_px)
        tiles_lst = []
        for row in range(tiles.shape[0]):
            for col in range(tiles.shape[1]):
                tt = tiles[row, col, 0, ...].copy()
                tiles_lst.append(tt)
        tiles_array = np.concatenate(tiles_lst)
        # You must reshape the tiles_array into (batch_size, width, height, 3)
        tiles_array = tiles_array.reshape(int(tiles_array.shape[0] / w_size), w_size, w_size, 3)
    else:
        img_pad = np.expand_dims(np.pad(in_img, [(pad_px, pad_px), (pad_px, pad_px)], 'constant'), axis=2)
        tiles = view_as_windows(img_pad, (win_size, win_size, 1), step=pad_px)
        tiles_lst = []
        for row in range(tiles.shape[0]):
            for col in range(tiles.shape[1]):
                tt = tiles[row, col, 0, ...].copy()
                tiles_lst.append(tt)
        tiles_array = np.concatenate(tiles_lst)
        # You must reshape the tiles_array into (batch_size, width, height, 3)
        tiles_array = tiles_array.reshape(int(tiles_array.shape[0] / w_size), w_size, w_size, 1)
    return tiles_array


def _lv03_to_pixel(coords_lv03, geotransform):
    """
    Convert LV03 coordinates to pixel coordinates

    Args:
    coords_lv03: tuple, LV03 coordinates
    geotransform: tuple, geotransform parameters

    Returns:
    tuple, pixel coordinates
    """
    x_pixel = int((coords_lv03[0] - geotransform[0]) / geotransform[1] - 0.5)
    y_pixel = int((coords_lv03[1] - geotransform[3]) / geotransform[5] - 0.5)
    coords_pixel = (x_pixel, y_pixel)
    return coords_pixel


def _get_offset_geom(geom, distance):
    """
    Get the geometry of a line with a parallel offset.

    Args:
    geom: LineString, the original geometry
    distance: float, the offset distance

    Returns:
    LineString, the offset geometry
    """
    offset_geom = offset_curve(geom, distance, join_style='mitre', mitre_limit=30.0)
    return offset_geom


def _get_random_siegfried_background(shape):
    """
    Get a random Siegfried background.

    Args:
    shape: tuple, the shape of the image

    Returns:
    np.array, the Siegfried background
    """
    r = np.random.normal(239.7, 3.0, size=(shape[0], shape[1], 1))
    g = np.random.normal(237.8, 3.0, size=(shape[0], shape[1], 1))
    b = np.random.normal(222.6, 3.5, size=(shape[0], shape[1], 1))
    if shape[2] == 4:
        a = 255 * np.ones((shape[0], shape[1], 1))
        siegfried_background = np.concatenate((r, g, b, a), axis=2)
    else:
        siegfried_background = np.concatenate((r, g, b), axis=2)
    return siegfried_background


def _get_random_siegfried_black(shape):
    """
    Get a random Siegfried black.

    Args:
    shape: tuple, the shape of the image

    Returns:
    np.array, the Siegfried black
    """
    ds_black_values = gdal.Open(str(const.DATA_DIR.joinpath('siegfried_black_values.tif')))
    array_black_values = ds_black_values.ReadAsArray()
    array_black_values = array_black_values.reshape(
        (array_black_values.shape[0], array_black_values.shape[1] * array_black_values.shape[2]))
    siegfried_black = 255 * np.ones(shape=shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            idx = np.random.choice(array_black_values.shape[1])
            siegfried_black[i, j, 0] = array_black_values[0, idx]
            siegfried_black[i, j, 1] = array_black_values[1, idx]
            siegfried_black[i, j, 2] = array_black_values[2, idx]
    return siegfried_black


def _draw_dashed_line(img, geom, geotransform, color, thickness, gap_length, dash_length):
    """
    Draw a dashed line on an image.

    Args:
    img: np.array, the image
    geom: LineString, the geometry
    geotransform: tuple, the GDAL geotransform parameters
    color: tuple, the color
    thickness: int, the thickness
    gap_length: float, the gap length
    dash_length: float, the dash length

    Returns:
    np.array, the image with the dashed line
    """
    # Cumulate length of rendered segments
    cumulated_length = 0
    # Cumulate length of current rendered line
    rendered_length = 0

    # Loop over the segments
    for i in range(1, len(geom.coords)):
        rendered_length = cumulated_length

        # Current segment
        line_segment = LineString([geom.coords[i - 1], geom.coords[i]])

        # Check how the last pattern ended
        current_pattern_start = cumulated_length % (gap_length + dash_length)
        if current_pattern_start < 0.5 or current_pattern_start > gap_length + dash_length - 0.5:
            remaining_pattern_length = 0
        else:
            remaining_pattern_length = gap_length + dash_length - current_pattern_start

        # Check if the segment is longer than the remaining pattern
        if remaining_pattern_length <= line_segment.length:

            # Check if the segment starts with a dash
            if remaining_pattern_length - gap_length > 0:
                start_point = line_segment.coords[0]
                end_point = line_segment.interpolate(remaining_pattern_length - gap_length).coords[0]
                cv2.line(
                    img=img,
                    pt1=_lv03_to_pixel(start_point, geotransform),
                    pt2=_lv03_to_pixel(end_point, geotransform),
                    color=color,
                    thickness=thickness,
                    shift=0,
                    lineType=cv2.LINE_AA
                )

            rendered_length += remaining_pattern_length

            # The last pattern is now rendered. As the pattern starts now with a dash, we initialise the current
            # segment to paint the dashes.
            cumulated_length += remaining_pattern_length
            line_segment = LineString([line_segment.interpolate(remaining_pattern_length).coords[0], geom.coords[i]])

            # Paint the full patterns
            while rendered_length - cumulated_length + (dash_length + gap_length) < line_segment.length:
                cur_dash_length = dash_length + np.random.normal(0, PAINTING_STD['dash'])
                cur_gap_length = gap_length + np.random.normal(0, PAINTING_STD['dash'])
                start_point = line_segment.interpolate(rendered_length - cumulated_length).coords[0]
                end_point = line_segment.interpolate(rendered_length - cumulated_length + cur_dash_length).coords[0]
                cv2.line(
                    img=img,
                    pt1=_lv03_to_pixel(start_point, geotransform),
                    pt2=_lv03_to_pixel(end_point, geotransform),
                    color=color,
                    thickness=thickness,
                    shift=0,
                    lineType=cv2.LINE_AA
                )

                rendered_length += cur_dash_length + cur_gap_length

            # Paint the last pattern

            # Case 1: Segment ends within a gap
            if rendered_length - cumulated_length + dash_length < line_segment.length:
                cur_dash_length = dash_length + np.random.normal(0, PAINTING_STD['dash'])
                start_point = line_segment.interpolate(rendered_length - cumulated_length).coords[0]
                end_point = line_segment.interpolate(rendered_length - cumulated_length + cur_dash_length).coords[0]
                cv2.line(
                    img=img,
                    pt1=_lv03_to_pixel(start_point, geotransform),
                    pt2=_lv03_to_pixel(end_point, geotransform),
                    color=color,
                    thickness=thickness,
                    shift=0,
                    lineType=cv2.LINE_AA
                )

                rendered_length += LineString([start_point, line_segment.coords[-1]]).length

            # Case 2: Segment ends within a dash
            else:
                start_point = line_segment.interpolate(rendered_length - cumulated_length).coords[0]
                end_point = line_segment.coords[-1]
                cv2.line(
                    img=img,
                    pt1=_lv03_to_pixel(start_point, geotransform),
                    pt2=_lv03_to_pixel(end_point, geotransform),
                    color=color,
                    thickness=thickness,
                    shift=0,
                    lineType=cv2.LINE_AA
                )
                rendered_length += LineString([start_point, end_point]).length

        # Segment is shorter than the pattern
        else:
            # Check if the segment starts with a dash
            if (remaining_pattern_length - gap_length) > 0:
                remaining_dash_length = remaining_pattern_length - gap_length
                if line_segment.length < remaining_dash_length:
                    start_point = line_segment.coords[0]
                    end_point = line_segment.coords[-1]
                else:
                    start_point = line_segment.coords[0]
                    end_point = line_segment.interpolate(remaining_dash_length).coords[0]
                cv2.line(
                    img=img,
                    pt1=_lv03_to_pixel(start_point, geotransform),
                    pt2=_lv03_to_pixel(end_point, geotransform),
                    color=color,
                    thickness=thickness,
                    shift=0,
                    lineType=cv2.LINE_AA
                )

            rendered_length += line_segment.length

        cumulated_length += line_segment.length

    return img


def _draw_solid_line(img, geom, geotransform, color, thickness):
    """
    Draw a solid line on an image.

    Args:
    img: np.array, the image
    geom: LineString, the geometry
    geotransform: tuple, the GDAL geotransform parameters
    color: tuple, the color
    thickness: int, the thickness

    Returns:
    np.array, the image with the solid line
    """
    for i in range(1, len(geom.coords)):
        cv2.line(
            img=img,
            pt1=_lv03_to_pixel(geom.coords[i - 1], geotransform),
            pt2=_lv03_to_pixel(geom.coords[i], geotransform),
            color=color,
            thickness=thickness,
            shift=0,
            lineType=cv2.LINE_AA
        )
    return img


def _paint_lines(img, features, geotransform):
    """
    Function for overpainting the roads on an image e.g. Siegfried map.

    Args:
    img: np.array, the image (Siegfried map)
    features: list, the features
    geotransform: tuple, the geotransform parameters

    Returns:
    np.array, the image with the lines
    """
    UPSAMPLE_FACTOR = 2

    # Upsample image to have a better painting resolution
    img_upsampled = np.repeat(np.repeat(img, UPSAMPLE_FACTOR, axis=0), UPSAMPLE_FACTOR, axis=1)
    geotransform_upsampled = list(geotransform).copy()
    geotransform_upsampled[1] /= UPSAMPLE_FACTOR
    geotransform_upsampled[5] /= UPSAMPLE_FACTOR
    geotransform_upsampled = tuple(geotransform_upsampled)

    # Rendering order - three stages
    sorted_features = sorted(features, key=lambda x: x['properties']['road_cat'])

    with tqdm(total=len(sorted_features)*3, desc='Paint synthetic lines') as pbar:

        # First stage - paint the background
        for feature in sorted_features:
            try:
                geom = shape(feature['geometry'])
            except:
                print('Warning: Could not parse geometry')
                continue

            img_upsampled = _draw_solid_line(
                img_upsampled,
                geom,
                geotransform_upsampled,
                (255, 255, 255, 255),
                BUFFERSIZE_BACKGROUND_OVERPAINTING * UPSAMPLE_FACTOR
            )
            pbar.update(1)

        # Second stage - paint the black lines
        for feature in sorted_features:
            try:
                road_cat = int(feature['properties']['road_cat'])
                geom = shape(feature['geometry'])
                orientation = np.random.choice([1, -1])
            except:
                continue

            if road_cat == 5:
                left_geom = _get_offset_geom(geom, -1 * orientation * (
                            SYMBOLIZATION[road_cat]['offset'] + np.random.normal(0, PAINTING_STD['offset'])))
                right_geom = _get_offset_geom(geom, orientation * (
                            SYMBOLIZATION[road_cat]['offset'] + np.random.normal(0, PAINTING_STD['offset'])))
                thickness = int(np.round(SYMBOLIZATION[road_cat]['thickness_line_1'] * UPSAMPLE_FACTOR, decimals=0))
                img_upsampled = _draw_solid_line(
                    img_upsampled,
                    left_geom,
                    geotransform_upsampled,
                    (0, 0, 0, 255),
                    thickness
                )
                thickness = int(np.round(SYMBOLIZATION[road_cat]['thickness_line_2'] * UPSAMPLE_FACTOR, decimals=0))
                img_upsampled = _draw_solid_line(
                    img_upsampled,
                    right_geom,
                    geotransform_upsampled,
                    (0, 0, 0, 255),
                    thickness
                )

            elif road_cat == 4:
                left_geom = _get_offset_geom(geom, -1 * (
                            SYMBOLIZATION[road_cat]['offset'] + np.random.normal(0, PAINTING_STD['offset'])))
                right_geom = _get_offset_geom(geom, (
                            SYMBOLIZATION[road_cat]['offset'] + np.random.normal(0, PAINTING_STD['offset'])))
                thickness = int(np.round(SYMBOLIZATION[road_cat]['thickness'] * UPSAMPLE_FACTOR, decimals=0))
                _draw_solid_line(
                    img_upsampled,
                    left_geom,
                    geotransform_upsampled,
                    (0, 0, 0, 255),
                    thickness
                )
                thickness = int(np.round(SYMBOLIZATION[road_cat]['thickness'] * UPSAMPLE_FACTOR, decimals=0))
                _draw_solid_line(
                    img_upsampled,
                    right_geom,
                    geotransform_upsampled,
                    (0, 0, 0, 255),
                    thickness
                )

            elif road_cat == 3:
                dashed_geom = _get_offset_geom(geom, -1 * orientation * (SYMBOLIZATION[road_cat]['offset'] + np.random.normal(0, PAINTING_STD['offset'])))
                solid_geom = _get_offset_geom(geom, orientation * (SYMBOLIZATION[road_cat]['offset'] + np.random.normal(0, PAINTING_STD['offset'])))
                thickness = int(np.round((SYMBOLIZATION[road_cat]['thickness_line_dashed'] + np.random.normal(0,PAINTING_STD['thickness'])) * UPSAMPLE_FACTOR,decimals=0))
                if thickness < 2:
                    thickness = 2
                img_upsampled = _draw_dashed_line(
                    img_upsampled,
                    dashed_geom,
                    geotransform_upsampled,
                    (0, 0, 0, 255),
                    thickness,
                    SYMBOLIZATION[road_cat]['gap_length'] + np.random.normal(0, PAINTING_STD['dash']),
                    SYMBOLIZATION[road_cat]['dash_length'] + np.random.normal(0, PAINTING_STD['dash'])
                )
                thickness = int(np.round(SYMBOLIZATION[road_cat]['thickness_line_solid'] * UPSAMPLE_FACTOR, decimals=0))
                img_upsampled = _draw_solid_line(
                    img_upsampled,
                    solid_geom,
                    geotransform_upsampled,
                    (0, 0, 0, 255),
                    thickness
                )

            elif road_cat == 2:
                thickness = int(np.round(SYMBOLIZATION[road_cat]['thickness'] * UPSAMPLE_FACTOR, decimals=0))
                img_upsampled = _draw_solid_line(
                    img_upsampled,
                    geom,
                    geotransform_upsampled,
                    (0, 0, 0, 255),
                    thickness
                )

            elif road_cat == 1:
                thickness = int(np.round((SYMBOLIZATION[road_cat]['thickness'] + np.random.normal(0, PAINTING_STD[
                    'thickness'])) * UPSAMPLE_FACTOR, decimals=0))
                if thickness < 2:
                    thickness = 2
                img_upsampled = _draw_dashed_line(
                    img_upsampled,
                    geom,
                    geotransform_upsampled,
                    (0, 0, 0, 255),
                    thickness,
                    SYMBOLIZATION[road_cat]['gap_length'] + np.random.normal(0, PAINTING_STD['dash']),
                    SYMBOLIZATION[road_cat]['dash_length'] + np.random.normal(0, PAINTING_STD['dash'])
                )

            else:
                raise ValueError('Unknown road category {}!'.format(road_cat))

            pbar.update(1)

        # Third stage - paint the center lines
        for feature in sorted_features:
            road_cat = int(feature['properties']['road_cat'])
            geom = shape(feature['geometry'])

            if road_cat == 5:
                offset_pixels = SYMBOLIZATION[road_cat]['offset'] / geotransform_upsampled[1]
                thickness_contour_line_1 = SYMBOLIZATION[road_cat]['thickness_line_1'] * UPSAMPLE_FACTOR
                thickness_contour_line_2 = SYMBOLIZATION[road_cat]['thickness_line_2'] * UPSAMPLE_FACTOR
                thickness_center_line = int(
                    np.floor(2 * offset_pixels - 2 * max(thickness_contour_line_1, thickness_contour_line_2)))

            elif road_cat == 4:
                offset_pixels = SYMBOLIZATION[road_cat]['offset'] / geotransform_upsampled[1]
                thickness_contour_line = SYMBOLIZATION[road_cat]['thickness'] * UPSAMPLE_FACTOR
                thickness_center_line = int(np.floor(2 * offset_pixels - 2 * thickness_contour_line))

            elif road_cat == 3:
                offset_pixels = SYMBOLIZATION[road_cat]['offset'] / geotransform_upsampled[1]
                thickness_contour_line_1 = SYMBOLIZATION[road_cat]['thickness_line_dashed'] * UPSAMPLE_FACTOR
                thickness_contour_line_2 = SYMBOLIZATION[road_cat]['thickness_line_solid'] * UPSAMPLE_FACTOR
                thickness_center_line = int(
                    np.floor(2 * offset_pixels - 2 * max(thickness_contour_line_1, thickness_contour_line_2)))

            else:
                pbar.update(1)
                continue

            img_upsampled = _draw_solid_line(
                img_upsampled,
                geom,
                geotransform_upsampled,
                (255, 255, 255, 255),
                thickness_center_line
            )

            pbar.update(1)

    print(f'[{datetime.datetime.now()}] Fill with random Siegfried background')
    # Replace white pixels with random Siegfried background
    siegfried_background = _get_random_siegfried_background(shape=img_upsampled.shape)
    selection_matrix = 255 * np.ones_like(img_upsampled)
    white_indices = np.all(
        img_upsampled == selection_matrix,
        axis=2,
        keepdims=True
    ).squeeze(axis=2)
    img_upsampled[white_indices, :] = siegfried_background[white_indices, :]

    print(f'[{datetime.datetime.now()}] Downsample image')
    # Downsample image
    img = cv2.resize(img_upsampled, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    print(f'[{datetime.datetime.now()}] Create black label noise')
    # Replace white pixels with random Siegfried black
    siegfried_black = _get_random_siegfried_black(shape=img.shape)
    black_indices = np.all(
        img[:, :, :3] == 0,
        axis=2,
        keepdims=True
    ).squeeze(axis=2)
    img[black_indices, :] = siegfried_black[black_indices, :]

    return img


def _paint_road_labels(img, features, geotransform, segmentation_mode=False, buffersize=BUFFERSIZE_LABELS):
    """
    Function for painting the labels on an image.

    Args:
    img: np.array, the image
    features: list, the features
    geotransform: tuple, the geotransform parameters
    segmentation_mode: bool, whether the features are in segmentation mode
    buffersize: int, the buffer size

    Returns:
    np.array, the image with the labels
    """
    for feature in tqdm(features, desc='Painting road labels'):
        try:
            if not segmentation_mode:
                road_cat = int(feature['properties']['road_cat'])
            else:
                road_cat = 1
            geom = shape(feature['geometry'])
        except:
            print('Warning: Could not parse geometry')
            continue

        img = _draw_solid_line(
            img,
            geom,
            geotransform,
            road_cat,
            buffersize
        )

    return img


def _paint_breakpoint_labels(img, breakpoint_coords, features, geotransform):
    """
    Function for painting the breakpoint labels on an image

    Args:
    img: np.array, the image
    breakpoint_coords: list, the breakpoint coords as tupels in the crs associated with geotransform
    features: list, the features
    geotransform: tuple, the geotransform parameters

    Returns:
    np.array, the image with the breakpoint labels
    """
    possible_breakpoint_coords_lv03 = {}
    for feature in tqdm(features, desc='Finding intersection and breakpoint coords'):
        geom = shape(feature['geometry'])
        if geom.coords[0] in possible_breakpoint_coords_lv03.keys():
            possible_breakpoint_coords_lv03[geom.coords[0]] += 1
        else:
            possible_breakpoint_coords_lv03[geom.coords[0]] = 1
        if geom.coords[-1] in possible_breakpoint_coords_lv03.keys():
            possible_breakpoint_coords_lv03[geom.coords[-1]] += 1
        else:
            possible_breakpoint_coords_lv03[geom.coords[-1]] = 1

    breakpoint_coords_lv03 = [item for item, count in possible_breakpoint_coords_lv03.items() if count >= 2]
    # Add the breakpoint coords to the list (they should be included already)
    breakpoint_coords_lv03 += breakpoint_coords
    breakpoint_coords_lv03 = set(breakpoint_coords_lv03)

    for breakpoint_coord_lv03 in tqdm(breakpoint_coords_lv03, desc='Painting breakpoint labels'):
       breakpoint_coord_pixel = _lv03_to_pixel(breakpoint_coord_lv03, geotransform)

       color = 1
       thickness = -1 # negative thickness means that the circle will be filled
       cv2.circle(
           img=img,
           center=breakpoint_coord_pixel,
           radius=BREAKPOINT_BUFFER_RADIUS,
           color=color,
           thickness=thickness
       )

    return img


def _save_array_as_gtiff(img, path, ref_ds):
    """
    Save an array as a GeoTIFF file.

    Args:
    img: np.array, the image
    path: str, the path to save the image
    ref_ds: gdal.Dataset, the dataset with the georefenc information
    """
    driver = gdal.GetDriverByName("GTiff")
    if img.ndim == 3:
        img = img.transpose((2, 0, 1))
        ds_painted_map = driver.Create(
            str(path),
            img.shape[2],
            img.shape[1],
            img.shape[0],
            gdal.GDT_Byte,
            ['COMPRESS=LZW']
        )
        ds_painted_map.SetProjection(ref_ds.GetProjection())
        ds_painted_map.SetGeoTransform(ref_ds.GetGeoTransform())
        color_interpretations = [gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand, gdal.GCI_AlphaBand]
        for i in range(img.shape[0]):
            if i < 3:
                ds_painted_map.GetRasterBand(i + 1).WriteArray(img[i])
            else:
                ds_painted_map.GetRasterBand(i + 1).WriteArray(255 * np.ones_like(img[i]))
            ds_painted_map.GetRasterBand(i + 1).SetNoDataValue(-1)
            ds_painted_map.GetRasterBand(i + 1).SetColorInterpretation(color_interpretations[i])
        ds_painted_map.FlushCache()
        ds_painted_map = None
    else:
        ds_painted_map = driver.Create(
            str(path),
            img.shape[1],
            img.shape[0],
            1,
            gdal.GDT_Byte,
            ['COMPRESS=LZW']
        )
        ds_painted_map.SetProjection(ref_ds.GetProjection())
        ds_painted_map.SetGeoTransform(ref_ds.GetGeoTransform())
        ds_painted_map.GetRasterBand(1).SetNoDataValue(-1)
        ds_painted_map.GetRasterBand(1).WriteArray(img)
        ds_painted_map.FlushCache()
        ds_painted_map = None


def _save_tiles(tiles_array, path, sheet_number, suffix):
    """
    Save the tiles as tif files.

    Args:
    tiles_array: np.array, the tiles
    path: str, the path to save the tiles
    sheet_number: str, the sheet number
    suffix: str, the suffix
    """
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(tiles_array.shape[0]):
        cur_image = Image.fromarray(np.squeeze(tiles_array[i]))
        cur_image.save(
            '{}/{}_{:0>4}_{}.tif'.format(
                path,
                sheet_number,
                i, suffix
            )
        )


def main():
    """
    Main function for preprocessing the classification data.
    """
    # Settings
    GEOM_LENGTH_THRESHOLD_SINGLE_LINE = 20
    GEOM_LENGTH_THRESHOLD_WITHOUT_BREAKPOINT = 400

    utils.set_seed(const.SEED)

    for dataset in ('train', 'validate', 'test'):
        print('#' * 80)
        print(f'[{datetime.datetime.now()}] Start preprocessing {dataset}')
        print('#' * 80)

        # Declare paths
        path_siegfried_sheets = const.DATA_DIR.joinpath("classification").joinpath(dataset).joinpath("siegfried_sheets")
        path_road_geoms_predicted = const.DATA_DIR.joinpath("classification").joinpath(dataset).joinpath("road_geoms_predicted")
        path_road_geoms_synthetic_labeled = const.DATA_DIR.joinpath("classification").joinpath(dataset).joinpath("road_geoms_syn_labeled")
        path_road_geoms_ground_truth = const.DATA_DIR.joinpath("classification").joinpath(dataset).joinpath("road_geoms_ground_truth")
        path_mask_sheets = const.DATA_DIR.joinpath("classification").joinpath(dataset).joinpath('masks')
        path_temp = const.DATA_DIR.joinpath("classification").joinpath(dataset).joinpath("temp")

        # Create the output directory if it does not exist
        path_preprocessed_tiles = const.DATA_DIR.joinpath("classification").joinpath(dataset).joinpath("tiles")
        if os.path.exists(path_preprocessed_tiles):
            try:
                shutil.rmtree(path_preprocessed_tiles)
            except:
                raise ValueError('Could not remove the existing directory.')
        os.mkdir(path_preprocessed_tiles)

        if dataset in ('train', 'validate'):
            # Iterate over Siegfried Map sheets respectively associated ground truth road geometries
            for shapefile in os.listdir(path_road_geoms_ground_truth):
                if shapefile.endswith('.shp'):
                    print('\n')
                    print('-' * 80)
                    print(f'[{datetime.datetime.now()}] Processing sheet number {shapefile.split("_")[0]} ({dataset})')
                    print('-' * 80)

                    # Read current Siegfried Map sheet and its derived data
                    cur_sheet_number = shapefile.split('_')[0]
                    cur_sheet_path = path_siegfried_sheets.joinpath(f'{cur_sheet_number}_map.tif')
                    cur_sheet_ds = gdal.Open(str(cur_sheet_path))
                    cur_sheet_geotransform = cur_sheet_ds.GetGeoTransform()
                    cur_sheet_array = cur_sheet_ds.ReadAsArray()
                    cur_sheet_array_reordered = cur_sheet_array.transpose((1, 2, 0)).copy()

                    # Breakpoint coordinates
                    cur_sheet_breakpoint_coords = []

                    # Prepare synthetic sheets
                    cur_sheet_painted_array = cur_sheet_array_reordered
                    cur_sheet_class_labels_array = np.zeros(
                        (cur_sheet_array_reordered.shape[0], cur_sheet_array_reordered.shape[1]),
                        dtype=np.uint8
                    )
                    cur_sheet_breakpoint_labels_array = np.zeros(
                        (cur_sheet_array_reordered.shape[0], cur_sheet_array_reordered.shape[1]),
                        dtype=np.uint8
                    )
                    cur_sheet_hard_mask_array = np.zeros(
                        (cur_sheet_array_reordered.shape[0], cur_sheet_array_reordered.shape[1]),
                        dtype=np.uint8
                    )

                    # Open input shapefile and create output shapefile with synthetic road classes
                    with fiona.open(path_road_geoms_ground_truth.joinpath(shapefile)) as src:
                        print(f'[{datetime.datetime.now()}] Create synthetic road classes for sheet number {cur_sheet_number} ({dataset})')

                        # Add a new property to the schema
                        schema = src.schema.copy()
                        schema['properties']['road_cat'] = 'int'

                        # Open output shapefile and write synthetic road class line segments as lines
                        with fiona.open(
                                path_road_geoms_synthetic_labeled.joinpath(shapefile.replace('.shp', '_labeled.shp')),
                                'w', 'ESRI Shapefile', schema
                        ) as dst:

                            # Loop over the features and write the synthetic road class line segments
                            for feature in src:
                                # Read the properties and geometry
                                properties = dict(feature['properties']).copy()
                                geom = shape(feature['geometry'])
                                assert geom.geom_type == 'LineString', "Only LineString geometries are supported."

                                # Generate synthetic label
                                if GEOM_LENGTH_THRESHOLD_SINGLE_LINE < geom.length < GEOM_LENGTH_THRESHOLD_WITHOUT_BREAKPOINT:
                                    # Generate a synthetic road class
                                    road_cat = np.random.randint(1, 6)
                                    properties['road_cat'] = int(road_cat)
                                    # Write the synthetic road class line
                                    dst.write({
                                        'geometry': mapping(geom),
                                        'properties': properties
                                    })
                                elif geom.length >= GEOM_LENGTH_THRESHOLD_WITHOUT_BREAKPOINT:
                                    if len(geom.coords) == 2:
                                        geom = geom.segmentize(max_segment_length=np.ceil(geom.length / 2))
                                    # Generate two synthetic road classes with a random breakpoint
                                    if len(geom.coords) == 3:
                                        random_breakpoint_index = 1
                                    else:
                                        random_breakpoint_index = np.random.randint(1, len(geom.coords) - 2)

                                    # Split the geometry at the breakpoint
                                    # Write first part with a random road class
                                    geom_1 = LineString([geom.coords[i] for i in range(random_breakpoint_index + 1)])
                                    road_cat_1 = np.random.randint(1, 6)
                                    properties['road_cat'] = int(road_cat_1)
                                    dst.write({
                                        'geometry': mapping(geom_1),
                                        'properties': properties
                                    })

                                    # Write second part with another random road class
                                    geom_2 = LineString(
                                        [geom.coords[i] for i in range(random_breakpoint_index, len(geom.coords))])
                                    remaining_road_cats = list(range(1, 6))
                                    remaining_road_cats.remove(road_cat_1)
                                    road_cat_2 = np.random.choice(remaining_road_cats)
                                    properties['road_cat'] = int(road_cat_2)
                                    dst.write({
                                        'geometry': mapping(geom_2),
                                        'properties': properties
                                    })

                                    # Append the breakpoint coordinates
                                    cur_sheet_breakpoint_coords.append(geom.coords[random_breakpoint_index])
                                else:
                                    # Define a synthetic road class because of the short length only with a not
                                    # dashed single line symbol, thus 2
                                    road_cat = 2
                                    properties['road_cat'] = int(road_cat)
                                    # Write the synthetic road class line
                                    dst.write({
                                        'geometry': mapping(geom),
                                        'properties': properties
                                    })

                    # Paint the synthetic road classes as well as breakpoints and save the arrays
                    with fiona.open(
                            path_road_geoms_synthetic_labeled.joinpath(shapefile.replace('.shp', '_labeled.shp'))
                    ) as src:
                        print(f'[{datetime.datetime.now()}] Paint synthetic road classes for sheet number {cur_sheet_number} ({dataset})')

                        # Overpaint the roads with the synthetic roads
                        cur_sheet_painted_array = _paint_lines(
                            cur_sheet_painted_array,
                            src,
                            cur_sheet_geotransform
                        )
                        # Paint the corresponding labels
                        cur_sheet_class_labels_array = _paint_road_labels(
                            cur_sheet_class_labels_array,
                            src,
                            cur_sheet_geotransform,
                            segmentation_mode=False
                        )
                        # Paint the corresponding breakpoint labels
                        cur_sheet_breakpoint_labels_array = _paint_breakpoint_labels(
                            cur_sheet_breakpoint_labels_array,
                            cur_sheet_breakpoint_coords,
                            src,
                            cur_sheet_geotransform
                        )

                        # Paint and widen the hard mask
                        cur_sheet_hard_mask_array = _paint_road_labels(
                            cur_sheet_hard_mask_array,
                            src,
                            cur_sheet_geotransform,
                            segmentation_mode=True
                        )
                        cur_sheet_hard_mask_array = cv2.dilate(
                            cur_sheet_hard_mask_array,
                            kernel=np.ones((5, 5)),
                            iterations=2
                        )

                        # Save the arrays as GeoTIFFs
                        _save_array_as_gtiff(
                            cur_sheet_painted_array,
                            path_temp.joinpath(f'{cur_sheet_number}_painted.tif'),
                            cur_sheet_ds
                        )
                        _save_array_as_gtiff(
                            cur_sheet_class_labels_array,
                            path_temp.joinpath(f'{cur_sheet_number}_road_class_labels.tif'),
                            cur_sheet_ds
                        )
                        _save_array_as_gtiff(
                            cur_sheet_breakpoint_labels_array,
                            path_temp.joinpath(f'{cur_sheet_number}_breakpoint_labels.tif'),
                            cur_sheet_ds
                        )
                        _save_array_as_gtiff(
                            cur_sheet_hard_mask_array,
                            path_temp.joinpath(f'{cur_sheet_number}_hard_mask.tif'),
                            cur_sheet_ds
                        )

                        print(f'[{datetime.datetime.now()}] Generate tiles for sheet number {cur_sheet_number} ({dataset})')
                        # Generate tiles and append them to the arrays
                        tiles_array_painted_sheet = _generate_tiling(
                            path_temp.joinpath(f'{cur_sheet_number}_painted.tif'),
                            500,
                            mode='rgb'
                        )
                        # if there exists a mask file, load it and tile it
                        if os.path.exists(path_mask_sheets.joinpath(f'{cur_sheet_number}_mask.tif')):
                            tiles_array_mask_sheet = _generate_tiling(
                                path_mask_sheets.joinpath(f'{cur_sheet_number}_mask.tif'),
                                500,
                                mode='grayscale'
                            )
                        else:
                            tiles_array_mask_sheet = None
                        tiles_array_road_class_label_sheet = _generate_tiling(
                            path_temp.joinpath(f'{cur_sheet_number}_road_class_labels.tif'),
                            500,
                            mode='grayscale'
                        )
                        tiles_array_breakpoint_label_sheet = _generate_tiling(
                            path_temp.joinpath(f'{cur_sheet_number}_breakpoint_labels.tif'),
                            500,
                            mode='grayscale'
                        )
                        tiles_array_hard_mask_sheet = _generate_tiling(
                            path_temp.joinpath(f'{cur_sheet_number}_hard_mask.tif'),
                            500,
                            mode='grayscale'
                        )

                        print(f'[{datetime.datetime.now()}] Save tiles for sheet number {cur_sheet_number} ({dataset})')
                        # Save the tiles
                        _save_tiles(
                            tiles_array_painted_sheet,
                            path_preprocessed_tiles.joinpath('painted'),
                            cur_sheet_number,
                            'painted'
                        )
                        _save_tiles(
                            tiles_array_road_class_label_sheet,
                            path_preprocessed_tiles.joinpath('road_class_labels'),
                            cur_sheet_number,
                            'road_class_labels'
                        )
                        _save_tiles(
                            tiles_array_breakpoint_label_sheet,
                            path_preprocessed_tiles.joinpath('breakpoint_labels'),
                            cur_sheet_number,
                            'breakpoint_labels'
                        )
                        _save_tiles(
                            tiles_array_hard_mask_sheet,
                            path_preprocessed_tiles.joinpath('hard_masks'),
                            cur_sheet_number,
                            'hard_mask'
                        )
                        if tiles_array_mask_sheet is not None:
                            _save_tiles(
                                tiles_array_mask_sheet,
                                path_preprocessed_tiles.joinpath('masks'),
                                cur_sheet_number,
                                'mask'
                            )

        else:  # test
            print(f'[{datetime.datetime.now()}] Processing {dataset}')
            path_breakpoint_geoms_ground_truth = const.DATA_DIR.joinpath("classification").joinpath(dataset).joinpath(
                "breakpoint_geoms_ground_truth")

            # Iterate over Siegfried Map sheets respectively associated predicted road geometries
            for shapefile in os.listdir(path_road_geoms_predicted):
                if shapefile.endswith('.shp'):
                    print(f'[{datetime.datetime.now()}] Processing sheet number {shapefile.split("_")[0]} ({dataset})')
                    # Read current Siegfried Map sheet and its derived data
                    cur_sheet_number = shapefile.split('_')[0]
                    cur_sheet_path = path_siegfried_sheets.joinpath(f'{cur_sheet_number}_map.tif')
                    cur_sheet_ds = gdal.Open(str(cur_sheet_path))
                    cur_sheet_geotransform = cur_sheet_ds.GetGeoTransform()
                    cur_sheet_array = cur_sheet_ds.ReadAsArray()
                    cur_sheet_array_reordered = cur_sheet_array.transpose((1, 2, 0)).copy()

                    # Prepare sheets with real testdata in the same way as the synthetic data
                    cur_sheet_class_labels_array = np.zeros(
                        (cur_sheet_array_reordered.shape[0], cur_sheet_array_reordered.shape[1]),
                        dtype=np.uint8
                    )
                    cur_sheet_breakpoint_labels_array = np.zeros(
                        (cur_sheet_array_reordered.shape[0], cur_sheet_array_reordered.shape[1]),
                        dtype=np.uint8
                    )
                    cur_sheet_hard_mask_array = np.zeros(
                        (cur_sheet_array_reordered.shape[0], cur_sheet_array_reordered.shape[1]),
                        dtype=np.uint8
                    )

                    # Open predicted road geometries and create hard mask
                    with fiona.open(path_road_geoms_predicted.joinpath(shapefile)) as src:
                        print(f'[{datetime.datetime.now()}] Create hard mask for sheet number {cur_sheet_number} ({dataset})')

                        # Paint and widen the hard mask
                        cur_sheet_hard_mask_array = _paint_road_labels(
                            cur_sheet_hard_mask_array,
                            src,
                            cur_sheet_geotransform,
                            segmentation_mode=True
                        )
                        cur_sheet_hard_mask_array = cv2.dilate(
                            cur_sheet_hard_mask_array,
                            kernel=np.ones((5, 5)),
                            iterations=2
                        )

                        # Save the array as GeoTIFFs
                        _save_array_as_gtiff(
                            cur_sheet_hard_mask_array,
                            path_temp.joinpath(f'{cur_sheet_number}_hard_mask.tif'),
                            cur_sheet_ds
                        )

                        print(f'[{datetime.datetime.now()}] Generate hard mask tiles for sheet number {cur_sheet_number} ({dataset})')
                        # Generate tiles and append them to the arrays
                        tiles_array_hard_mask_sheet = _generate_tiling(
                            path_temp.joinpath(f'{cur_sheet_number}_hard_mask.tif'),
                            500,
                            mode='grayscale'
                        )

                        print(f'[{datetime.datetime.now()}] Save hard mask tiles for sheet number {cur_sheet_number} ({dataset})')
                        # Save the tiles
                        _save_tiles(
                            tiles_array_hard_mask_sheet,
                            path_preprocessed_tiles.joinpath('hard_masks'),
                            cur_sheet_number,
                            'hard_mask'
                        )

                    cur_sheet_breakpoint_coords = []
                    # Open ground truth breakpoint geometries and create breakpoint labels
                    if os.path.exists(
                            path_breakpoint_geoms_ground_truth.joinpath(f'{cur_sheet_number}_breakpoints.shp')):
                        with fiona.open(
                                path_breakpoint_geoms_ground_truth.joinpath(f'{cur_sheet_number}_breakpoints.shp')
                        ) as src:
                            print(f'[{datetime.datetime.now()}] Create breakpoint labels for sheet number {cur_sheet_number} ({dataset})')
                            for feature in src:
                                geom = shape(feature['geometry'])
                                assert geom.geom_type == 'Point', "Only Point geometries are supported."
                                cur_sheet_breakpoint_coords.append(tuple(geom.coords[0]))

                    # Open ground truth road geometries and create road class labels
                    with fiona.open(path_road_geoms_ground_truth.joinpath(f'{cur_sheet_number}_roads.shp')) as src:
                        print(f'[{datetime.datetime.now()}] Create road class labels for sheet number {cur_sheet_number} ({dataset})')

                        # ------------------------------------------
                        # Only for quality control used: Paint ground truth and save it in the temp
                        cur_sheet_painted_array = _paint_lines(
                            cur_sheet_array_reordered.copy(),
                            src,
                            cur_sheet_geotransform
                        )
                        _save_array_as_gtiff(
                            cur_sheet_painted_array,
                            path_temp.joinpath(f'{cur_sheet_number}_painted.tif'),
                            cur_sheet_ds
                        )
                        # ------------------------------------------

                        # Paint the ground truth labels
                        cur_sheet_class_labels_array = _paint_road_labels(
                            cur_sheet_class_labels_array,
                            src,
                            cur_sheet_geotransform
                        )
                        # Paint the corresponding breakpoint labels
                        cur_sheet_breakpoint_labels_array = _paint_breakpoint_labels(
                            cur_sheet_breakpoint_labels_array,
                            cur_sheet_breakpoint_coords,
                            src,
                            cur_sheet_geotransform
                        )

                        # Save the arrays as GeoTIFFs
                        _save_array_as_gtiff(
                            cur_sheet_class_labels_array,
                            path_temp.joinpath(f'{cur_sheet_number}_road_class_labels.tif'),
                            cur_sheet_ds
                        )
                        _save_array_as_gtiff(
                            cur_sheet_breakpoint_labels_array,
                            path_temp.joinpath(f'{cur_sheet_number}_breakpoint_labels.tif'),
                            cur_sheet_ds
                        )

                        print(f'[{datetime.datetime.now()}] Generate tiles for sheet number {cur_sheet_number} ({dataset})')
                        # Generate tiles and append them to the arrays
                        tiles_array_siegfried_masked_sheet = _generate_tiling(
                            path_siegfried_sheets.joinpath(f'{cur_sheet_number}_map.tif'),
                            500,
                            mode='rgb'
                        )
                        tiles_array_road_class_label_sheet = _generate_tiling(
                            path_temp.joinpath(f'{cur_sheet_number}_road_class_labels.tif'),
                            500,
                            mode='grayscale'
                        )
                        tiles_array_breakpoint_label_sheet = _generate_tiling(
                            path_temp.joinpath(f'{cur_sheet_number}_breakpoint_labels.tif'),
                            500,
                            mode='grayscale'
                        )

                        print(f'[{datetime.datetime.now()}] Save tiles for sheet number {cur_sheet_number} ({dataset})')
                        # Save the tiles
                        _save_tiles(
                            tiles_array_siegfried_masked_sheet,
                            path_preprocessed_tiles.joinpath('map'),
                            cur_sheet_number,
                            'map'
                        )
                        _save_tiles(
                            tiles_array_road_class_label_sheet,
                            path_preprocessed_tiles.joinpath('road_class_labels'),
                            cur_sheet_number,
                            'road_class_labels'
                        )
                        _save_tiles(
                            tiles_array_breakpoint_label_sheet,
                            path_preprocessed_tiles.joinpath('breakpoint_labels'),
                            cur_sheet_number,
                            'breakpoint_labels'
                        )
    print(f'[{datetime.datetime.now()}] Done')


if __name__ == '__main__':
    main()
