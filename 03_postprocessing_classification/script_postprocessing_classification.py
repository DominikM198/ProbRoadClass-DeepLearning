import os
import datetime
import cv2
from PIL import Image
import numpy as np
from osgeo import gdal
import fiona
from shapely.geometry import shape, mapping, Point, LineString
import rasterstats
from tqdm import tqdm
import warnings
gdal.DontUseExceptions()
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Constants
SHEET_NUMBERS = ['199', '385']
SIGFRIED_FILENAME_PREFIX = 'rgb_TA_'
OUTPUT_FILENAME_PREFIX = 'minline_80m_seg_10m'
W_SIZE = 500
BUFFERSIZES_METER = [6]
EPSILON = 0.000001

ROAD_CAT_COLORS = {
    1: '#4e9ee9',
    2: '#009031',
    3: '#be189a',
    4: '#ff9901',
    5: '#cc0003'
}

BREAKPOINT_TRACING_DISCRETIZATION = 10 # meters
BREAKPOINT_TRACING_CROP_DISTANCE = 20 # meters
BREAKPOINT_TRACING_MINIMUM_LINE_LENGTH = 120 # meters
BREAKPOINT_TRACING_PLOT_FLAG = False

path_input_folder = './input'
path_temp_folder = './temp'
path_output_folder = './output'

if not os.path.exists(path_temp_folder):
    os.mkdir(path_temp_folder)
if not os.path.exists(path_output_folder):
    os.mkdir(path_output_folder)


def _reconstruct_from_patches(patches_images, patch_size, step_size, image_size_2d, image_dtype):
    """
    Adjust to take patch images directly.

    Arg:
    patches_images: np.array, patch images
    patch_size: int, the size of the tiles
    step_size: int, should be patch_size//2
    image_size_2d: tuple, size of the original image
    image_dtype: np.dtype, data type of the target image

    Returns:
    np.array, the reconstructed image
    """
    i_h, i_w = np.array(image_size_2d[:2]) + (patch_size, patch_size)
    p_h = p_w = patch_size
    if len(patches_images.shape) == 4:
        img = np.zeros((i_h+p_h//2, i_w+p_w//2, 4), dtype=image_dtype)
    else:
        img = np.zeros((i_h+p_h//2, i_w+p_w//2), dtype=image_dtype)

    numrows = (i_h)//step_size-1
    numcols = (i_w)//step_size-1
    expected_patches = numrows * numcols
    if len(patches_images) != expected_patches:
        raise ValueError(f"Expected {expected_patches} patches, got {len(patches_images)}")

    patch_offset = step_size//2
    patch_inner = p_h-step_size
    for row in range(numrows):
        for col in range(numcols):
            tt = patches_images[row*numcols+col]
            tt_roi = tt[patch_offset:-patch_offset,patch_offset:-patch_offset]
            img[row*step_size:row*step_size+patch_inner,
                col*step_size:col*step_size+patch_inner] = tt_roi # +1??
    return img[step_size//2:-(patch_size+step_size//2),step_size//2:-(patch_size+step_size//2),...]


def _interpolate_distance(dist_segment, y1_segment, y2_segment):
    """
    Interpolate the distance between two segments.

    Args:
    dist_segment: float
    y1_segment: float
    y2_segment: float

    Returns:
    float, the interpolated distance
    """
    # Find the intersection between these two segments
    denom = (dist_segment[1] - dist_segment[0]) * (y2_segment[1] - y2_segment[0]) - (y1_segment[1] - y1_segment[0]) * (dist_segment[1] - dist_segment[0])
    assert denom != 0, 'Segments are parallel'

    ua = ((dist_segment[1] - dist_segment[0]) * (y1_segment[0] - y2_segment[0])) / denom
    ub = ((dist_segment[1] - dist_segment[0]) * (y1_segment[0] - y2_segment[0])) / denom

    if not (0 <= ua <= 1) and (0 <= ub <= 1):
        print('Intersection is not within the segments')
    dist_intersect = dist_segment[0] + ua * (dist_segment[1] - dist_segment[0])
    return dist_intersect


def _split_line_at_distance(line, distance):
    """
    Split a line at a given distance.

    Args:
    line: LineString, the line to split
    distance: float, the distance to split the line at

    Returns:
    LineString, LineString, the two lines
    """
    if distance < 0:
        raise ValueError('Distance must be greater than zero')
    elif distance == 0:
        return None, line
    elif distance == line.length:
        return line, None
    elif line is None:
        raise ValueError('Line geometry must not be None')

    cumulated_distance = 0
    vertex_idx = 0
    while cumulated_distance < distance:
        vertex_idx += 1
        cumulated_distance += LineString([line.coords[vertex_idx - 1], line.coords[vertex_idx]]).length
    line_before_breakpnt = LineString([line.coords[i] for i in range(vertex_idx)] + [line.interpolate(distance).coords[0]])
    line_after_breakpnt = LineString([line.interpolate(distance).coords[0]] + [line.coords[i] for i in range(vertex_idx, len(line.coords))])
    return line_before_breakpnt, line_after_breakpnt


# Iterate over all sheets
for sheet_number in SHEET_NUMBERS:
    print('----------------------------------------------------------------------------------------------------')
    print(f'[{datetime.datetime.now()}] Processing {sheet_number}')
    print('----------------------------------------------------------------------------------------------------')

    print(f'[{datetime.datetime.now()}] Map Stiching: Load the classification tiles')
    # Open the Siegfried map
    ds_siegfried = gdal.Open(f'{path_input_folder}/{SIGFRIED_FILENAME_PREFIX}{sheet_number}.tif')
    array_siegfried_map = ds_siegfried.GetRasterBand(1).ReadAsArray()

    # Iterate over all road categories
    for road_cat in range(1, 6):
        tiles_array = []
        # Read all patches for the current road category
        for file_in in ['{}/{}_{:0>4}_class{}.tif'.format(path_input_folder, sheet_number, i, road_cat) for i in range(580)]:
            if not os.path.exists(file_in):
                tiles_array.append(-1 * np.ones((500, 500)))
            else:
                tiles_array.append(np.array(Image.open(file_in)))

        # Reconstruct the image from the patches
        tiles_array = np.array(tiles_array)
        pad_px = W_SIZE // 2
        img_reconstructed = _reconstruct_from_patches(tiles_array, W_SIZE, pad_px, array_siegfried_map.shape[:2], np.float32)

        # Write the reconstructed image to a new raster image
        driver = gdal.GetDriverByName("GTiff")
        ds_out_classification = driver.Create(
            f'{path_temp_folder}/{sheet_number}_classification_class_{road_cat}.tif',
            img_reconstructed.shape[1],
            img_reconstructed.shape[0],
            1,
            gdal.GDT_Float32,
            ['COMPRESS=LZW']
        )
        ds_out_classification.SetProjection(ds_siegfried.GetProjection())
        ds_out_classification.SetGeoTransform(ds_siegfried.GetGeoTransform())
        ds_out_classification.GetRasterBand(1).SetNoDataValue(-1)
        ds_out_classification.GetRasterBand(1).WriteArray(img_reconstructed)
        ds_out_classification.FlushCache()
        ds_out_classification = None

    # Release resources
    ds_siegfried.FlushCache()
    ds_siegfried = None

    # Iterate over all buffer sizes
    for BUFFERSIZE_METER in BUFFERSIZES_METER:
        print(f'[{datetime.datetime.now()}] Breakpoint Tracing: Buffer Size --> {BUFFERSIZE_METER} m')
        shapefile_name = f'{sheet_number}_road_geoms.shp'

        # Open the input shapefile
        with fiona.open(f'{path_input_folder}/{shapefile_name}') as src:
            schema = src.schema.copy()
            schema['properties']['parentId'] = 'str'

            # Open the output shapefile
            with fiona.open('{path_temp_folder}/{sheet_number}_road_geoms_breakpoints_{buffer}.shp'.format(
                    path_temp_folder=path_temp_folder,
                    sheet_number=str(sheet_number),
                    buffer=str(BUFFERSIZE_METER).replace('.', '-')
            ), 'w', 'ESRI Shapefile', schema) as dst:
                
                for linestring in tqdm(src, desc='Breakpoint Tracing'):
                    properties = dict(linestring['properties'])
                    properties['parentId'] = str(linestring.id)
                    geom = shape(linestring['geometry'])
                    segmented_geom = geom.segmentize(max_segment_length=BREAKPOINT_TRACING_DISCRETIZATION)
                    cur_distances = []
                    cur_means = {1: [], 2: [], 3: [], 4: [], 5: []}
                    cur_stds = {1: [], 2: [], 3: [], 4: [], 5: []}
                    cur_breakpoint_distances = []

                    for i in range(len(segmented_geom.coords) - 1):
                        cur_center_point = Point(
                            0.5 * (segmented_geom.coords[i][0] + segmented_geom.coords[i + 1][0]),
                            0.5 * (segmented_geom.coords[i][1] + segmented_geom.coords[i + 1][1])
                        )
                        cur_distances.append(geom.project(cur_center_point))
                        segment_geom = LineString([segmented_geom.coords[i], segmented_geom.coords[i + 1]])
                        buffered_segment_geom = segment_geom.buffer(BUFFERSIZE_METER, cap_style='flat')
                        for road_cat in range(1, 6):
                            res = rasterstats.zonal_stats(
                                buffered_segment_geom,
                                f'{path_temp_folder}/{sheet_number}_classification_class_{road_cat}.tif',
                                stats=['mean', 'std']
                            )
                            if res:
                                cur_means[road_cat].append(res[0]['mean'])
                                cur_stds[road_cat].append(res[0]['std'])
                            else:
                                cur_means[road_cat].append(0.0)
                                cur_stds[road_cat].append(0.0)
                                print(f'[{datetime.datetime.now()}] Breakpoint Tracing: ERROR')

                    # Crop the data at the beginning and the end
                    crop_ids = int(BREAKPOINT_TRACING_CROP_DISTANCE // BREAKPOINT_TRACING_DISCRETIZATION)
                    if 2 * crop_ids + 1 < len(cur_distances):
                        cur_distances = cur_distances[crop_ids:-crop_ids]
                        for road_cat in range(1, 6):
                            cur_means[road_cat] = cur_means[road_cat][crop_ids:-crop_ids]
                            cur_stds[road_cat] = cur_stds[road_cat][crop_ids:-crop_ids]
                    else:
                        dst.write({
                            'geometry': mapping(geom),
                            'properties': properties,
                            'id': linestring.id
                        })
                        continue

                    # Find the breakpoints
                    cur_means_array = np.array([cur_means[road_cat] for road_cat in range(1, 6)])
                    cur_road_cats = np.argmax(cur_means_array, axis=0) + 1

                    breakpoint_candidates = []
                    for i in range(1, len(cur_road_cats)):
                        if cur_road_cats[i] != cur_road_cats[i - 1]:
                            interpolated_distance = _interpolate_distance(
                                [cur_distances[i - 1], cur_distances[i]],
                                [cur_means[cur_road_cats[i - 1]][i - 1], cur_means[cur_road_cats[i - 1]][i]],
                                [cur_means[cur_road_cats[i]][i - 1], cur_means[cur_road_cats[i]][i]]
                            )
                            breakpoint_candidates.append({
                                'dist': interpolated_distance,
                                'road_cat_before': cur_road_cats[i - 1],
                                'road_cat_after': cur_road_cats[i]}
                            )

                    # Create segments from all breakpoints
                    if len(breakpoint_candidates) > 0:
                        # Add first segment
                        segments = [{
                            'dist_from': 0.0,
                            'dist_to': breakpoint_candidates[0]['dist'],
                            'length_segment': breakpoint_candidates[0]['dist'],
                            'length_segment_below_threshold': breakpoint_candidates[0]['dist'] < BREAKPOINT_TRACING_MINIMUM_LINE_LENGTH,
                            'road_cat': breakpoint_candidates[0]['road_cat_before']
                        }]
                        # Add the middle segments
                        for i in range(len(breakpoint_candidates) - 1):
                            segments.append({
                                'dist_from': breakpoint_candidates[i]['dist'],
                                'dist_to': breakpoint_candidates[i + 1]['dist'],
                                'length_segment': breakpoint_candidates[i + 1]['dist'] - breakpoint_candidates[i]['dist'],
                                'length_segment_below_threshold': breakpoint_candidates[i + 1]['dist'] - breakpoint_candidates[i]['dist'] < BREAKPOINT_TRACING_MINIMUM_LINE_LENGTH,
                                'road_cat': breakpoint_candidates[i]['road_cat_after']
                            })
                        # Add last segment
                        segments.append({
                            'dist_from': breakpoint_candidates[-1]['dist'],
                            'dist_to': geom.length,
                            'length_segment': geom.length - breakpoint_candidates[-1]['dist'],
                            'length_segment_below_threshold': geom.length - breakpoint_candidates[-1]['dist'] < BREAKPOINT_TRACING_MINIMUM_LINE_LENGTH,
                            'road_cat': breakpoint_candidates[-1]['road_cat_after']
                        })

                        # Filter to short segments
                        while any([segment['length_segment_below_threshold'] for segment in segments]):

                            # Check if the line is too short
                            if geom.length < BREAKPOINT_TRACING_MINIMUM_LINE_LENGTH:
                                segments = [{
                                    'dist_from': 0,
                                    'dist_to': geom.length,
                                    'length_segment': geom.length,
                                    'length_segment_below_threshold': geom.length < BREAKPOINT_TRACING_MINIMUM_LINE_LENGTH,
                                    'road_cat': segments[0]['road_cat']
                                }]
                                break

                            # Algorithm to merge segments if there are more than 2 segments in the list
                            if len(segments) >= 3:
                                if segments[0]['length_segment_below_threshold']:
                                    segments[1]['dist_from'] = segments[0]['dist_from']
                                    segments[1]['length_segment'] = segments[1]['dist_to'] - segments[1]['dist_from']
                                    segments[1]['length_segment_below_threshold'] = segments[1]['length_segment'] < BREAKPOINT_TRACING_MINIMUM_LINE_LENGTH
                                    segments = segments[1:]

                                if segments[-1]['length_segment_below_threshold']:
                                    segments[-2]['dist_to'] = segments[-1]['dist_to']
                                    segments[-2]['length_segment'] = segments[-2]['dist_to'] - segments[-2]['dist_from']
                                    segments[-2]['length_segment_below_threshold'] = segments[-2]['length_segment'] < BREAKPOINT_TRACING_MINIMUM_LINE_LENGTH
                                    segments = segments[:-1]

                                # Break the algorithm if there are only two segments left (second part of the algorithm
                                # comes in the next else if block)
                                if len(segments) < 3:
                                    continue

                                for i in reversed(range(1, len(segments) - 1)):
                                    if i == len(segments) - 1:
                                        continue

                                    if segments[i]['length_segment_below_threshold']:
                                        if segments[i - 1]['road_cat'] == segments[i + 1]['road_cat']:
                                            merged_segment = {
                                                'dist_from': segments[i - 1]['dist_from'],
                                                'dist_to': segments[i + 1]['dist_to'],
                                                'length_segment': segments[i + 1]['dist_to'] - segments[i - 1]['dist_from'],
                                                'length_segment_below_threshold': segments[i + 1]['dist_to'] - segments[i - 1]['dist_from'] < BREAKPOINT_TRACING_MINIMUM_LINE_LENGTH,
                                                'road_cat': segments[i - 1]['road_cat']
                                            }
                                            segments = segments[:i - 1] + [merged_segment] + segments[i + 2:]
                                        else:
                                            segments[i - 1]['dist_to'] += 0.5 * segments[i]['length_segment']
                                            segments[i - 1]['length_segment'] = segments[i - 1]['dist_to'] - segments[i - 1]['dist_from']
                                            segments[i - 1]['length_segment_below_threshold'] = segments[i - 1]['length_segment'] < BREAKPOINT_TRACING_MINIMUM_LINE_LENGTH
                                            segments[i + 1]['dist_from'] -= 0.5 * segments[i]['length_segment']
                                            segments[i + 1]['length_segment'] = segments[i + 1]['dist_to'] - segments[i + 1]['dist_from']
                                            segments[i + 1]['length_segment_below_threshold'] = segments[i + 1]['length_segment'] < BREAKPOINT_TRACING_MINIMUM_LINE_LENGTH
                                            segments = segments[:i] + segments[i + 1:]

                            # Handle the case when there are only two segments left
                            elif len(segments) == 2:
                                if segments[0]['length_segment_below_threshold']:
                                    segments[1]['dist_from'] = segments[0]['dist_from']
                                    segments[1]['length_segment'] = segments[1]['dist_to'] - segments[1]['dist_from']
                                    segments[1]['length_segment_below_threshold'] = segments[1]['length_segment'] < BREAKPOINT_TRACING_MINIMUM_LINE_LENGTH
                                    segments = segments[1:]

                                elif segments[1]['length_segment_below_threshold']:
                                    segments[0]['dist_to'] = segments[1]['dist_to']
                                    segments[0]['length_segment'] = segments[0]['dist_to'] - segments[0]['dist_from']
                                    segments[0]['length_segment_below_threshold'] = segments[0]['length_segment'] < BREAKPOINT_TRACING_MINIMUM_LINE_LENGTH
                                    segments = segments[:-1]

                            # Break the loop if there is only one segment / the whole line left
                            else:
                                break

                        # Read out the breakpoint distances i.e. the distance to split the line from the list of
                        # filtered segments
                        cur_breakpoint_distances = sorted([segments[i]['dist_to'] for i in range(len(segments) - 1)])

                        # Write the segments to the temporary output shapefile
                        if len(cur_breakpoint_distances) > 0:
                            cur_breakpoint_distances_relative = [cur_breakpoint_distances[0]] + [cur_breakpoint_distances[i + 1] - cur_breakpoint_distances[i] for i in range(len(cur_breakpoint_distances) - 1)]

                            # Split the line at the breakpoints and write the segments to the output shapefile
                            for cur_breakpoint_distance_relative in cur_breakpoint_distances_relative:
                                if cur_breakpoint_distance_relative <= EPSILON:
                                    continue
                                line_before_breakpnt, line_after_breakpnt = _split_line_at_distance(geom, cur_breakpoint_distance_relative)
                                dst.write({
                                    'geometry': mapping(line_before_breakpnt),
                                    'properties': properties,
                                    'id': linestring.id
                                })
                                geom = line_after_breakpnt
                            dst.write({
                                'geometry': mapping(geom),
                                'properties': properties,
                                'id': linestring.id
                            })

                        else:
                            dst.write({
                                'geometry': mapping(geom),
                                'properties': properties,
                                'id': linestring.id
                            })

                    else:
                        dst.write({
                            'geometry': mapping(geom),
                            'properties': properties,
                            'id': linestring.id
                        })

                    if BREAKPOINT_TRACING_PLOT_FLAG:
                        import matplotlib.pyplot as plt
                        if not os.path.exists('{path_temp_folder}/plots'.format(path_temp_folder=path_temp_folder)):
                            os.mkdir('{path_temp_folder}/plots'.format(path_temp_folder=path_temp_folder))

                        fig, ax = plt.subplots(
                            nrows=1, ncols=1,
                            figsize=(7, 4)
                        )
                        for i in range(1, 6):
                            ax.plot(
                                np.array(cur_distances),
                                np.array(cur_means[i]),
                                label=f'Class {i}',
                                color=ROAD_CAT_COLORS[i]
                            )
                            plt.fill_between(
                                np.array(cur_distances),
                                np.array(cur_means[i]) - np.array(cur_stds[i]),
                                np.array(cur_means[i]) + np.array(cur_stds[i]),
                                alpha=0.2,
                                color=ROAD_CAT_COLORS[i]
                            )
                        ax.set_ylabel('Mean Probability')
                        ax.set_xlabel('Distance (m)')
                        if len(cur_breakpoint_distances) < len(breakpoint_candidates):
                            for idx, breakpoint_candidate in enumerate(breakpoint_candidates):
                                if idx == 0:
                                    ax.plot(
                                        [breakpoint_candidate['dist'], breakpoint_candidate['dist']],
                                        [0, 1],
                                        '--', color='gray', linewidth=1.0, label='Split (all)'
                                    )
                                else:
                                    ax.plot(
                                        [breakpoint_candidate['dist'], breakpoint_candidate['dist']],
                                        [0, 1],
                                        '--', color='gray', linewidth=1.0
                                    )
                        for idx, cur_breakpoint_distance in enumerate(cur_breakpoint_distances):
                            if idx == 0:
                                ax.plot(
                                    [cur_breakpoint_distance, cur_breakpoint_distance],
                                    [0, 1],
                                    '--', color='black', linewidth=2.0, label='Split'
                                )
                            else:
                                ax.plot(
                                    [cur_breakpoint_distance, cur_breakpoint_distance],
                                    [0, 1],
                                    '--', color='black', linewidth=2.0
                                )
                        ax.legend(loc='lower right')
                        plt.savefig('{path_temp_folder}/plots/{sheet_number}_breakpoint_tracing_{id}.png'.format(
                            path_temp_folder=path_temp_folder,
                            sheet_number=str(sheet_number),
                            id=str(linestring.id)
                        ))
                        plt.close()

        # In the last step we want to compute the zonal statistics for all line geometries after splitting them and
        # assign the road class with the highest mean probability to the line
        print(f'[{datetime.datetime.now()}] Zonal Statistics: Buffer Size --> {BUFFERSIZE_METER} m')
        
        # Open the input shapefile
        with fiona.open('{path_temp_folder}/{sheet_number}_road_geoms_breakpoints_{buffer}.shp'.format(
                    path_temp_folder=path_temp_folder,
                    sheet_number=str(sheet_number),
                    buffer=str(BUFFERSIZE_METER).replace('.', '-')
            )
        ) as src:
            # Copy the schema of the input shapefile and add a new property 'road_cat' to it
            schema = src.schema.copy()
            schema['properties']['road_cat'] = 'int'

            # Open the output shapefile
            if not os.path.exists('{}/{}_{}m'.format(
                path_output_folder,
                OUTPUT_FILENAME_PREFIX,
                str(BUFFERSIZE_METER).replace('.', '-')
            )):
                os.mkdir('{}/{}_{}m'.format(
                    path_output_folder,
                    OUTPUT_FILENAME_PREFIX,
                    str(BUFFERSIZE_METER).replace('.', '-'),
                ))
            with fiona.open('{path_output_folder}/{output_file_prefix}_{buffer}m/{sheet_number}_road_geoms_{output_file_prefix}_{buffer}m.shp'.format(
                    path_output_folder=path_output_folder,
                    output_file_prefix=OUTPUT_FILENAME_PREFIX,
                    sheet_number=str(sheet_number),
                    buffer=str(BUFFERSIZE_METER).replace('.', '-')
            ), 'w', 'ESRI Shapefile', schema) as dst:

                for feature in tqdm(src, desc='Zonal Statistics'):
                    # Extract the geometry of the feature
                    geom = shape(feature['geometry'])

                    # Compute the zonal statistics for the entire geometry
                    buffered_geom = geom.buffer(BUFFERSIZE_METER, cap_style='flat')
                    line_mean = 0.0
                    line_road_cat = -1
                    for road_cat in range(1, 6):
                        res = rasterstats.zonal_stats(
                            buffered_geom,
                            f'{path_temp_folder}/{sheet_number}_classification_class_{road_cat}.tif',
                            stats=['mean']
                        )
                        if res:
                            mean = res[0]['mean']
                            if mean > line_mean:
                                line_mean = mean
                                line_road_cat = road_cat

                    properties = feature['properties']
                    properties['road_cat'] = line_road_cat

                    # Write the feature to the output shapefile
                    dst.write({
                        'geometry': mapping(geom),
                        'properties': properties
                    })
