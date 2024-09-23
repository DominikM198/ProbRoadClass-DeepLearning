import os
import datetime
import cv2
from PIL import Image
import numpy as np
from osgeo import gdal
from skimage.morphology import skeletonize
from shapely.geometry import LineString, MultiLineString, mapping
from shapely.ops import linemerge
from shapely import unary_union
import fiona
from tqdm import tqdm
gdal.DontUseExceptions()


# Constants
SHEET_NUMBERS = ['199', '385']
SIGFRIED_FILENAME_PREFIX = ''
SIGFRIED_FILENAME_SUFFIX = '_map'
W_SIZE = 500
CC_AREA_THRESHOLD = 100
DOUGLAS_PEUCKER_THRESHOLD = 1.9

path_input_folder = os.path.join(os.path.dirname(__file__), 'input')
path_temp_folder = os.path.join(os.path.dirname(__file__), 'temp')
path_output_folder = os.path.join(os.path.dirname(__file__), 'output')

if not os.path.exists(path_temp_folder):
    os.mkdir(path_temp_folder)
if not os.path.exists(path_output_folder):
    os.mkdir(path_output_folder)


def reconstruct_from_patches(patches_images, patch_size, step_size, image_size_2d, image_dtype):
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


def transform_pixel_coord_to_coord(geotransform, pixel_x, pixel_y):
    """
    Transform pixel coordinates to coordinates.

    Args:
    geotransform: tuple, geotransform of the raster image
    pixel_x: int, x-coordinate of the pixel
    pixel_y: int, y-coordinate of the pixel

    Returns:
    tuple, the coordinates
    """
    origin_x = geotransform[0]
    origin_y = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]
    coord_x = origin_x + pixel_width * (pixel_x + 0.5)
    coord_y = origin_y + pixel_height * (pixel_y + 0.5)
    return coord_x, coord_y


# Iterate over all sheets
for sheet_number in SHEET_NUMBERS:
    print('----------------------------------------------------------------------------------------------------')
    print(f'[{datetime.datetime.now()}] Processing {sheet_number}')
    print('----------------------------------------------------------------------------------------------------')

    print(f'[{datetime.datetime.now()}] Map Stitching: Load the segmentation and mask tiles')
    # Open the Siegfried map
    ds_siegfried = gdal.Open(f'{path_input_folder}/{SIGFRIED_FILENAME_PREFIX}{sheet_number}{SIGFRIED_FILENAME_SUFFIX}.tif')
    array_siegfried_map = ds_siegfried.GetRasterBand(1).ReadAsArray()

    # Load the segmentation and mask tiles
    tiles_array_segmentation = []
    tiles_array_mask = []

    for file_in in [f'{path_input_folder}/{sheet_number}_{i}.tif' for i in range(580)]:
        if not os.path.exists(file_in):
            tiles_array_segmentation.append(-1 * np.ones((500, 500)))
            tiles_array_mask.append(np.zeros((500, 500)))
        else:
            tiles_array_segmentation.append(np.array(Image.open(file_in)))
            tiles_array_mask.append(np.ones((500, 500)))

    print(f'[{datetime.datetime.now()}] Map Stitching: Reconstruct the image from the patches')
    # Reconstruct the image from the patches
    tiles_array_segmentation = np.array(tiles_array_segmentation)
    tiles_array_mask = np.array(tiles_array_mask)
    pad_px = W_SIZE // 2
    segmentation_reconstructed = reconstruct_from_patches(tiles_array_segmentation, W_SIZE, pad_px, array_siegfried_map.shape[:2], np.uint8)
    mask_reconstructed = reconstruct_from_patches(tiles_array_mask, W_SIZE, pad_px, array_siegfried_map.shape[:2], np.uint8)

    print(f'[{datetime.datetime.now()}] Map Stitching: Write the processed raster data to a new raster image')
    # Write the processed raster data to a new raster image
    # segmentation
    driver = gdal.GetDriverByName("GTiff")
    ds_out_segmentation = driver.Create(
        f'{path_temp_folder}/{sheet_number}_segmentation.tif',
        segmentation_reconstructed.shape[1],
        segmentation_reconstructed.shape[0],
        1,
        gdal.GDT_Byte,
        ['COMPRESS=LZW']
    )
    # mask
    ds_out_mask = driver.Create(
        f'{path_temp_folder}/{sheet_number}_mask.tif',
        segmentation_reconstructed.shape[1],
        segmentation_reconstructed.shape[0],
        1,
        gdal.GDT_Byte,
        ['COMPRESS=LZW']
    )
    ds_out_segmentation.SetProjection(ds_siegfried.GetProjection())
    ds_out_mask.SetProjection(ds_siegfried.GetProjection())
    ds_out_segmentation.SetGeoTransform(ds_siegfried.GetGeoTransform())
    ds_out_mask.SetGeoTransform(ds_siegfried.GetGeoTransform())
    ds_out_segmentation.GetRasterBand(1).SetNoDataValue(-1)
    ds_out_segmentation.GetRasterBand(1).WriteArray(segmentation_reconstructed)
    ds_out_mask.GetRasterBand(1).WriteArray(mask_reconstructed)

    # Release resources
    ds_out_segmentation.FlushCache()
    ds_out_segmentation = None
    ds_out_mask.FlushCache()
    ds_out_mask = None

    print(f'[{datetime.datetime.now()}] Morphological Operations: Connected Components')
    raster_data = segmentation_reconstructed
    raster_mask = mask_reconstructed

    # Extract connected components from binary raster image
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raster_data, 4, cv2.CV_32S)
    for label_idx in range(num_labels):
        label_mask = (labels == label_idx)

        if stats[label_idx, 4] <= CC_AREA_THRESHOLD:
            raster_data[label_mask] = 0

    print(f'[{datetime.datetime.now()}] Morphological Operations: Closing')
    kernel_closing = np.ones((3, 3), np.uint8)

    # Closing
    raster_data = cv2.dilate(raster_data, kernel_closing, iterations=1)
    raster_data = cv2.erode(raster_data, kernel_closing, iterations=1)

    # Write the processed raster data to a new raster image
    driver = gdal.GetDriverByName("GTiff")
    ds_mop = driver.Create(
        f'{path_temp_folder}/{sheet_number}_morphological_operations.tif',
        raster_data.shape[1],
        raster_data.shape[0],
        1,
        gdal.GDT_Byte,
        ['COMPRESS=LZW']
    )
    ds_mop.SetProjection(ds_siegfried.GetProjection())
    ds_mop.SetGeoTransform(ds_siegfried.GetGeoTransform())
    ds_mop.GetRasterBand(1).SetNoDataValue(-1)
    ds_mop.GetRasterBand(1).WriteArray(raster_data)
    ds_mop.FlushCache()
    ds_mop = None

    print(f'[{datetime.datetime.now()}] Morphological Operations: Skeletonize')
    # Skeletonize and write the skeletonized raster data to a new raster image
    raster_data = skeletonize(raster_data, method='lee')
    raster_data = raster_mask * raster_data
    driver = gdal.GetDriverByName("GTiff")
    ds_skeleton = driver.Create(
        f'{path_temp_folder}/{sheet_number}_skeleton.tif',
        raster_data.shape[1],
        raster_data.shape[0],
        1,
        gdal.GDT_Byte,
        ['COMPRESS=LZW']
    )
    ds_skeleton.SetProjection(ds_siegfried.GetProjection())
    ds_skeleton.SetGeoTransform(ds_siegfried.GetGeoTransform())
    ds_skeleton.GetRasterBand(1).SetNoDataValue(-1)
    ds_skeleton.GetRasterBand(1).WriteArray(raster_data)

    # Release resources
    ds_siegfried.FlushCache()
    ds_siegfried = None

    print(f'[{datetime.datetime.now()}] Vectorization: Generate Line Segments with 8-neighbourhood')
    raster_data = ds_skeleton.GetRasterBand(1).ReadAsArray()
    geotransform = ds_skeleton.GetGeoTransform()

    with tqdm(total=(raster_data.shape[0] - 2) * (raster_data.shape[1] - 2)) as pbar:
        lines = []
        crossroads = []
        for pixel_y in range(1, raster_data.shape[0] - 1):
            for pixel_x in range(1, raster_data.shape[1] - 1):
                if raster_data[pixel_y, pixel_x] > 0:
                    x_1, y_1 = transform_pixel_coord_to_coord(geotransform, pixel_x, pixel_y)
                    if raster_data[pixel_y, pixel_x + 1] > 0:
                        x_2, y_2 = transform_pixel_coord_to_coord(geotransform, pixel_x + 1, pixel_y)
                        lines.append(LineString([[x_1, y_1], [x_2, y_2]]))
                    if raster_data[pixel_y + 1, pixel_x] > 0:
                        x_2, y_2 = transform_pixel_coord_to_coord(geotransform, pixel_x, pixel_y + 1)
                        lines.append(LineString([[x_1, y_1], [x_2, y_2]]))
                    if (raster_data[pixel_y + 1, pixel_x - 1] > 0) and not ((raster_data[pixel_y + 1, pixel_x] > 0) or (raster_data[pixel_y, pixel_x-1] > 0)):
                        x_2, y_2 = transform_pixel_coord_to_coord(geotransform, pixel_x - 1, pixel_y + 1)
                        lines.append(LineString([[x_1, y_1], [x_2, y_2]]))
                    if (raster_data[pixel_y + 1, pixel_x + 1] > 0) and not ((raster_data[pixel_y + 1, pixel_x] > 0) or (raster_data[pixel_y, pixel_x+1] > 0)):
                        x_2, y_2 = transform_pixel_coord_to_coord(geotransform, pixel_x + 1, pixel_y + 1)
                        lines.append(LineString([[x_1, y_1], [x_2, y_2]]))
                pbar.update(1)

    print(f'[{datetime.datetime.now()}] Vectorization: Dissolve line segments')
    multilinestring = unary_union(lines)
    if isinstance(multilinestring, LineString):
        multilinestring = MultiLineString([multilinestring])

    print(f'[{datetime.datetime.now()}] Vectorization: Merge lines to get disjoint features')
    linemerge_multilinestring = linemerge(multilinestring)
    if isinstance(linemerge_multilinestring, LineString):
        linemerge_multilinestring = MultiLineString([linemerge_multilinestring])

    schema = {
        'geometry': 'LineString',
        'properties': {'id': 'int'},
        'crs': 'EPSG:21781'
    }

    print(f'[{datetime.datetime.now()}] Filtering and Generalization: Filter coordinate grid, simplify and export as ESRI Shapefiles')
    with fiona.open(f'{path_output_folder}/{sheet_number}_road_geoms.shp', 'w', 'ESRI Shapefile', schema) as dst:
        for idx, geom in enumerate(linemerge_multilinestring.geoms):
            properties = {'id': idx}

            # Flag to filter out geometries that are likely to be coordinate grid lines
            filter_geom = False

            dx = 0
            dy = 0
            mean_x = geom.coords[0][0]
            mean_y = geom.coords[0][1]
            for i in range(1, len(geom.coords)):
                dx += geom.coords[i][0] - geom.coords[i - 1][0]
                dy += geom.coords[i][1] - geom.coords[i - 1][1]
                mean_x += geom.coords[i][0]
                mean_y += geom.coords[i][1]
            mean_x /= len(geom.coords)
            mean_y /= len(geom.coords)

            # Check if the geometry is likely to be a coordinate grid line
            if abs(((mean_x + 500) % 1000) - 500) < 15 and abs(dx) <= min(len(geom.coords) * 4, 12) and abs(
                    dy) > 10 * abs(dx):
                filter_geom = True
            if abs(((mean_y + 500) % 1000) - 500) < 15 and abs(dy) <= min(len(geom.coords) * 4, 12) and abs(
                    dx) > 10 * abs(dy):
                filter_geom = True

            # Write the feature to the output shapefile if it is not a coordinate grid line
            if not filter_geom:
                geom_simplified = geom.simplify(DOUGLAS_PEUCKER_THRESHOLD, preserve_topology=True)
                dst.write({
                    'geometry': mapping(geom_simplified),
                    'properties': properties
                })

    with fiona.open(f'{path_output_folder}/{sheet_number}_road_geoms.shp', 'r') as src:
        print(f'[{datetime.datetime.now()}] Filtering and Generalization: Dissolve and merge lines to get disjoint features')
        multilinestring = unary_union(lines)
        if isinstance(multilinestring, LineString):
            multilinestring = MultiLineString([multilinestring])

        linemerge_multilinestring = linemerge(multilinestring)
        if isinstance(linemerge_multilinestring, LineString):
            linemerge_multilinestring = MultiLineString([linemerge_multilinestring])

