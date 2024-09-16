#!/usr/bin/env python

"""
Implements the data loaders for the research project
"""

### IMPORTS ###
from typing import Tuple
import torch 
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import const
import os
from PIL import Image
import cv2
import random
from skimage.util import view_as_windows
import fiona
from shapely.geometry import shape, mapping, LineString
from osgeo import gdal
import pickle
gdal.DontUseExceptions()


# Constants
Image.MAX_IMAGE_PIXELS = 140000000

# Buffer size for the road labels
BUFFERSIZE_LABELS = 10


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


def _draw_solid_line(img, geom, geotransform, color, thickness):
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
    for feature in features:
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


### CLASS DEFINITIONS ###
class Historic_Data_Zurich_Dynamic(Dataset):
    """
    Class to create historic map dataset from Zurich
    """
    def __init__(self, dataset: str,  transform: transforms.Compose = None, standardize: transforms.Compose = None) -> None:
        """
        Initializes the dataset

        Parameters
        ----------
        dataset : str
            The dataset to use, either "train", "validate", or "test".
        transform : transforms.Compose
            Transformations to be applied to the dataset
        standardize : transforms.Compose
            Standardization transformations to be applied to the dataset
        """
        self.transform = transform
        self.standardize = standardize
        self.dataset = dataset
        self.arrays_map = []
        self.arrays_label = []
        self.arrays_mask = []
        self.array_file_names = []

        path_siegfried_sheets = const.DATA_DIR.joinpath("segmentation").joinpath(dataset).joinpath('siegfried_sheets')
        path_road_geoms = const.DATA_DIR.joinpath("segmentation").joinpath(dataset).joinpath('road_geoms')
        path_mask_sheets = const.DATA_DIR.joinpath("segmentation").joinpath(dataset).joinpath('masks')
        path_label_sheets = const.DATA_DIR.joinpath("segmentation").joinpath(dataset).joinpath('labels')

        for siegfried_sheet in os.listdir(path_siegfried_sheets):
            cur_sheet_number = siegfried_sheet.split('_')[0]
            tiles_array_map = _generate_tiling(
                path_siegfried_sheets.joinpath(f'{cur_sheet_number}_map.tif'),
                500,
                mode='rgb'
            )

            path_cur_siegfried_sheet = path_siegfried_sheets.joinpath(siegfried_sheet)
            shapefile = path_road_geoms.joinpath(f'{cur_sheet_number}_road_geoms.shp')
            ds_cur_siegfried_sheet = gdal.Open(str(path_cur_siegfried_sheet))
            array_cur_siegfried_sheet = ds_cur_siegfried_sheet.ReadAsArray()
            array_cur_siegfried_sheet_reordered = array_cur_siegfried_sheet.transpose((1, 2, 0)).copy()
            cur_label_sheet = np.zeros(
                (array_cur_siegfried_sheet_reordered.shape[0], array_cur_siegfried_sheet_reordered.shape[1]),
                dtype=np.uint8
            ) * 255
            cur_geotransform = ds_cur_siegfried_sheet.GetGeoTransform()

            with fiona.open(shapefile) as src:
                cur_label_sheet = _paint_road_labels(cur_label_sheet, src, cur_geotransform, segmentation_mode=True)
                _save_array_as_gtiff(
                    cur_label_sheet,
                    path_label_sheets.joinpath(f'{cur_sheet_number}_labels.tif'),
                    ds_cur_siegfried_sheet
                )
                tiles_array_label = _generate_tiling(
                    path_label_sheets.joinpath(f'{cur_sheet_number}_labels.tif'),
                    500,
                    mode='grayscale'
                )

            if self.dataset == 'train':
                tiles_array_mask = _generate_tiling(
                    path_mask_sheets.joinpath(f'{cur_sheet_number}_mask.tif'),
                    500,
                    mode='grayscale'
                )
            else:
                tiles_array_mask = np.ones_like(tiles_array_label)

            for i in range(tiles_array_mask.shape[0]):
                if np.any(tiles_array_mask[i] == 1):
                    self.arrays_map.append(tiles_array_map[i])
                    self.arrays_label.append(tiles_array_label[i])
                    self.arrays_mask.append(tiles_array_mask[i])
                    self.array_file_names.append(f'{cur_sheet_number}_{i}')

    def __len__(self) -> int:
        """
        Returns the length of the dataset
        """
        return len(self.arrays_map)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns an image and a road segmentation mask

        Parameters:
        ----------
        - idx: int
            The index of the image to return

        Returns:
        --------
        - image: torch.Tensor
            The image
        - class_label: torch.Tensor
            The class label
        """
        # read file name
        file_name = self.array_file_names[idx]

        # read the map, label and mask
        map_image = torch.from_numpy(np.moveaxis(self.arrays_map[idx], -1, 0)).float()
        label = torch.from_numpy(self.arrays_label[idx]).squeeze().unsqueeze(0).float()
        mask = torch.from_numpy(self.arrays_mask[idx]).squeeze().unsqueeze(0).float()

        # Apply transformations
        if self.transform:
            images = torch.cat((map_image, label, mask))
            images = self.transform(images)
            map_image = images[:3]
            label = images[3]
            mask = images[4]
        else:
            label = label.squeeze(1)
            mask = mask.squeeze(1)

        # Apply standardization
        if self.standardize:
            map_image = self.standardize(map_image)

        return map_image, label, mask, file_name


class Swiss_Map_Dynamic(Dataset):
    """
    Class to create historic map dataset from Zurich
    """
    def __init__(self, dataset: str,  transform: transforms.Compose = None, standardize: transforms.Compose = None, preprocess = True) -> None:
        """
        Initializes the dataset

        Parameters
        ----------
        dataset : str
            The dataset to use, either "train", "validate", or "test".
        transform : transforms.Compose
            Transformations to be applied to the dataset
        standardize : transforms.Compose
            Standardization transformations to be applied to the dataset
        """
        self.transform = transform
        self.standardize = standardize
        self.dataset = dataset
        self.arrays_map = []
        self.arrays_label = []
        self.array_file_names = []

        if preprocess:
            path_swissmap_sheets = const.DATA_DIR.joinpath("segmentation_pretraining").joinpath(dataset).joinpath('swissmap_sheets')
            path_road_geoms = const.DATA_DIR.joinpath("segmentation_pretraining").joinpath(dataset).joinpath("road_geoms")
            path_label_sheets = const.DATA_DIR.joinpath("segmentation_pretraining").joinpath(dataset).joinpath('labels')

            for swissmap_sheet in os.listdir(path_swissmap_sheets):
                cur_sheet_number = swissmap_sheet.split('_')[0]
                path_cur_swissmap_sheet = path_swissmap_sheets.joinpath(swissmap_sheet)
                shapefile = path_road_geoms.joinpath(f'{cur_sheet_number}_road_geoms.shp')
                ds_cur_swissmap_sheet = gdal.Open(str(path_cur_swissmap_sheet))
                array_cur_swissmap_sheet = ds_cur_swissmap_sheet.ReadAsArray()
                array_cur_swissmap_sheet_reordered = array_cur_swissmap_sheet.transpose((1, 2, 0)).copy()
                cur_label_sheet = np.zeros(
                    (array_cur_swissmap_sheet_reordered.shape[0], array_cur_swissmap_sheet_reordered.shape[1]),
                    dtype=np.uint8
                )
                cur_geotransform = ds_cur_swissmap_sheet.GetGeoTransform()

                with fiona.open(shapefile) as src:
                    cur_label_sheet = _paint_road_labels(cur_label_sheet, src, cur_geotransform, segmentation_mode=True)
                    _save_array_as_gtiff(
                        cur_label_sheet,
                        path_label_sheets.joinpath(f'{cur_sheet_number}_labels.tif'),
                        ds_cur_swissmap_sheet
                    )
                    tiles_array_label = _generate_tiling(
                        path_label_sheets.joinpath(f'{cur_sheet_number}_labels.tif'),
                        500,
                        mode='grayscale'
                    )
                tiles_array_map = _generate_tiling(
                    path_swissmap_sheets.joinpath(f'{cur_sheet_number}_map.tif'),
                    500,
                    mode='rgb'
                )
                for i in range(tiles_array_label.shape[0]):
                    self.arrays_map.append(tiles_array_map[i])
                    self.arrays_label.append(tiles_array_label[i])
                    self.array_file_names.append(f'{cur_sheet_number}_{i}')
            
            for i in range(len(self.array_file_names)):
                with open(const.DATA_DIR.joinpath("segmentation_pretraining").joinpath("data").joinpath("{}_{}.npy".format(dataset,i)),'wb') as f:
                    np.save(f, self.arrays_map[i])
                    np.save(f, self.arrays_label[i])
                
            with open(const.DATA_DIR.joinpath("segmentation_pretraining").joinpath("data").joinpath("filenames"),'wb') as f:
                pickle.dump( self.array_file_names, f)

            # release memory
            self.arrays_map = []
            self.arrays_label = []
            tiles_array_map = []
            tiles_array_label = []
        else:
            with open(const.DATA_DIR.joinpath("segmentation_pretraining").joinpath("data").joinpath("filenames"),'rb') as f:
                self.array_file_names = pickle.load(f)

    def __len__(self) -> int:
        """
        Returns the length of the dataset
        """
        return len(self.array_file_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns an image and a road segmentation mask

        Parameters:
        ----------
        - idx: int
            The index of the image to return

        Returns:
        --------
        - image: torch.Tensor
            The image
        - class_label: torch.Tensor
            The class label
        """
        # read file name
        file_name = self.array_file_names[idx]

        # read the map, label and mask
        with open(const.DATA_DIR.joinpath("segmentation_pretraining").joinpath("data").joinpath("{}_{}.npy".format(self.dataset,idx)),'rb') as f:
            map_image = torch.from_numpy(np.moveaxis(np.load(f), -1, 0)).float()
            label = torch.from_numpy(np.load(f)).squeeze().unsqueeze(0).float()

        mask = torch.from_numpy(np.ones_like(label)).squeeze().unsqueeze(0).float()

        # Apply transformations
        if self.transform:
            images = torch.cat((map_image, label, mask))
            images = self.transform(images)
            map_image = images[:3]
            label = images[3]
            mask = images[4]
        else:
            label = label.squeeze(1)
            mask = mask.squeeze(1)

        # Apply standardization
        if self.standardize:
            map_image = self.standardize(map_image)
    
        return map_image, label, mask, file_name


class Painting_based_Classification(Dataset):
    """
    Class to classify road category labels with synthetic training and validation data using the symbol definition
    of those road categories in the Siegfried maps for rendering the vector road data.
    """

    def __init__(self, dataset: str,  transform: transforms.Compose = None, standardize: transforms.Compose = None, task = "road_classification") -> None:
        """
        Initializes the dataset

        Parameters
        ----------
        dataset : str
            The dataset to use, either "train", "validate", or "test".
        transform : transforms.Compose
            Transformations to be applied to the dataset
        standardize : transforms.Compose
            Standardization transformations to be applied to the dataset
        """
        self.transform = transform
        self.standardize = standardize
        self.task = task

        # Set the dataset (train, validate, test)
        self.dataset = dataset

        # Declare paths
        path_base = const.DATA_DIR.joinpath("classification").joinpath(dataset).joinpath("tiles")
        if self.dataset in ('train', 'validate'):
            self.path_tiles_map = path_base.joinpath("painted")
            self.path_tiles_breakpoint_input = path_base.joinpath("breakpoint_input")
            self.path_tiles_road_class_labels = path_base.joinpath("road_class_labels")
            self.path_tiles_breakpoint_labels = path_base.joinpath("breakpoint_labels")
            self.path_tiles_mask = path_base.joinpath("masks") if os.path.exists(path_base.joinpath("masks")) else None
            self.path_tiles_hard_mask = path_base.joinpath("hard_masks")
        else:
            self.path_tiles_map = path_base.joinpath("map")
            self.path_tiles_breakpoint_input = path_base.joinpath("breakpoint_input")
            self.path_tiles_road_class_labels = path_base.joinpath("road_class_labels")
            self.path_tiles_breakpoint_labels = path_base.joinpath("breakpoint_labels")
            self.path_tiles_mask = path_base.joinpath("masks") if os.path.exists(path_base.joinpath("masks")) else None
            self.path_tiles_hard_mask = path_base.joinpath("hard_masks")

        self.file_name_prefixes = [f'{file.split("_")[0]}_{file.split("_")[1]}' for file in os.listdir(self.path_tiles_map)]

    def __len__(self) -> int:
        """
        Returns the length of the dataset
        """
        return len(self.file_name_prefixes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns an image and a road segmentation mask

        Parameters:
        ----------
        - idx: int
            The index of the image to return

        Returns:
        --------
        - image: torch.Tensor
            The image
        - class_label: torch.Tensor
            The class label
        """
        # read the file name
        file_name = self.file_name_prefixes[idx]

        if self.task == "road_classification":
            # move the channel dimension to the first dimension
            map_suffix = 'painted' if self.dataset in ('train', 'validate') else 'map'
            map_image = torch.from_numpy(
                np.moveaxis(np.array(Image.open(self.path_tiles_map.joinpath(f'{file_name}_{map_suffix}.tif'))), -1, 0)
            ).float()

            # read the road class label
            label = torch.from_numpy(
                np.array(Image.open(self.path_tiles_road_class_labels.joinpath(f'{file_name}_road_class_labels.tif')))
            ).squeeze().unsqueeze(0).float()

        elif self.task == "breakpoint_prediction":
            class0 = torch.tensor(
                np.array(Image.open(self.path_tiles_breakpoint_input.joinpath(f'{file_name}_class0.tif')))
            ).unsqueeze(0)
            class1 = torch.tensor(
                np.array(Image.open(self.path_tiles_breakpoint_input.joinpath(f'{file_name}_class1.tif')))
            ).unsqueeze(0)
            class2 = torch.tensor(
                np.array(Image.open(self.path_tiles_breakpoint_input.joinpath(f'{file_name}_class2.tif')))
            ).unsqueeze(0)
            class3 = torch.tensor(
                np.array(Image.open(self.path_tiles_breakpoint_input.joinpath(f'{file_name}_class3.tif')))
            ).unsqueeze(0)
            class4 = torch.tensor(
                np.array(Image.open(self.path_tiles_breakpoint_input.joinpath(f'{file_name}_class4.tif')))
            ).unsqueeze(0)
            class5 = torch.tensor(
                np.array(Image.open(self.path_tiles_breakpoint_input.joinpath(f'{file_name}_class5.tif')))
            ).unsqueeze(0)
            map_image = torch.cat((class0, class1, class2, class3, class4, class5))

            # read the breakpoint label
            label = torch.from_numpy(
                np.array(Image.open(self.path_tiles_breakpoint_labels.joinpath(f'{file_name}_breakpoint_labels.tif')))
            ).squeeze().unsqueeze(0).float()

        else:
            raise ValueError("Invalid task")

        # read the hard mask
        hard_mask = torch.from_numpy(
            np.array(Image.open(self.path_tiles_hard_mask.joinpath(f'{file_name}_hard_mask.tif')))
        ).squeeze().unsqueeze(0).float()

        # mask
        if self.path_tiles_mask is not None:
            mask = torch.from_numpy(
                np.array(Image.open(self.path_tiles_mask.joinpath(f'{file_name}_mask.tif')))
            ).squeeze().unsqueeze(0).float()
        else:
            mask = torch.from_numpy(np.full_like(label, dtype=np.uint8, fill_value=1))

        # Apply transformations
        if self.transform:
            if self.task == "breakpoint_prediction":
                images = torch.cat((map_image, label, mask, hard_mask))
                images = self.transform(images)
                map_image = images[:6]
                label = images[6]
                mask = images[7].unsqueeze(0)
                hard_mask = images[8]

            else: 
                images = torch.cat((map_image, label, mask, hard_mask))
                images = self.transform(images)
                map_image = images[:3]
                label = images[3]
                mask = images[4].unsqueeze(0)
                hard_mask = images[5]
        else:
            label = label.squeeze(0)
            mask = mask
            hard_mask = hard_mask

        # Apply standardization
        if self.standardize:
            map_image = self.standardize(map_image)

        return map_image, label, (mask, hard_mask), file_name


class RandomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles = [0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        # Choose a random angle
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)


### STATIC FUNCTIONS ###
def create_data_augmentation(image_size: Tuple[int, int], flip_image: bool = False, rotate_image: bool = False, 
                             rotate_image_continous: bool = False, standardize: bool = False, 
                             standardize_mean: torch.tensor = torch.tensor([125, 125, 125]), 
                             standardize_std: torch.tensor = torch.tensor([50, 50, 50])) -> transforms.Compose:
    """
    This function creates a data augmentation pipeline.

    Parameters:
    ----------
    image_size: Tuple[int, int]
        The size of the image to be used for the training
    flip_image: bool
        Whether to flip the image or not
    rotate_image: bool, optional
        Whether to rotate the image or not
    standardize: bool, optional
        Whether to standardize the image or not
    standardize_mean: torch.tensor, optional
        The mean pixel value for standardization
    standardize_std: torch.tensor, optional
        The standard deviation pixel value for standardization

    Returns:
    ----------
    transforms.Compose
        The data augmentation pipeline
    """

    # Create the list of augmentations
    augmentation_list = []

    # Data standardization
    if standardize:
        augmentation_list.append(transforms.Normalize(mean=standardize_mean, std=standardize_std))

    # Add flip augmentation
    if flip_image:
        augmentation_list.append(transforms.RandomHorizontalFlip())
        augmentation_list.append(transforms.RandomVerticalFlip())

    # Add rotation augmentation
    if rotate_image:
        # augmentation_list.append(transforms.RandomRotation(degrees=(0, 180)))      
        augmentation_list.append(RandomRotationTransform())  
    
    # Add rotation augmentation continous
    if rotate_image_continous:
        augmentation_list.append(transforms.RandomRotation(degrees=(0, 360)))

    # Add resizing augmentation
    augmentation_list.append(transforms.Resize((image_size[0], image_size[1]), antialias=True))

    return transforms.Compose(augmentation_list)


def class_weights_inverse_num_of_samples(nr_classes, samples_per_class, power=1):
    """
    Function to calculate class weights based on the inverse number of samples per class. 
    The power parameter can be used to adjust the weights. Popular values are 0.5 (square root) or 1.
    The weights are normalized to sum to 1.

    Source
    ------
    https://medium.com/gumgum-tech/handling-class-imbalance-by-introducing-sample-weighting-in-the-loss-function-3bdebd8203b4

    Parameters
    ----------
    nr_classes : int
        The number of classes
    samples_per_class : list
        The number of samples per class
    power : int, optional
        The power to apply to the weights, by default 1
        
    Returns
    -------
    class_weights : list
        The class weights
    """

    # calculate class weights
    class_weights = 1.0 / np.array(np.power(samples_per_class, power))
    class_weights = class_weights / np.sum(class_weights) * nr_classes

    # normalize class weights
    class_weights = class_weights / class_weights.sum()
    return class_weights




