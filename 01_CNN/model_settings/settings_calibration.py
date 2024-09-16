##!/usr/bin/env python

"""
Functions for reading and implementing model settings specified in json files
"""

### IMPORTS ###
import json
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from data.data_utils import Historic_Data_Zurich_Dynamic, Swiss_Map_Dynamic, Painting_based_Classification, create_data_augmentation, class_weights_inverse_num_of_samples
from training_evaluation.train_validate import Cosine_Schedule
from models.models import Resnet18_Unet_Big_Attention, DeepEnsemble, UNet_small, Resnet18_Unet_Big_Classification
import const
from losses.loss import DiceLoss, LabelSmoothingCrossEntropy
from utils import DEVICE


### FUNCTIONS ###
def dataloader(data: dict, dataset_type: str, transform: transforms.Compose = None, standardize: transforms.Compose  = None, evaluate_training = False) -> None: 
    """
    Function to get the dataloader for the dataset.

    Parameters
    ----------
    data : dict
        Dictionary containing the data settings.
    dataset_type : str
        The dataset to use, either "train", "validate", or "test".
    transform : transforms.Compose
        Transformations to be applied to the dataset during training, optional.
    standardize : transforms.Compose
        Standardization transformations to be applied to the dataset, optional.
    """
    num_workers=1
    # Implement historic maps Zurich dataset
    if data["data"]["data_set"] == "historic_maps_zurich":
        dataset = Historic_Data_Zurich_Dynamic(dataset_type, transform=transform, standardize=standardize)

    # Implement swiss map dataset
    elif data["data"]["data_set"] == "swissmap":
        dataset = Swiss_Map_Dynamic(dataset_type, transform=transform, standardize=standardize)

    # Implement painting based classification dataset (training and validation based on synthetic data), test based on real data
    elif data["data"]["data_set"] == "synthetic_road_classification":
        if "breakpoint_prediction" in data["data"].keys() and data["data"]["breakpoint_prediction"] is True:
            dataset = Painting_based_Classification(dataset_type, transform=transform, standardize=standardize, task="breakpoint_prediction")
            num_workers = 4
        else:
            dataset = Painting_based_Classification(dataset_type, transform=transform, standardize=standardize)

    # Case of other datasets
    else:
        raise ValueError("Dataset not supported or recognized")
    
    # Get training dataloader 
    if dataset_type == "train":
        if evaluate_training is False:
            data["training"]["dataloader"] = DataLoader(dataset, batch_size=data["training"]["batch_size"], shuffle=data["data"]["shuffle"], num_workers=num_workers)
            data["training"]["steps_per_epoch"] = len(data["training"]["dataloader"])
        else:
            data["evaluation"]["dataloader"] = DataLoader(dataset, batch_size=data["evaluation"]["batch_size"], shuffle=data["data"]["shuffle"], num_workers=1)

        # Get class weights if inverse_num_of_samples is selected
        if data["training"]["class_weights"] == "inverse_num_of_samples":
            # Iterate over the dataset to get the number of samples per class
            list_samples_per_class = [0] * data["model"]["num_classes"]
            for i, (image, target, mask, filename) in enumerate(data["training"]["dataloader"]):
                for i in range(data["model"]["num_classes"]):
                    list_samples_per_class[i] += torch.sum(target == i).item()
            data["training"]["class_weights"] = class_weights_inverse_num_of_samples(data["model"]["num_classes"], list_samples_per_class).tolist()

            # Print the class weights
            print("Class weights: ", data["training"]["class_weights"])

    # Get evaluation dataloader 
    else:
        data["evaluation"]["dataloader"] = DataLoader(dataset, batch_size=data["evaluation"]["batch_size"], shuffle=data["data"]["shuffle"], num_workers=1)


def data_augmentation(data: dict, train) -> None:
    """
    Adds data augmentation to the data dictionary.

    Parameters
    ----------
    data : dict
        Dictionary containing the settings.
    train : bool
        Boolean indicating whether the data augmentation is for training or evaluation.
    """

    # Initialize the variables
    flip_image = False
    rotate_image = False
    rotate_image_continous = False
    mean_pixel = None
    std_pixel = None

    # If standardize is part of the data augmentation
    if  data["evaluation"]["standardize"] == "studentization":
        # Get the mean and standard deviation of the pixels
        mean_pixel = torch.tensor(data["data"]["channel_mean"])
        std_pixel = torch.tensor(data["data"]["channel_std"])

        # Create the data augmentation and add it to the data dictionary
        data["training"]["standardize"] = create_data_augmentation(data["data"]["input_size"], 
                                                                   standardize = True, 
                                                                   standardize_mean = mean_pixel, 
                                                                   standardize_std = std_pixel)
        data["evaluation"]["standardize"] = data["training"]["standardize"]
        
    # Data augmentation for training 
    if train:

        # If flip is part of the data augmentation
        if "flip" in data["training"]["training_augmentation"]:
            flip_image = True

        # If rotate is part of the data augmentation (discrete roation of 0, 90, 180, 270 degrees)
        if "rotate" in data["training"]["training_augmentation"]:
            rotate_image = True
        
        # If rotate continous is part of the data augmentation (rotation of 0-360 degrees)
        if "rotate_continous" in data["training"]["training_augmentation"]:
            rotate_image_continous = True

        # Create the data augmentation and add it to the data dictionary
        data["training"]["training_augmentation"] = create_data_augmentation(data["data"]["input_size"], flip_image, rotate_image, rotate_image_continous)
        

def optimizer(data: dict) -> None:
    """
    Function to get the optimizer for the training. 

    Parameters
    ----------
    data : dict
        Dictionary containing the settings.
    """

    # If Adam is the optimizer
    if data["training"]["optimizer"] == "Adam":

        # Initalize the optimizer and specify the settings
        lr = data["training"]["learning_rate"]
        betas = data["training"]["Adam_betas"]
        weight_decay = data["training"]["weight_decay"]
        data["training"]["optimizer"] = torch.optim.Adam(data["model"]["model"].parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    # If AdamW is the optimizer
    elif data["training"]["optimizer"] == "AdamW":

        # Initalize the optimizer and specify the settings
        lr = data["training"]["learning_rate"]
        betas = data["training"]["Adam_betas"]
        weight_decay = data["training"]["weight_decay"]
        data["training"]["optimizer"] = torch.optim.AdamW(data["model"]["model"].parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    # Not implemented or recognized
    else:
        raise ValueError("Optimizer not implemented or recognized")
    

def learning_rate_schedule(data: dict) -> None:
    """
    Function to get the learning rate schedule for the training. 

    Parameters
    ----------
    data : dict
        Dictionary containing the settings.
    """

    # Add the learning rate schedule to the settings dictionary

    # Cosine learning rate schedule. Base learning rate with linear warmup and cosine decay
    if data["training"]["lr_schedule"] == "cosine":
        data["training"]["lr_schedule"] = Cosine_Schedule(data["training"]["optimizer"], data["training"]["steps_lr_warmup"], data["training"]["max_epochs"] * data["training"]["steps_per_epoch"])
    
    # Constant learning rate 
    elif data["training"]["lr_schedule"] == "constant":
        data["training"]["lr_schedule"] = torch.optim.lr_scheduler.ConstantLR(data["training"]["optimizer"], factor=1, total_iters=1, last_epoch=-1)
    
    # Not implemented or recognized
    else:
        raise ValueError("Learning rate schedule not implemented or recognized")
    

def loss(data: dict) -> None:
    """
    Function to get the loss for the training. The loss is added to the settings dictionary.

    Parameters
    ----------
    data : dict
        Dictionary containing the settings.
    """

    # Class weights are converted to tensor
    data["training"]["class_weights"] = torch.tensor(data["training"]["class_weights"])

    # If the loss is binary dice loss
    if data["training"]["loss"] == "Dice":
        data["training"]["loss"] = DiceLoss()
        data["training"]["loss_name"] = "Dice"

    # If the loss is label smoothing cross entropy
    elif data["training"]["loss"] == "LabelSmoothingNLL":
        data["training"]["loss"] = LabelSmoothingCrossEntropy(epsilon=data["training"]["epsilon"], reduction='mean', weight=data["training"]["class_weights"], ignore_index=99)
        data["training"]["loss_name"] = "LabelSmoothingNLL"
        
    # If the loss is negative log likelihood (same as cross entropy but without the softmax layer implemented in the loss function)
    elif data["training"]["loss"] == "NLL":
        data["training"]["loss"] = torch.nn.NLLLoss(weight=data["training"]["class_weights"], ignore_index=99)
        data["training"]["loss_name"] = "NLL"

    # Not implemented or recognized
    else:
        raise ValueError("Loss not implemented or recognized")


def init_model(data: dict) -> None:
    """
    Function to get the model for the training and evaluation. 

    Parameters
    ----------
    data : dict
        Dictionary containing the settings.

    """ 

    # Set the device
    model = None

    # Implement U-Net-small (baseline)
    if data["model"]["Name"] == "U-Net-small":
        data["model"]["model"] = UNet_small(3).to(DEVICE)

    # Implement U-Net-Resnet18-Big-Attention for binary classification 
    elif data["model"]["Name"] == "U-Net-Resnet18-Big-Attention":

        if "imagenet1k" in data["model"].keys():
            imagenet1k = data["model"]["imagenet1k"]
        else:
            imagenet1k = True
        data["model"]["model"] = Resnet18_Unet_Big_Attention(1, imagenet1k).to(DEVICE)
    
    # Implement DeepEnsemble for multi-class classification
    elif data["model"]["Name"] == "DeepEnsemble":
        data["model"]["model"] = DeepEnsemble(data["model"]["num_classes"], data["model"]["ensemble_list"])


    # Implement U-Net-Resnet18-Big-Classification for multi-class classification
    elif data["model"]["Name"] == "U-Net-Resnet18-Big-Classification":
            
        if "imagenet1k" in data["model"].keys():
            imagenet1k = data["model"]["imagenet1k"]
        else:
            imagenet1k = True
        if data["model"]["pretrained_segmentation"] is True:
            path_pretrained_segmentation = const.MODEL_STORAGE_DIR.joinpath(data["model"]["pretrained_segmentation_model_name"] + ".pt")
        data["model"]["model"] = Resnet18_Unet_Big_Classification(data["model"]["num_classes"], imagenet1k, path_pretrained_segmentation).to(DEVICE)
    else:
        raise ValueError("Model not implemented or recognized")

    # Load weights if pretrained model is used
    if data["model"]["Name"] == "DeepEnsemble":
        pass
    elif data["model"]["pretrained"] is True:
        path = const.MODEL_STORAGE_DIR.joinpath(data["model"]["pretrained_model_name"] + ".pt")
        data["model"]["model"].load_state_dict(torch.load(path))
    
    # Freeze the model parameters if the model is not for training
    if data["training"]["training"] is False:
        for param in data["model"]["model"].parameters():
            param.requires_grad = False
    
    # Print the model parameters trainable and non-trainable
    print("Model parameters:")
    total_params = sum(p.numel() for p in data["model"]["model"].parameters())
    trainable_params = sum(p.numel() for p in data["model"]["model"].parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    
def interpret_json(path: str) -> dict:
    """
    Function to read and interpret the json file. Adds implementations for data augmentation, optimizer, learning rate schedule and loss.

    Parameters
    ----------
    path : str
        Path to the json file.

    Returns
    -------
    data : dict
        Dictionary containing the settings.
    """

    # Read the json file
    data = json.load(open(path))
    
    # Implement data augmentation and dataloader for training 
    if data["training"]["training"] is True:
        # Implement data augmentation for training 
        data_augmentation(data, train = True)
        dataloader(data, dataset_type = "train", transform = data["training"]["training_augmentation"], standardize=data["training"]["standardize"])
    
    # Implement data augmentation and dataloader for evaluation
    if data["evaluation"]["evaluation"] is True:
        if data["evaluation"]["use_test_set"] is True:
            # Implement data for test set
            data_augmentation(data, train = False)
            dataloader(data, dataset_type= "test", standardize=data["evaluation"]["standardize"])
        elif "use_training_set" in data["evaluation"].keys() and data["evaluation"]["use_training_set"] is True:
            # Implement data for training set
            data_augmentation(data, train = False)
            dataloader(data, dataset_type= "train", standardize=data["evaluation"]["standardize"], evaluate_training = True)
        else:
            # Implement data for validation set
            data_augmentation(data, train = False)
            dataloader(data, dataset_type= "validate", standardize=data["evaluation"]["standardize"])

    # Implement the model
    init_model(data)

    if data["training"]["training"] is True:
        # Implement optimizer
        optimizer(data)

        # Implement learning rate schedule
        learning_rate_schedule(data)

    # Implement loss
    loss(data)     

    # Generate file name for saving results and models
    data["model"]["json_file"] = os.path.basename(path).split(".")[0]
    data["data"]["json_file_name"] = os.path.basename(path).split(".")[0]
    data["data"]["result_file_name"] = f"{data['model']['Name']}_{data['data']['json_file_name']}"
    data["data"]["result_file_name"] = data["data"]["result_file_name"].replace("-", "_")

    # Add tensorboard summary writer to the settings dictionary
    if data["data"]["tensorboard"] is True:
        data["data"]["tensorboard_writer"] = SummaryWriter(const.LOGS_DIR.joinpath('runs/{}'.format(data["data"]["result_file_name"])))

    
    return data
