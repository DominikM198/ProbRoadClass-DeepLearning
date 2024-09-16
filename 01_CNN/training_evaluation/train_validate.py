##!/usr/bin/env python

"""
Implements training and validation functions for image segmentation models.
"""

### IMPORTS ###
import math
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from utils import DEVICE
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import const
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import torchmetrics

### FUNCTIONS ###

def save_output(prediction, back_transform, settings, file_name, category=""):
    """
    Save the model output to disk. Back transform the prediction to discrete RGB values (0-255).

    Parameters
    ----------
    prediction : torch.Tensor
        Model output.
    back_transform : torchvision.transforms
        Back transformation to discrete RGB values.
    settings : dict
        Dictionary containing all settings for training and evaluation.
    file_name : str
        Name of the file.
    category : str, optional
        Category of the prediction (default is "").
    """

    # Back transform the prediction
    predicted_images = back_transform(prediction).cpu().numpy().astype(float)

    # Save each prediction to disk
    for i in range(len(file_name)):
        # Get the predicted image
        predicted_image = predicted_images[i]

        # Get the save path
        save_path = const.RESULTS_DIR.joinpath(settings["data"]["result_file_name"]).joinpath("segmentation").joinpath(file_name[i] + category + ".tif")

        # Create the directory if it does not exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the prediction as .tif file
        cur_image = Image.fromarray(predicted_image)
        cur_image.save(save_path)


class Cosine_Schedule(LambdaLR):
    """ 
    Cosine learning rate schedule. Linearly increases the learning rate from 0 to 1 during the warm-up phase and then follows a cosine pattern.
    """
    def __init__(self, optimizer: torch.optim, steps_warmup: int, steps_total: int, cycles: float =.5, last_epoch: int = -1):
        """
        Initialize the cosine learning rate schedule.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer for which the learning rate is scheduled.
        steps_warmup : int
            Number of warm-up steps.
        steps_total : int
            Total number of training steps.
        cycles : float, optional
            Number of cosine cycles (default is 0.5).
        last_epoch : int, optional
            The index of the last epoch (default is -1).
        """
        self.steps_warmup = steps_warmup
        self.steps_total = steps_total
        self.cycles = cycles
        super(Cosine_Schedule, self).__init__(optimizer, self.lr_lambda, last_epoch = last_epoch)

    def lr_lambda(self, step: int) -> float:
        """
        Compute the learning rate at a given step.

        Parameters
        ----------
        step : int
            Training step.

        Returns
        -------
        float
            Learning rate at the given step.
        """
        # Warmup
        if step < self.steps_warmup:
            return float(step) / float(max(1.0, self.steps_warmup))
        
        # Follow cosine pattern after warmup
        progress = float(step - self.steps_warmup) / float(max(1, self.steps_total - self.steps_warmup))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


def train_evaluate_segmentation_model(settings: dict) -> None:
    """
    Train and/or evaluate a segmentation model.

    Parameters
    ----------
    settings : dict
        Dictionary containing all settings for training and evaluation.
    """

    # Move model to device (GPU if possible)
    model = settings["model"]["model"]
    model = model.to(DEVICE)

    # Move loss function weights to device (GPU if possible)
    if settings["training"]["loss_name"] == "NLL":
        settings["training"]["loss"].weight = settings["training"]["loss"].weight.to(DEVICE)
    elif settings["training"]["loss_name"] == "LabelSmoothingNLL":
        settings["training"]["loss"].weight = settings["training"]["loss"].weight.to(DEVICE)

    # Initialize loss function
    criterion = settings["training"]["loss"]

    # check if model is in training mode
    if settings["training"]["training"] is True:
        # Get optimizer settings
        optimizer = settings["training"]["optimizer"]

        # Get learning rate schedule
        lr_schedule = settings["training"]["lr_schedule"]

        # Get train data loader
        train_dataloader = settings['training']["dataloader"]

        # Initialize variables
        gradient_updates = 0
        finish_training = False
        best_val_Jaccard_score = 0

    else:
        finish_training = True

    # Get evaluation data loader
    val_data_loader = settings['evaluation']["dataloader"]

    # Resize the images (from original 500x500 to 512x512)
    transform = transforms.Compose([transforms.Resize(settings["data"]["input_size"], antialias=True)])
    back_transform = transforms.Compose([transforms.Resize(settings["data"]["original_size"], antialias=True)])

    # Initialize gradient scaler
    scaler = torch.cuda.amp.GradScaler()

    # Train and/or evaluate the model
    for epoch in range(settings["training"]["max_epochs"]):

        # Initialize train loss of current epoch
        train_loss = 0

        if settings["training"]["training"] is True:
            # Set model in training mode
            model.train()

            # Iterate over training batches
            with tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Epoch", position=0) as pbar:
                for batch_idx, (data_train, target, mask, file_name) in pbar:
                                        
                    # Set optimizers gradients to zero
                    optimizer.zero_grad()

                    # Initialize dictionary for training parameters
                    train_params = {}

                    # Move training data and labels to device (GPU if possible)
                    data_train = transform(data_train.to(DEVICE))
                    target = transform(target.to(DEVICE))

                    # Get hard mask from binary segmentation and mask for ignoring pixels (only for multiclass segmentation)
                    if "mutliclass" in settings["data"] and settings["data"]["mutliclass"] is True:
                        # Get hard mask from binary segmentation
                        segmentation = mask[1]

                        # Get mask for ignoring pixels
                        mask = mask[0]

                        # Move masks to device (GPU if possible), apply transformations
                        segmentation = transform(segmentation.to(DEVICE))

                    # Move mask to device (GPU if possible), apply transformations  
                    mask = transform(mask.to(DEVICE))

                    # Use automatic mixed precision for training
                    with torch.autocast(device_type=str(DEVICE), dtype=torch.float16, enabled=settings["model"]["torch_autocast"]):

                        # Forward pass
                        output = model(data_train).squeeze(1)  
                        
                        # Handle multiclass segmentation
                        if "mutliclass" in settings["data"] and settings["data"]["mutliclass"] is True:

                            # Round target to nearest integer
                            target = target.round()
                            target = target.long()

                            # Apply the mask to target and model output (ignoring pixels without valid labels)
                            target[mask.squeeze(1) == 0] = 99
                            output = output * mask
                           
                            # Apply softmax to model output
                            output = torch.nn.functional.softmax(output, dim=1)

                            # Apply hard mask from binary segmentation to get pixels of roads
                            mask_roads = (segmentation == 1).float()

                            # Apply hard mask from binary segmentation to get pixels without roads
                            mask_no_roads = (segmentation == 0).float()

                            # Concatenate mask for pixels without roads and model output
                            expanded_mask_no_roads = mask_no_roads.unsqueeze(1)

                            # Concatenate mask for pixels without roads and model output (probabilities of no road class)
                            output = torch.cat((expanded_mask_no_roads, output), dim=1)

                            # Multiply model output with hard mask for pixels with roads (probabiliites of a road class is zero if hardmask is no road)
                            output[:, 1:, :, :] = output[:, 1:, :, :].clone() * mask_roads.unsqueeze(1)

                            # Replace small values with 0.00001 for numerical stability
                            output = torch.where(output < 0.00001, 0.00001, output)

                            # Calculate the log of the model output (for NLL loss)
                            output = torch.log(output)
                            
                        else:
                            # Apply the mask to target, if mask has value of zero, assign 99 to target
                            target[mask == 0] = 99
                        
                        # Calculate the loss
                        loss = criterion(output, target)
                            
                    if "mutliclass" in settings["data"] and settings["data"]["mutliclass"] is True:
                        # If loss is NaN, skip the current iteration
                        if math.isnan(loss.item()):
                            continue

                    # Calculate training loss, scales and calls ``backward()`` on scaled loss to create scaled gradients.
                    scaler.scale(loss).backward()

                    # Unscale the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], settings["training"]["gradient_clip"])

                    # Update weights
                    scaler.step(optimizer)

                    # Updates the scale for next iteration.
                    scaler.update()

                    # Count number of gradient updates
                    gradient_updates += 1

                    # update learning rate schedule
                    lr_schedule.step()

                    # sum up loss of epoch only if not NaN
                    if not math.isnan(loss.item()):
                        train_loss += loss.item()

                    # Update progress bar
                    train_params["loss"] = loss.item()
                    pbar.set_postfix(train_params)
                    pbar.update()
                        
                    # If maximum number of steps per epoch reached
                    if "subset_train_iterations" in settings["data"].keys():
                        if batch_idx == (settings["data"]["subset_train_iterations"] - 1):
                            break
            
            # Print training loss scores
            print(f'Training: Epoch [{epoch + 1}/{settings["training"]["max_epochs"]}], Loss: {train_loss / len(train_dataloader.dataset)}')

        # Evaluate the model if evaluation is enabled
        if settings["evaluation"]["evaluation"] is True:
           
            # Set model to evaluation mode
            model.eval()
            
            # Set model to monte carlo mode if enabled
            if "MonteCarlo" in settings["evaluation"].keys() and settings["evaluation"]["MonteCarlo"] is True:
                model.set_p()

            # No gradient calculation during evaluation
            with torch.no_grad():

                # Initialize variables
                val_loss = 0
                predictions = np.array([])
                propabilities = np.array([])
                labels = np.array([])

                # Iterate over evaluation data batches
                for batch_idx, (data_val, target, mask, file_name) in enumerate(val_data_loader): 

                    # Use automatic mixed precision for evaluation       
                    with torch.autocast(device_type=str(DEVICE), dtype=torch.float16, enabled=settings["model"]["torch_autocast"]):

                        # Handle multiclass segmentation
                        if "mutliclass" in settings["data"] and settings["data"]["mutliclass"] is True:

                            # Move validation data and labels to device (GPU if possible), apply transformations
                            data_val = transform(data_val.to(DEVICE))
                            target = transform(target.to(DEVICE))

                            # Get hard mask from binary segmentation and mask for ignoring pixels
                            segmentation = mask[1]
                            mask = mask[0]
                            mask = transform(mask.to(DEVICE))
                            segmentation = transform(segmentation.to(DEVICE))

                            # Round target to nearest integer
                            target = target.round()

                            # Forward pass
                            output = model(data_val).squeeze(1)

                            # Apply the mask to target and model output (ignoring pixels without valid labels)
                            output = output * mask
                            target[mask.squeeze(1) == 0] = 99
#
                            # Convert target to long
                            target = target.long()

                            # Handling for DeepEnsemble
                            if settings["model"]["Name"] == "DeepEnsemble":
                                output = output
                            else:
                                output = torch.nn.functional.softmax(output, dim=1)

                            # Apply hard mask from binary segmentation to get pixels of roads
                            mask_roads = (segmentation == 1).float()

                            # Apply hard mask from binary segmentation to get pixels without roads
                            mask_no_roads = (segmentation == 0).float()

                            # Get probabilities of no road class
                            output = torch.cat((mask_no_roads, output), dim=1)

                            # Multiply model output with hard mask for pixels with roads (probabiliites of a road class is zero if hardmask is no road)
                            output[:, 1:, :, :] = output[:, 1:, :, :].clone() * mask_roads

                            # varriable to store the softmax output
                            output_softmax = output

                            # Replace small values with 0.00001 for numerical stability
                            output_logsoftmax = torch.where(output < 0.00001, 0.00001, output)

                            # Calculate the log of the model output (for NLL loss)
                            output_logsoftmax = torch.log(output_logsoftmax)

                            # NLL loss
                            criterion_val = nn.NLLLoss(ignore_index=99)

                            # Calculate the loss, store the loss in val_loss
                            val_loss += criterion_val(output_logsoftmax, target).item()

                            # Get predicted class
                            prediction = torch.argmax(output, dim=1)
                            prediction.cpu().numpy()[0]
                            mask = mask.squeeze(1)

                        # Handle binary segmentation
                        else:
                            # Move validation data and labels to device (GPU if possible)
                            data_val = transform(data_val.to(DEVICE))
                            target = transform(target.to(DEVICE)).squeeze(1)
                            if "breakpoint_prediction" in settings["data"] and settings["data"]["breakpoint_prediction"] is True:
                                mask = mask[0].squeeze(1)
                            mask = transform(mask.to(DEVICE)).squeeze(1)

                            # Forward pass
                            output = model(data_val).squeeze(1)
                            
                            # Handling for Monte Carlo dropout
                            if "Ensemble" in settings["evaluation"].keys() and settings["evaluation"]["Ensemble"] is True:
                                output_softmax = output
                            
                            # Handling for other models
                            else:
                                softmax = nn.Sigmoid()
                                output_softmax = softmax(output)

                            # Apply the mask to target and model output (ignoring pixels without valid labels)
                            target[mask == 0] = 99
                            output = output * mask

                            # Get index of predicted class (if probability is above threshold)
                            prediction = torch.where(output_softmax > settings["evaluation"]["threshold"], torch.ones_like(output_softmax), torch.zeros_like(output_softmax))

                            # Calculate the loss
                            val_loss += criterion(output, target).item()

                        # Functionality for plotting predictions if enabled
                        if settings["evaluation"]["plots_samples_last_epoch"] is True and epoch == settings["training"]["max_epochs"] - 1:
                            # Denormalize images (RGB values from 0-255)
                            image = data_val[0].permute(1, 2, 0).cpu().numpy()  
                            for i in range(3): 
                                image[:, :, i] = (image[:, :, i] * settings["data"]["channel_std"][i]) + settings["data"]["channel_mean"][i]
                            image = np.clip(image.astype(np.uint8), 0, 255)

                            # Plot images
                            plt.imshow(image)
                            plt.title("Input Image")
                            plt.show()
                            plt.imshow(output.cpu().numpy()[0])
                            plt.title("Raw Prediction")
                            plt.show()
                            plt.title("Softmax Prediction")
                            plt.imshow(output_softmax.detach().cpu().numpy()[0])
                            plt.show()
                            plt.title("Binary Prediction")
                            plt.imshow(prediction.cpu().numpy()[0])
                            plt.show()
                            plt.imshow(target.cpu().numpy()[0])
                            plt.title("Ground Truth")
                            plt.show()

                        # Save the output if enabled (hard predictions)
                        if settings["evaluation"]["save_output"] is True:
                            # save the output
                            save_output(prediction, back_transform, settings, file_name)

                        # Save the softmax output if enabled (soft predictions)
                        if "save_softmax" in settings["evaluation"].keys() and settings["evaluation"]["save_softmax"] is True:
                            # Save softmax output for each class separately
                            for j in range(0, settings["model"]["num_classes"]):
                                save_output(output_softmax[:, j], back_transform, settings, file_name, category=f"_class{j}")
                            
                        # Crop predictions due to overlapping images
                        if settings["evaluation"]["crop"] is True:
                            # Crop mask from 512*512 to 256*256 (input images are overlapping) to get reliable evaluation metrics
                            width_min = int((settings["data"]["input_size"][0] - settings["data"]["input_size"][0]*0.5) / 2)
                            width_max = int((settings["data"]["input_size"][0] + settings["data"]["input_size"][0]*0.5) / 2)
                            height_min = int((settings["data"]["input_size"][1] - settings["data"]["input_size"][1]*0.5) / 2)
                            height_max = int((settings["data"]["input_size"][1] + settings["data"]["input_size"][1]*0.5) / 2)
                            prediction = prediction[:, width_min: width_max, height_min: height_max]
                            target = target[:, width_min: width_max, height_min: height_max]
                            mask = mask[:, width_min: width_max, height_min: height_max]

                            # Handle multiclass segmentation and binary segmentation
                            if "mutliclass" in settings["data"] and settings["data"]["mutliclass"] is True:
                                output_softmax = output_softmax[:, :, width_min: width_max, height_min: height_max]
                            else:
                                output_softmax = output_softmax[:, width_min: width_max, height_min: height_max]

                        # Filter out pixels without valid labels
                        mask = mask.bool()
                        prediction = prediction[mask]
                        target = target[mask]

                        # Handle multiclass segmentation
                        if "mutliclass" in settings["data"] and settings["data"]["mutliclass"] is True:
                            # Reshape the tensors (for calculating metrics)
                            mask = mask.unsqueeze(1).repeat_interleave(6, dim = 1)
                            output_softmax = output_softmax.permute(1, 0, 2, 3) 
                            mask = mask.permute(1, 0, 2, 3)
                            output_softmax = output_softmax.reshape(6, -1)
                            mask = mask.reshape(6, -1)
                            sum = output_softmax.sum(dim=0)
                            output_softmax = output_softmax[mask]
                            output_softmax = output_softmax.reshape(6,-1)
                
                        else:
                            output_softmax = output_softmax[mask]
                        
                    # Append labels and predictions to list
                    labels = np.concatenate((labels, np.round(target.cpu().numpy().flatten()).astype(int)))
                    predictions = np.concatenate((predictions, prediction.cpu().numpy().flatten()))

                    # Append propabilities to list
                    if "mutliclass" in settings["data"] and settings["data"]["mutliclass"] is True:
                        if batch_idx == 0:
                            propabilities = output_softmax.cpu().numpy()
                        else:
                            propabilities = np.concatenate((propabilities, output_softmax.cpu().numpy()), axis=1)

                    else:
                        propabilities = np.concatenate((propabilities, output_softmax.cpu().numpy().flatten()))

                    # Break if maximum number of steps per epoch is reached
                    if "subset_evaluation_iterations" in settings["data"].keys():
                        if batch_idx == (settings["data"]["subset_evaluation_iterations"] - 1):
                            break

                    # Finish training if maximum number of epochs is reached
                    if epoch == settings["training"]["max_epochs"] - 1:
                        finish_training = True

            # Calculate metrics    
            # Handle multiclass segmentation
            if "mutliclass" in settings["data"] and settings["data"]["mutliclass"] is True:
                # IoU
                Jaccard = torchmetrics.classification.MulticlassJaccardIndex(settings["model"]["num_classes"], average='macro')

                # Classification report
                #print(classification_report(labels, predictions, target_names=["No road", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5" ], digits=4))

                # Confusion matrix
                #print(confusion_matrix(labels, predictions, normalize="true").diagonal())

                # Brier score
                brier_score = torch.sum((torch.Tensor(propabilities) - torch.nn.functional.one_hot(torch.Tensor(np.round(labels)).to(torch.int64)).permute(1,0))**2) / (val_data_loader.dataset.__len__()*512*512) 

            # Handle binary segmentation
            else:
                # IoU
                Jaccard = torchmetrics.classification.BinaryJaccardIndex()

            # Calculate IoU
            Jaccard_score = Jaccard(torch.Tensor(predictions), torch.Tensor(labels))

            # Calculate accuracy
            accuracy = accuracy_score(labels, predictions)

            # Calculate F1
            f1 = f1_score(labels, predictions, average='macro')

            # Calculate precision
            precision = precision_score(labels, predictions, average='macro')

            # Calculate recall
            recall = recall_score(labels, predictions, average='macro')

            # Calculate average confidence
            # Handle multiclass segmentation
            if "mutliclass" in settings["data"] and settings["data"]["mutliclass"] is True:
                indices = np.argmax(propabilities, axis=0)
                propabilities = propabilities[indices, np.arange(propabilities.shape[1])]

            # Handle binary segmentation
            else:
                propabilities = np.maximum(propabilities, 1 - propabilities)
            propabilities = np.mean(propabilities)

            # Print loss scores 
            print(f'Validation: Epoch [{epoch + 1}/{settings["training"]["max_epochs"]}], Loss: {val_loss / len(val_data_loader.dataset)}, Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}, IoU: {Jaccard_score}, Average confidence: {propabilities}, Brier {brier_score}')
        
        # Log the los and metrics to tensorboard
        if settings["data"]["tensorboard"] is True:
            settings["data"]["tensorboard_writer"].add_scalar('Training loss', train_loss / len(train_dataloader), epoch + 1)
            settings["data"]["tensorboard_writer"].add_scalar('Validation loss', val_loss / len(val_data_loader), epoch + 1)
            settings["data"]["tensorboard_writer"].add_scalar('Validation Accuracy', accuracy, epoch + 1)
            settings["data"]["tensorboard_writer"].add_scalar('Validation F1', f1, epoch + 1)
            settings["data"]["tensorboard_writer"].add_scalar('Validation Precision', precision, epoch + 1)
            settings["data"]["tensorboard_writer"].add_scalar('Validation Recall', recall, epoch + 1)
            settings["data"]["tensorboard_writer"].add_scalar('Validation IoU', Jaccard_score, epoch + 1)
            settings["data"]["tensorboard_writer"].close()

        # Save the model (based on early stopping on evaluation IoU) if early stopping is enabled   
        if settings["training"]["training"] is True and settings["training"]["early_stopping"] is True:

            # Save the model if the IoU is better than the best IoU
            if Jaccard_score > best_val_Jaccard_score:
                # update best IoU
                best_val_Jaccard_score = Jaccard_score

                # Get the model name
                model_name = settings["data"]["result_file_name"]

                # Get name for saving the model
                save_name = f"{model_name}.pt"

                # Get the save path
                save_path = const.MODEL_STORAGE_DIR.joinpath(save_name)

                # Save the model
                model_state_dict = model.state_dict()
                const.MODEL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(model_state_dict, save_path)
                print(f"Model saved to {save_path} with accuracy {accuracy} after {gradient_updates} iterations")

        # Check if maximum number of epochs is reached
        if finish_training:

            # Save the model if early stopping is disabled
            if settings["training"]["training"] is True and settings["training"]["early_stopping"] is False:
                # Get the model name
                model_name = settings["data"]["result_file_name"]

                # Get name for saving the model
                save_name = f"{model_name}.pt"

                # Get the save path
                save_path = const.MODEL_STORAGE_DIR.joinpath(save_name)

                # Save the model
                model_state_dict = model.state_dict()
                const.MODEL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(model_state_dict, save_path)
                print(f"Model saved to {save_path} with accuracy {accuracy} after {gradient_updates} iterations")

            # Break the loop if maximum number of epochs is reached. Training and/or evaluation is finished.
            break
        