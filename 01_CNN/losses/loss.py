##!/usr/bin/env python

"""
Functions of the loss functions used for model training.
"""

### IMPORTS ###
import torch
import torch.nn as nn
import torch.nn.functional as F


### FUNCTIONS ###
def linear_combination(x, y, epsilon):
    """
    Linearly combine two tensors, for calculating the label smoothing cross entropy loss.

    Parameters
    ----------
    x : torch.Tensor
        A tensor representing the smoothed loss.
    y : torch.Tensor
        A tensor representing the cross entropy loss.
    epsilon : float
        A float representing the smoothing factor. 

    Returns
    -------
    torch.Tensor
        Smoothed loss.
    """
    return epsilon * x + (1 - epsilon) * y


### CLASSES ###
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss. Label smoothing is a regularization technique that introduces noise for the labels.
    This technique is used to prevent the model from becoming too confident about its predictions, which can lead to overfitting
    and bad model calibration.
    """
    def __init__(self, weight, epsilon = 0.0, reduction='mean', ignore_index=99):
        """
        Initialize the loss function

        Parameters
        ----------
        weight : torch.Tensor
            A tensor representing the weights for each class. The weights are used to weight the loss for each class.
        epsilon : float
            A float representing the smoothing factor. The smoothing factor is used to smooth the labels. The smoothing factor
            determines how much probability mass is moved from the true label to the other labels. Default is 0.0.
        reduction : str
            A string representing the reduction method. The reduction method is used to reduce the loss to a scalar value. Default is 'mean'.
        ignore_index : int
            An integer representing the index of the label to ignore. Default is 99.
        """
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, log_preds, target):
        """
        Compute the loss

        Parameters
        ----------
        log_preds : torch.Tensor
            A tensor representing the log probabilities of the model's predictions.
        target : torch.Tensor
            A tensor representing the true labels.

        Returns
        -------
        torch.Tensor
            A tensor representing the loss.
        """

        # Compute the number of classes 
        n = log_preds.size()[1]

        # Create a mask for ignore_index
        ignore_mask = (target != self.ignore_index).float()

        # Compute the smoothed loss
        loss = -log_preds.sum(dim=1)

        # Apply the mask to the smoothed loss
        loss = loss * ignore_mask

        # Flatten last three dimensions
        loss = loss.mean()

        # Compute the nll loss
        nll = F.nll_loss(log_preds, target, reduction='none', ignore_index=self.ignore_index, weight=self.weight).mean()

        # Return smoothed loss
        return linear_combination(loss / n, nll, self.epsilon)
        

class MulticlassDiceLoss(nn.Module):
    """
    Multiclass Dice Loss. 
    
    Reference: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss
    """

    def __init__(self, num_classes, softmax_dim=None):
        """
        Initialize the loss function

        Parameters
        ----------
        num_classes : int
            An integer representing the number of classes.
        softmax_dim : int
            An integer representing the dimension to apply the softmax function. Default is None.
        """
        super().__init__()
        self.num_classes = num_classes
        self.softmax_dim = softmax_dim

    def forward(self, logits, targets, reduction='mean', smooth=1e-6):
        probabilities = logits

        if self.softmax_dim is not None:
            probabilities = nn.Softmax(dim=self.softmax_dim)(logits)
      
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)

        intersection = (targets_one_hot * probabilities).sum()

        mod_a = intersection.sum()
        mod_b = targets.numel()

        dice_coefficient = 2. * intersection / (mod_a + mod_b + smooth)
        dice_loss = -dice_coefficient.log()
        return dice_loss
    

class DiceLoss(nn.Module):
    """
    Dice Loss.

    Reference: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss
    """
    def __init__(self, weight=None, size_average=True):
        """
        Initialize the loss function

        Parameters
        ----------
        weight : torch.Tensor
            A tensor representing the weights for each class. The weights are used to weight the loss for each class.
        size_average : bool
            A boolean representing whether to average the loss. Default is True.
        """
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        """
        Compute the loss

        Parameters
        ----------
        inputs : torch.Tensor
            A tensor representing the model's predictions (logits).
        targets : torch.Tensor
            A tensor representing the true labels.
        smooth : float
            A float representing the smoothing factor to prevent devision by zero. Default is 1.

        Returns
        -------
        torch.Tensor
            A tensor representing the loss.
        """

        # Apply sigmoid activation function
        sigmoid = nn.Sigmoid()
        inputs = sigmoid(inputs)     
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Mask out labels with value 99 (ignore_index)
        mask = targets != 99
        inputs = inputs[mask]
        targets = targets[mask]

        # Compute the Dice loss
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        return 1 - dice
    

class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) Loss.

    Reference: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#IoU-Loss
    """
    def __init__(self, weight=None, size_average=True):
        """
        Initialize the loss function

        Parameters
        ----------
        weight : torch.Tensor
            A tensor representing the weights for each class. The weights are used to weight the loss for each class.
        size_average : bool
            A boolean representing whether to average the loss. Default is True.
        """
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        """
        Compute the loss

        Parameters
        ----------
        inputs : torch.Tensor
            A tensor representing the model's predictions (logits).
        targets : torch.Tensor
            A tensor representing the true labels.
        smooth : float
            A float representing the smoothing factor to prevent devision by zero. Default is 1.

        Returns
        -------
        torch.Tensor
            A tensor representing the loss.
        """     
        
        # Apply sigmoid activation function
        sigmoid = nn.Sigmoid()
        inputs = sigmoid(inputs)       
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Intersection is equivalent to True Positive count
        # Union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        # Compute the IoU loss
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU
    

class DiceBCELoss(nn.Module):
    """
    Dice Loss combined with Binary Cross Entropy Loss.

    Reference: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Dice-BCE-Loss
    """
    def __init__(self, weight=None, size_average=True):
        """
        Initialize the loss function

        Parameters
        ----------
        weight : torch.Tensor
            A tensor representing the weights for each class. The weights are used to weight the loss for each class.
        size_average : bool
            A boolean representing whether to average the loss. Default is True.
        """
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        """
        Compute the loss

        Parameters
        ----------
        inputs : torch.Tensor
            A tensor representing the model's predictions (logits).
        targets : torch.Tensor
            A tensor representing the true labels.
        smooth : float
            A float representing the smoothing factor to prevent devision by zero. Default is 1.

        Returns
        -------
        torch.Tensor
            A tensor representing the loss.
        """
        
        # Apply sigmoid activation function
        sigmoid = nn.Sigmoid()
        inputs = sigmoid(inputs)      
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Compute the Dice loss
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE