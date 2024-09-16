##!/usr/bin/env python

"""
Implementation of the models for semantic segmentation.
"""

### IMPORTS ###
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
import const

### U-Net-small (baseline for binary road segmentation) ###
class UNet_small(nn.Module):
    """
    A PyTorch neural network model implementing the UNet_small architecture. Tiny U-Net with maxpooling as downsampling,
    and upsampling with transposed convolution. Leakly ReLU activation and batch normalization are used.
    """
    def __init__(self, in_channels):
        """
        Parameters
        ----------
        in_channels: int
            The number of input channels.
        out_channels: int
            The number of output channels.
        """
        super(UNet_small, self).__init__()

        # Encoding
        self.encoder1 = UNetEncoder(in_channels, 8)
        self.encoder2 = UNetEncoder(8, 8)
        self.maxpool = nn.MaxPool2d(2, stride=2) # 512 -> 256

        self.encoder3 = UNetEncoder(8, 16)
        self.encoder4 = UNetEncoder(16, 16)
        self.maxpool = nn.MaxPool2d(2, stride=2) # 256 -> 128

        self.encoder5 = UNetEncoder(16, 32)
        self.encoder6 = UNetEncoder(32, 32)
        self.maxpool = nn.MaxPool2d(2, stride=2) # 128 -> 64
        
        self.encoder7 = UNetEncoder(32, 64)
        self.encoder8 = UNetEncoder(64, 64)

        # Decoding
        self.decoder1 = UNetDecoder(64, 32) # 64 -> 128
        self.decoder1_conv = UNetEncoder(64, 32) 
        
        self.decoder2 = UNetDecoder(32, 16) # 128 -> 256
        self.decoder2_conv = UNetEncoder(32, 16) 
        
        self.decoder3 = UNetDecoder(16, 8) # 256 -> 512

        # Head
        self.decoder3_conv1 = UNetEncoder(16, 16)
        self.decoder3_conv2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoding
        x = self.encoder1(x)
        x1 = self.encoder2(x)
        x = self.maxpool(x1)
    
        x = self.encoder3(x)
        x2 = self.encoder4(x)
        x = self.maxpool(x2)
        
        x = self.encoder5(x)
        x3 = self.encoder6(x)
        x = self.maxpool(x3)
        
        x = self.encoder7(x)
        x = self.encoder8(x)
        
        # Decoding
        x = self.decoder1(x)
        x = torch.cat((x, x3), dim=1)
        x = self.decoder1_conv(x)
        
        x = self.decoder2(x)
        x = torch.cat((x, x2), dim=1)
        x = self.decoder2_conv(x)
        
        x = self.decoder3(x)
        x = torch.cat((x, x1), dim=1)

        # Head
        x = self.decoder3_conv1(x)
        x = self.decoder3_conv2(x)

        return x
    

class UNetEncoder(nn.Module):
    """
    A PyTorch neural network modul implementing the UNet encoder.

    Parameters
    ----------
    in_channels: int
        The number of input channels.
    out_channels: int
        The number of output channels.

    Methods
    ----------
    forward(x)
        Forward pass through the network.

    Returns
    -------
    UNet
        A PyTorch neural network enocder.
    """
    def __init__(self, in_channels, out_channels):
        super(UNetEncoder, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class UNetDecoder(nn.Module):
    """
    A PyTorch neural network modul implementing the UNet decoder.

    Parameters
    ----------
    in_channels: int
        The number of input channels.
    out_channels: int
        The number of output channels.

    Methods
    ----------
    forward(x)
        Forward pass through the network.

    Returns
    -------
    UNet
        A PyTorch neural network decoder.
    """
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, padding=0, stride=2)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
    

### Attention Res-U-Net (binary road segmentation) ###
class Resnet18_Decoder(nn.Module):
    """
    A PyTorch neural network modul implementing the Resnet18 decoder. Two layers (convolutional and transposed convolutional) are used.
    ReLU activation and batch normalization are used, together with dropout with probability p.
    """
    def __init__(self, in_channel, mid_channel, out_channel, p=0.3):
        """
        Initialize the Resnet18 decoder.

        Parameters
        ----------
        in_channel: int
            The number of input channels.
        mid_channel: int
            The number of channels in the middle layer.
        out_channel: int
            The number of output channels.
        """
        super(Resnet18_Decoder, self).__init__()
        self.p = p
        self.layer = nn.Sequential(nn.Dropout2d(p=self.p),
                                    nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(mid_channel),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout2d(p=self.p),
                                    nn.ConvTranspose2d(mid_channel, out_channel, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(out_channel),
                                    nn.ReLU(inplace=True)
                                    )
       
    def forward(self, x):
       x = self.layer(x)
       return x


class AttentionBlock(nn.Module):

    def __init__(self, F_g, F_l, n_coefficients):
        """
        Initialize the AttentionBlock.

        Parameters
        ----------
        F_g: int
            The number of feature maps (channels) in the previous layer.
        F_l: int
            The number of feature maps in the corresponding encoder layer, transferred via skip connection.
        n_coefficients: int
            The number of learnable multi-dimensional attention coefficients.
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        Parameters
        ----------
        gate: tensor
            The gating signal from the previous layer.
        skip_connection: tensor
            The activation from the corresponding encoder layer.
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out
    

class Resnet18_Unet_Big_Attention(nn.Module):
    """
    A PyTorch neural network model implementing the Res-U-Net architecture.
    """
    def __init__(self, n_classes, imagenet1k=True, p = 0.3):
        """
        Initialize the Resnet18_Unet_Big_Attention.

        Parameters
        ----------
        n_classes: int
            The number of output classes.
        imagenet1k: bool
            Whether to use the pretrained model.
        """
        super(Resnet18_Unet_Big_Attention, self).__init__()
        self.p = p
        
        # Get ResNet18 model
        if imagenet1k:
            # get pretrained resnet model
            print("imagenet1k", imagenet1k)
            self.encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            # not pretrained
            print("imagenet1k", imagenet1k)
            self.encoder = models.resnet18(weights=None)
        
        # Layer operating at 256x256
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = nn.Sequential(nn.Dropout2d(p=self.p),
                            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(32), 
                            nn.ReLU(inplace=True),
                            )

        # Encoder network (ResNet18)
        self.conv1 = nn.Sequential(self.encoder.conv1, 
                                   self.encoder.bn1,
                                   self.encoder.relu
                                   ) 
        self.conv2 = self.encoder.layer1 
        self.conv3 = self.encoder.layer2 
        self.conv4 = self.encoder.layer3 
        self.conv5 = self.encoder.layer4 
                                           
        # Decoder network
        self.decoder1 = Resnet18_Decoder(512, 256, 256, p = self.p)
        self.att1 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.decoder2 = Resnet18_Decoder(256+256, 128, 128, p = self.p)
        self.att2 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.decoder3 = Resnet18_Decoder(128+128, 64, 64, p = self.p)
        self.att3 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.decoder4 = Resnet18_Decoder(64+64, 32, 32, p = self.p)
        self.att4 = AttentionBlock(F_g=32, F_l=32, n_coefficients=16)
        self.decoder5 = Resnet18_Decoder(64, 32, 32, p = self.p)

        # Head
        self.final1 = nn.Sequential(nn.Dropout2d(p = self.p),
                                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False), 
                                    nn.BatchNorm2d(32), 
                                    nn.ReLU(inplace=True)
                                    )
        self.final2 = nn.Conv2d(in_channels=32, out_channels=n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        

    def forward(self, x):
        
        # Encoder
        conv1 = self.conv1(x) # 256x256
        conv1_1 = self.conv1_1(conv1)
        pool = self.pool(conv1)
        conv2 = self.conv2(pool) # 128x128
        conv3 = self.conv3(conv2) # 64x64
        conv4 = self.conv4(conv3) # 32x32
        conv5 = self.conv5(conv4) # 16x16

        # Decoder
        dec1 = self.decoder1(conv5) # 32x32
        skip1 = self.att1(gate=dec1, skip_connection=conv4)
        dec2 = self.decoder2(torch.cat([dec1, skip1], 1)) # 64x64
        skip2 = self.att2(gate=dec2, skip_connection=conv3)
        dec3 = self.decoder3(torch.cat([dec2, skip2], 1)) # 128x128
        skip3 = self.att3(gate=dec3, skip_connection=conv2)
        dec4 = self.decoder4(torch.cat([dec3, skip3], 1)) # 256x256
        skip4 = self.att4(gate=dec4, skip_connection=conv1_1)
        dec5 = self.decoder5(torch.cat([dec4, skip4], 1)) # 512 x 512

        # Head
        final1 = self.final1(dec5) # 512 x 512
        final2 = self.final2(final1) # 512 x 512

        return final2


### Deep Ensemble (road classification) ###
class DeepEnsemble(nn.Module):
    """
    A PyTorch neural network model implementing the Deep Ensemble architecture.
    The model uses a list of pre-trained models and averages the predictions over all models.
    """
    def __init__(self, n_classes, model_list):
        """
        Initialize the Deep Ensemble model.

        Parameters
        ----------
        n_classes: int
            The number of output classes.
        model_list: list
            The list of pre-trained models.
        """
        super(DeepEnsemble, self).__init__()
        self.model_list = model_list
        self.n_members = len(model_list)
        self.models = []

        # Load all models
        for model_name in model_list:
            model = Resnet18_Unet_Big_Classification(n_classes, True)
            path = const.MODEL_STORAGE_DIR.joinpath(model_name + ".pt")
            model.load_state_dict(torch.load(path))
            self.models.append(model)
   
    def load_state_dict(self, state_dict):
        """
        Load the state dictionary for a ensemble member.

        Parameters
        ----------
        state_dict: dict
            The state dictionary.
        """
        self.model.load_state_dict(state_dict)

    def forward(self, x):
        predictions = []

        # Make predictions for each ensemble member
        for model in self.models:
            # Move model to GPU and set to evaluation mode
            model.to('cuda')  
            model.eval()  
            
            # No need to compute gradients for inference. 
            with torch.no_grad():  
                # Make prediction
                out = model(x)  

                # Apply softmax to get class probabilities
                prediction = F.softmax(out, dim=1)

            # Store prediction
            predictions.append(prediction) 

            # Move model back to CPU (to avoid memory issues)
            model.to('cpu')  

        # Stack all predictions
        prediction = torch.stack(predictions)

        # Average predictions over all ensemble members
        prediction = torch.mean(prediction, dim=0)

        return prediction
    

### Res-U-Net for road classification ###
class Resnet18_Unet_Big_Classification(nn.Module):
    """
    A PyTorch neural network model implementing the Res-U-Net architecture for road classification.
    The model is based on the binary Res-U-Net model, but with a different head for classification.
    """
    def __init__(self, n_classes, imagenet1k=True, pretrained_path=None):
        """
        Initialize the Resnet18_Unet_Big_Classification.

        Parameters
        ----------
        n_classes: int
            The number of output classes.
        imagenet1k: bool
            Whether to use the pretrained model.
        pretrained_path: str
            The path to the pre-trained model.
        """
        super(Resnet18_Unet_Big_Classification, self).__init__()

        # Initalze Res-U-Net model from binary road segmentation
        self.resnet18_unet_big = Resnet18_Unet_Big_Attention(1, imagenet1k, p = 0.0)

        # Load pretrained weights
        if pretrained_path is not None:
            self.resnet18_unet_big.load_state_dict(torch.load(pretrained_path))

        # Get number of classes (no road class is omitted, since hard mask is used for road classification)
        classes = n_classes -1

        # Replace last layer for classification
        self.resnet18_unet_big.final2 = nn.Conv2d(in_channels=32, out_channels=classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        return self.resnet18_unet_big(x)
