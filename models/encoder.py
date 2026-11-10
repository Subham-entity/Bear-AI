import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_dim, fine_tune=True):
        super(EncoderCNN, self).__init__()
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Freeze all backbone parameters by default
        for param in resnet.parameters():
            param.requires_grad = False
            
        # Unfreeze layer4 so it can adapt to Flickr8k visual features
        if fine_tune:
            for param in resnet.layer4.parameters():
                param.requires_grad = True

        # Strip avgpool and fc layers
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(2048, embed_dim)

    def forward(self, images):
        # Input images shape: (Batch_Size, 3, 224, 224)
        
        # Pass through convolutional backbone -> Shape: (Batch_Size, 512, 7, 7)
        features = self.resnet(images)
        
        # Reshape spatial dimensions for Transformer compatibility:
        # (Batch, 512, 7, 7) -> (Batch, 512, 49) -> (Batch, 49, 512)
        batch_size, channels, height, width = features.shape
        features = features.view(batch_size, channels, height * width).permute(0, 2, 1)
        
        # Project spatial feature vectors to match Transformer embedding dimension
        # Output shape: (Batch_Size, 49, Embed_Dim)
        features = self.linear(features)
        
        return features