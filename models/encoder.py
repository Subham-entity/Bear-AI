import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_dim):
        super(EncoderCNN, self).__init__()
        
        # Load pre-trained ResNet-18 with default weights
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Strip away the final average pooling and fully connected layers 
        # (keeping all convolutional feature extraction blocks up to layer4)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        # Linear projection layer to match the Transformer's hidden dimension (d_model)
        # ResNet-18 layer4 outputs 512 feature channels
        self.linear = nn.Linear(512, embed_dim)

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