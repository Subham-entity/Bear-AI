import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_dim):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Freeze all backbone parameters by default
        for param in resnet.parameters():
            param.requires_grad = False
            
        # Unfreeze BOTH layer3 and layer4 for better visual adaptation
        for param in resnet.layer3.parameters():
            param.requires_grad = True
        for param in resnet.layer4.parameters():
            param.requires_grad = True

        modules = list(resnet.children())[:-2] # Keeps (Batch, 2048, 7, 7) feature grid
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(2048, embed_dim)

    def forward(self, images):
        features = self.resnet(images) # (batch, 2048, 7, 7)
        batch_size, channels, height, width = features.shape
        features = features.view(batch_size, channels, height * width).permute(0, 2, 1) # (batch, 49, 2048)
        return self.linear(features)