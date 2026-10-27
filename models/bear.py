import torch
import torch.nn as nn
from models.encoder import EncoderCNN
from models.decoder import DecoderTransformer

class Bear(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers=2):
        super().__init__()
        self.encoder = EncoderCNN(embed_dim)
        self.decoder = DecoderTransformer(vocab_size, embed_dim, num_heads, hidden_dim, num_layers)

    def forward(self, images, captions):
        # 1. Extract spatial feature grids from images -> (Batch, 49, Embed_Dim)
        encoder_features = self.encoder(images)
        
        # 2. Pass captions and image features through decoder -> (Batch, Seq_Len, Vocab_Size)
        logits = self.decoder(captions, encoder_features)
        
        return logits