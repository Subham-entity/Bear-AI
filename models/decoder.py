import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (Batch, Seq_Len, d_model)
        return x + self.pe[:, :x.size(1)]
    

class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, hidden_dim=1024, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, captions, encoder_features):
        # captions shape: (batch, seq_len)
        seq_len = captions.size(1)
        device = captions.device
        
        # Generate causal mask to prevent looking into the future
        tgt_mask = self.generate_square_subsequent_mask(seq_len, device)
        
        embeddings = self.embedding(captions) # (batch, seq_len, embed_dim)
        
        # TransformerDecoder expects memory shape: (batch, num_patches, embed_dim)
        # and tgt shape: (batch, seq_len, embed_dim)
        output = self.transformer_decoder(
            tgt=embeddings, 
            memory=encoder_features, 
            tgt_mask=tgt_mask
        )
        
        logits = self.fc_out(output)
        return logits