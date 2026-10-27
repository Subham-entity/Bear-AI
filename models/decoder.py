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
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers=2, max_len=100):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        
        # Transformer Decoder Layer handles self-attention on text + cross-attention on image features
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final linear classification head to map hidden states back to vocabulary tokens
        self.fc = nn.Linear(embed_dim, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        # Causal mask to prevent the decoder from peeking into future tokens during training
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, captions, encoder_features):
        # captions shape: (Batch, Seq_Len)
        # encoder_features shape: (Batch, 49, Embed_Dim)
        
        seq_len = captions.size(1)
        
        # 1. Embed tokens and add positional encodings
        x = self.embed(captions) * math.sqrt(self.embed.embedding_dim)
        x = self.pos_encoding(x)
        
        # 2. Generate causal mask for text autoregression
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(captions.device)
        
        # 3. Pass through Transformer Decoder
        # memory argument takes the image features (acting as Keys and Values)
        output = self.transformer_decoder(
            tgt=x, 
            memory=encoder_features, 
            tgt_mask=tgt_mask
        )
        
        # 4. Project back to vocabulary logits
        logits = self.fc(output)
        return logits