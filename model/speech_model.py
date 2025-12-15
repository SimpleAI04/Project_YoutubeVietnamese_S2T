import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNNSubsampling(nn.Module):
    def __init__(self, n_mels, d_model, dropout=0.1):
        super(CNNSubsampling, self).__init__()

        self.input_norm = nn.InstanceNorm2d(1)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )

        self.out_dim = 2560

        self.linear = nn.Linear(self.out_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.conv(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, t, c * f)
        x = self.linear(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

class SpeechRecognitionModel(nn.Module):
    def __init__(self, n_mels, n_class, d_model=256, nhead=4, num_layers=4, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        
        self.cnn = CNNSubsampling(n_mels, d_model, dropout)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, n_class)

    def get_new_lengths(self, input_lengths):
        return (input_lengths + 1) // 2

    def forward(self, x, input_lengths):
        x = self.cnn(x) 
        enc_lengths = self.get_new_lengths(input_lengths)
        
        max_len = x.size(1)
        src_key_padding_mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= enc_lengths.unsqueeze(1)
        
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        x = self.classifier(x)
        x = F.log_softmax(x, dim=-1)
        return x, enc_lengths