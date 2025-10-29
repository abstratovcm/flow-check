import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimeFourier(nn.Module):
    """
    Map a scalar t∈[0,1] → a nonlinear vector via sin/cos frequencies.
    """
    def __init__(self, n_freq: int = 8, out_dim: int = 256):
        super().__init__()
        self.n_freq = n_freq
        self.fc = nn.Linear(2 * n_freq, out_dim)
        self.act = nn.GELU()
        freqs = (2.0 ** torch.arange(n_freq).float()) * math.pi
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor):
        x = t.repeat(1, self.n_freq) * self.freqs.unsqueeze(0)
        sin_emb = torch.sin(x)
        cos_emb = torch.cos(x)
        fourier = torch.cat([sin_emb, cos_emb], dim=1)
        return self.act(self.fc(fourier))

class VisionTransformerDiscreteFlow(nn.Module):
    """
    Vision Transformer based discrete flow model for chess boards.
    """
    def __init__(
        self,
        squares: int = 64,
        vocab_size: int = 13,
        label_classes: int = 3,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        ff_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.squares = squares
        self.vocab_size = vocab_size
        self.label_classes = label_classes
        self.d_model = d_model

        self.emb_board = nn.Embedding(vocab_size, d_model)
        self.time_fourier = TimeFourier(n_freq=8, out_dim=d_model)
        self.label_mlp    = nn.Sequential(
            nn.Linear(label_classes, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model)
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.empty(1, squares + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.norm = nn.LayerNorm(d_model)

        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, vocab_size)
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, label: torch.Tensor):
        B = x_t.size(0)

        board_tokens = self.emb_board(x_t)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        tokens = torch.cat((cls_tokens, board_tokens), dim=1)
        tokens = tokens + self.pos_embed

        t_col      = t.unsqueeze(-1)
        t_feat     = self.time_fourier(t_col)
        label_oneh = F.one_hot(label, num_classes=self.label_classes).float()
        lbl_feat   = self.label_mlp(label_oneh)
        tokens[:, 0, :] = tokens[:, 0, :] + t_feat + lbl_feat

        encoded = self.transformer(tokens)
        encoded = self.norm(encoded)

        board_encoded = encoded[:, 1:, :]
        logits = self.mlp_head(board_encoded)
        return logits
