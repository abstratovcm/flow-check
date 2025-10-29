import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

PIECE_ORDER = ".PNBRQKpnbrqk"

def bb_to_grid(bitboard: int) -> torch.Tensor:
    """
    Turn a 64-bit int into a (64,) tensor of 0/1 bits.
    LSB → square 0 (a1), MSB → square 63 (h8).
    """
    bits = [(bitboard >> i) & 1 for i in range(64)]
    return torch.tensor(bits, dtype=torch.long)


class ChessBoardDataset(Dataset):
    """
    Dataset of bitboard positions + outcome label.

      - x: [64] long tensor with values 0(empty) or 1-12 (PNBRQKpnbrqk)
      - y: 0=equal, 1=black winning, 2=white winning
    """
    def __init__(self,
                 parquet_file: str,
                 labels: list[str],
                 label_column: str = "label"):
        square_cols = [f"sq_{i}" for i in range(64)]
        df = pd.read_parquet(parquet_file,
                             columns=square_cols + [label_column])

        LABEL_MAPPING = {name: idx for idx, name in enumerate(labels)}

        x_np = df[square_cols].to_numpy(dtype=np.uint8)
        self.x = torch.from_numpy(x_np).long()

        series = df[label_column].map(LABEL_MAPPING)
        if series.isna().any():
            missing = set(df[label_column][series.isna()].unique())
            raise ValueError(
                f"Unknown labels in '{label_column}': {missing}. "
                f"Expected one of: {list(LABEL_MAPPING.keys())}"
            )
        y_np = series.to_numpy(dtype=np.int64)
        self.y = torch.from_numpy(y_np).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def grid_to_bitboards(grid: torch.Tensor) -> dict:
    """
    Convert a 64-long grid back into per-piece bitboards.
    Returns dict {piece_symbol: int_bitboard}.
    """
    bb_dict = {p: 0 for p in PIECE_ORDER}
    for sq in range(64):
        v = grid[sq].item()
        if v == 0:
            continue
        symbol = PIECE_ORDER[v]
        bb_dict[symbol] |= (1 << sq)
    return bb_dict