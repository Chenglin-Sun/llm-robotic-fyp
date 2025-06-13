#!/usr/bin/env python3
"""
Generate chessboard_positions_list from calibrated robot-world grid positions.

Reads from a .npy file (default: grid_robot_xy_corrected.npy) containing an (8, 8, 2) array.
Each element is (x, y) in meters.

Usage:
    import chess_board_corrected
    positions = chess_board_corrected.chessboard_positions_list
"""

import numpy as np
from pathlib import Path

FILES = "ABCDEFGH"  # Columns A–H
RANKS = "12345678"  # Rows 1–8 (bottom to top)

# Load the calibrated grid from .npy
GRID_PATH = Path(__file__).resolve().parent / "grid_robot_xy.npy"
try:
    grid = np.load(GRID_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Calibration grid not found at {GRID_PATH}")

if grid.shape != (8, 8, 2):
    raise ValueError(f"Expected shape (8, 8, 2), got {grid.shape}")

# Construct the chessboard position list
chessboard_positions_list = []

for row_idx in range(8):
    for col_idx in range(8):
        square_name = FILES[col_idx] + RANKS[7 - row_idx]  # row 0 = rank 8
        x, y = grid[row_idx, col_idx]
        chessboard_positions_list.append({
            "name": square_name,
            "position": (float(x), float(y))  # Ensure standard Python float types
        })

# Optional: preview
if __name__ == "__main__":
    for entry in chessboard_positions_list:
        print(entry)
