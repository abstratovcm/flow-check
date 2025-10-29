import chess

TARGET_FLAGS = {
    chess.STATUS_TOO_MANY_WHITE_PAWNS:  "too_many_white_pawns",
    chess.STATUS_TOO_MANY_BLACK_PAWNS:  "too_many_black_pawns",
    chess.STATUS_PAWNS_ON_BACKRANK:     "pawns_on_backrank",
    chess.STATUS_TOO_MANY_WHITE_PIECES: "too_many_white_pieces",
    chess.STATUS_TOO_MANY_BLACK_PIECES: "too_many_black_pieces",
}

INVALID_FLAGS = {
    chess.STATUS_NO_WHITE_KING:    "no_white_kings",
    chess.STATUS_NO_BLACK_KING:    "no_black_kings",
    chess.STATUS_TOO_MANY_KINGS:   "too_many_kings",
    chess.STATUS_OPPOSITE_CHECK:   "opposite_check",
}

PIECE_VALS = {
    '.':  0, 'P':  1, 'N':  3, 'B':  3, 'R':  5, 'Q':  9,
    'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9,
    'K':  0, 'k':  0
}

BAD_FLAGS = (
    chess.STATUS_NO_WHITE_KING |
    chess.STATUS_NO_BLACK_KING |
    chess.STATUS_TOO_MANY_KINGS |
    chess.STATUS_OPPOSITE_CHECK
)

# Define label constraints: cp range and material range
# Each entry: (cp_min, cp_max, mat_min, mat_max, cp_min_strict, cp_max_strict)
# Corresponds to labels 0-8
LABEL_CONSTRAINTS = {
    0: (-0.3, 0.3, 0, 0, False, False),      # (M^0,E^0)
    1: (0.3, 1.0, 0, 0, True, False),        # (M^0,E^+)
    2: (-0.3, 0.3, -15, -1, False, False),   # (M^-,E^0)
    3: (-1.0, -0.3, 0, 0, False, True),      # (M^0,E^-)
    4: (-0.3, 0.3, 1, 15, False, False),     # (M^+,E^0)
    5: (-1.0, -0.3, -15, -1, False, True),   # (M^-,E^-)
    6: (0.3, 1.0, -15, -1, True, False),     # (M^-,E^+)
    7: (0.3, 1.0, 1, 15, True, False),       # (M^+,E^+)
    8: (-1.0, -0.3, 1, 15, False, True),     # (M^+,E^-)
}
