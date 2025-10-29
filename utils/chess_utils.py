import chess
import chess.engine
from data.chess_dataset import PIECE_ORDER

from utils.constants import BAD_FLAGS, PIECE_VALS

class EngineManager:
    """Wraps Stockfish, manages its lifecycle, and restarts it if it dies."""
    def __init__(self, path, **uci_kwargs):
        self.path = path
        self.uci_kwargs = uci_kwargs
        self._start()

    def _start(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(self.path)
        self.engine.configure(self.uci_kwargs)

    def ensure(self):
        """Returns a working engine, restarting it if the process died."""
        proc = getattr(self.engine, "_proc", None)
        if proc is None or proc.poll() is not None:
            try:
                self.engine.quit()
            except:
                pass
            self._start()
        return self.engine

    def analyse(self, board, limit):
        """Runs analysis, ensuring the engine is alive first."""
        try:
            return self.ensure().analyse(board, limit)
        except (chess.engine.EngineTerminatedError, BrokenPipeError, TimeoutError):
            # Engine died mid-analysis, try one more time
            print("WARNING: Engine died mid-analysis, restarting...")
            return self.ensure().analyse(board, limit)

    def quit(self):
        """Properly shuts down the engine process."""
        if self.engine:
            try:
                self.engine.quit()
            except chess.engine.EngineTerminatedError:
                pass # Already dead
            self.engine = None

def tensor_to_fen(board_tensor):
    """
    board_tensor: length-64 vector, idx 0=a1 … idx63=h8.
    """
    squares = [PIECE_ORDER[int(p)] for p in board_tensor]

    # build ranks bottom-to-top
    ranks = []
    for r in range(7, -1, -1):           # 7→0
        row = squares[r*8:(r+1)*8]
        fen_row = ""
        empty = 0
        for c in row:
            if c == ".":
                empty += 1
            else:
                if empty:
                    fen_row += str(empty)
                    empty = 0
                fen_row += c
        if empty:
            fen_row += str(empty)
        ranks.append(fen_row)

    return "/".join(ranks) + " w - - 0 1"

def compute_material_advantage(board: chess.Board) -> int:
    """
    Sum up piece_vals over all pieces on the board.
    Positive → White is ahead; Negative → Black is ahead; Zero → even.
    """
    total = 0
    for _, piece in board.piece_map().items():
        total += PIECE_VALS[piece.symbol()]
    return total

def get_piece_counts(board: chess.Board):
    """Returns (white_piece_count, black_piece_count)."""
    white_count = sum(1 for p in board.piece_map().values() if p.color == chess.WHITE)
    black_count = sum(1 for p in board.piece_map().values() if p.color == chess.BLACK)
    return white_count, black_count

def classify(cp, mate):
    """Classifies a (cp, mate) score into a human-readable label."""
    if mate is not None:
        return "mate"
    if cp is None:
        return ""
    if -0.3 <= cp <= 0.3:
        return "balanced"
    if 0.3 < cp <= 1.0:
        return "white slight advantage"
    if cp > 1.0:
        return "white decisive advantage"
    if -1.0 <= cp < -0.3:
        return "black slight advantage"
    return "black decisive advantage"

def quick_validate(board_tensor):
    """Cheap pre-filter (no engine). Returns (fen, status, wk, bk, mat) or (None, ...)."""
    fen    = tensor_to_fen(board_tensor)
    board  = chess.Board(fen)
    status = board.status()
    if status & BAD_FLAGS:
        # invalid — must still return 5 slots
        return None, None, None, None, None
    wk, bk = fen.count('K'), fen.count('k')
    if wk != 1 or bk != 1:
        # invalid king counts
        return None, None, None, None, None
    mat = sum(PIECE_VALS[c] for c in fen if c in PIECE_VALS)
    return fen, status, wk, bk, mat

def evaluate_cp(fen, mgr: EngineManager):
    """Single engine call, returns CP score."""
    board = chess.Board(fen)
    info  = mgr.analyse(board, chess.engine.Limit(depth=15))
    raw   = info['score'].pov(chess.WHITE).score(mate_score=10000)
    return max(-2000, min(2000, raw)) / 100.0

def analyze_board(board_tensor, engine: EngineManager):
    """
    Full analysis: quick validate + stockfish.
    Returns (cp, fen, mat, status, wk, bk)
    """
    fen, status, wk, bk, mat = quick_validate(board_tensor)
    if fen is None:
        return None, fen, mat, status, wk, bk

    try:
        cp = evaluate_cp(fen, engine)
        return cp, fen, mat, status, wk, bk
    except:
        try:
            cp = evaluate_cp(fen, engine)
            return cp, fen, mat, status, wk, bk
        except:
            return None, fen, mat, status, wk, bk
