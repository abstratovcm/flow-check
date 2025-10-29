import torch
import math
import chess
import chess.engine
from torch.distributions import Categorical
from data.chess_dataset import PIECE_ORDER
from utils.chess_utils import tensor_to_fen

def compute_king_loss(probs, w):
    """Calculates the king penalty loss."""
    iK = PIECE_ORDER.index('K')
    ik = PIECE_ORDER.index('k')
    counts_Kw = probs[..., iK].sum(dim=1)
    counts_Kb = probs[..., ik].sum(dim=1)

    king_penalty = (w * ((counts_Kw - 1).abs() + (counts_Kb - 1).abs())).mean()

    return king_penalty

def compute_piece_loss(probs, w):
    """Calculates the piece count penalty loss."""
    empty_id     = PIECE_ORDER.index('.')
    presence     = 1 - probs[..., empty_id]
    piece_counts = presence.sum(dim=1)
    piece_penalty= (w * (16 - piece_counts).clamp(min=0)).mean()

    return piece_penalty

def compute_mat_loss(probs, mat_sign, label, w, no_mat_ids_t,
                     white_mat_ids_t, black_mat_ids_t):
    """Calculates the material advantage violation loss."""
    mat_diff = (probs * mat_sign).sum(dim=(1,2))

    mat_loss = torch.zeros((), device=probs.device)

    mask_no    = torch.isin(label, no_mat_ids_t)
    mask_white = torch.isin(label, white_mat_ids_t)
    mask_black = torch.isin(label, black_mat_ids_t)

    if mask_no.any():
        vio0 = mat_diff[mask_no].abs()
        mat_loss += (w[mask_no] * vio0).mean()

    if mask_white.any():
        vioW = (1.0 - mat_diff[mask_white]).clamp(min=0)
        mat_loss += (w[mask_white] * vioW).mean()

    if mask_black.any():
        vioB = (mat_diff[mask_black] + 1.0).clamp(min=0)
        mat_loss += (w[mask_black] * vioB).mean()

    return mat_loss

def compute_cp_loss(logits, label, sample, B, device,
                    stockfish, engine_path,
                    cp_bal_ids_t, cp_white_ids_t, cp_black_ids_t):
    """Calculates the centipawn policy gradient loss."""
    K = 1
    full_pg_sum = torch.zeros(B, device=device)
    for _ in range(K):
        dist = Categorical(logits=logits)
        sampled_board = dist.sample()
        log_prob_per_sq = dist.log_prob(sampled_board)
        log_prob =      log_prob_per_sq.sum(dim=1)

        p_total = 1.0
        num_to_eval = int(math.ceil(p_total * B))

        eligible = (sample.t >= 0.97).nonzero(as_tuple=False).view(-1)

        if eligible.numel() <= num_to_eval:
            selected = eligible
        else:
            perm = torch.randperm(eligible.numel(), device=device)
            selected = eligible[perm[:num_to_eval]]

        mask_eval = torch.zeros(B, dtype=torch.bool, device=device)
        mask_eval[selected] = True

        num_eval = int(mask_eval.sum().item())
        if num_eval > 0:
            cp_vals          = torch.zeros(B, device=device)
            status_penalties = torch.zeros(B, device=device)

            eval_indices = mask_eval.nonzero(as_tuple=False).view(-1)
            for i in eval_indices:
                board_tensor = sampled_board[i]
                fen = tensor_to_fen(board_tensor)
                board = chess.Board(fen)
                status = board.status()
                status_mask = (
                    chess.STATUS_NO_WHITE_KING
                    | chess.STATUS_NO_BLACK_KING
                    | chess.STATUS_TOO_MANY_KINGS
                    | chess.STATUS_OPPOSITE_CHECK
                )
                if status & status_mask:
                    cp_vals[i]          = 0.0
                    status_penalties[i] = 10.0
                else:
                    try:
                        info = stockfish.analyse(board, chess.engine.Limit(depth=10))
                        raw = info["score"].pov(chess.WHITE).score(mate_score=10000)
                        raw = max(-2000, min(2000, raw))
                        cp_vals[i] = raw / 100.0
                    except chess.engine.EngineTerminatedError:
                        stockfish = chess.engine.SimpleEngine.popen_uci(engine_path)
                        stockfish.configure({"Threads": 4, "UCI_LimitStrength": False, "Hash": 2048})
                        try:
                            info = stockfish.analyse(board, chess.engine.Limit(depth=10))
                            raw = info["score"].pov(chess.WHITE).score(mate_score=10000)
                            raw = max(-2000, min(2000, raw))
                            cp_vals[i] = raw / 100.0
                        except chess.engine.EngineTerminatedError:
                            cp_vals[i] = 0.0
                            status_penalties[i] = 10.0

        else:
            cp_vals          = torch.zeros(B, device=device)
            status_penalties = torch.zeros(B, device=device)

        max_violation = 10.0

        invalid_mask = status_penalties > 0
        valid_mask   = ~invalid_mask
        violation = status_penalties.clone()

        mask_bal_cp     = torch.isin(label, cp_bal_ids_t)
        mask_bal_eval   = mask_bal_cp & mask_eval & valid_mask
        if mask_bal_eval.any():
            cp_sel        = cp_vals[mask_bal_eval]
            raw_violation = torch.clamp(-0.3 - cp_sel, min=0.0) \
                          + torch.clamp(cp_sel - 0.3,   min=0.0)
            violation[mask_bal_eval] += torch.clamp(raw_violation, max=max_violation)

        mask_white_cp = torch.isin(label, cp_white_ids_t)
        mask_wht_eval    = mask_white_cp & mask_eval & valid_mask
        if mask_wht_eval.any():
            cp_sel        = cp_vals[mask_wht_eval]
            raw_violation = torch.clamp(0.3  - cp_sel, min=0.0) \
                          + torch.clamp(cp_sel - 1.0,  min=0.0)
            violation[mask_wht_eval] += torch.clamp(raw_violation, max=max_violation)

        mask_black_cp   = torch.isin(label, cp_black_ids_t)
        mask_blk_eval   = mask_black_cp & mask_eval & valid_mask
        if mask_blk_eval.any():
            cp_sel        = cp_vals[mask_blk_eval]
            raw_violation = torch.clamp(-1.0 - cp_sel, min=0.0) \
                          + torch.clamp(cp_sel + 0.3,   min=0.0)
            violation[mask_blk_eval] += torch.clamp(raw_violation, max=max_violation)

        violation.clamp_(0.0, max_violation)

        w = torch.zeros_like(sample.t)
        w[sample.t >= 0.97] = 1.0
        if num_eval > 0:
            baseline = violation[mask_eval].mean().detach()
        else:
            baseline = torch.tensor(0.0, device=device)
        advantage = violation - baseline

        if num_eval > 0:
            adv_sel      = advantage[mask_eval]
            logp_sel     = log_prob[mask_eval]
            w_sel        = w[mask_eval]
            pg = w_sel * adv_sel * (-logp_sel)
            pg = pg / (float(num_to_eval) / B)
        else:
            pg = torch.zeros(0, device=device)
        full_pg = torch.zeros(B, device=device)
        full_pg[mask_eval] = pg
        full_pg_sum += full_pg

    full_pg_avg = full_pg_sum / float(K)
    cp_pg_loss  = full_pg_avg.mean()

    return cp_pg_loss