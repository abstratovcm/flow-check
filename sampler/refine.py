import torch
import torch.nn.functional as F
from itertools import product
from data.chess_dataset import PIECE_ORDER
from utils.constants import LABEL_CONSTRAINTS
from utils.chess_utils import quick_validate, evaluate_cp, analyze_board
import time

def _compute_entropy(probs):
    return -(probs*probs.clamp(min=1e-9).log()).sum(dim=-1)

def _analyze_base_board(board, engine, log, label_name):
    """Analyzes the initial sampled board and logs info."""
    base_cp, base_fen, base_mat, base_status, base_wk, base_bk = analyze_board(board, engine)
    sample_valid = base_cp is not None

    if log:
        header = label_name or f"#{label_name}"
        print(f"\n=== Refinement for label: {header} ===")
        if sample_valid:
            print(f"Base board: {base_fen}")
            print(f"  cp={base_cp:+.2f}, mat={base_mat}, status={base_status}, kings W{base_wk} B{base_bk}")
        else:
            print(f"Base board invalid (status={base_status}). Will still try candidates.")

    return sample_valid, base_cp, base_mat, base_fen, (board.clone(), base_cp)

def _check_early_return(sample_valid, base_cp, base_mat, constraints):
    """Checks if the base board already satisfies all constraints."""
    if not sample_valid:
        return False

    cp_min, cp_max, mat_min, mat_max, strict_min, strict_max = constraints

    # Material check
    if not (mat_min <= base_mat <= mat_max):
        return False

    # CP-bounds check (strict vs non-strict)
    valid_low  = (base_cp >  cp_min) if strict_min else (base_cp >= cp_min)
    valid_high = (base_cp <  cp_max) if strict_max else (base_cp <= cp_max)

    return valid_low and valid_high

def _get_candidate_choices(model, board, t, lbl, n_uncertain, n_candidates, log):
    """Gets model probabilities and identifies candidate piece swaps."""
    logits = model(board.unsqueeze(0), t=t, label=lbl)[0]
    probs = F.softmax(logits, dim=-1)
    entropy = _compute_entropy(probs)
    top_idxs = entropy.topk(n_uncertain).indices.tolist()

    if log:
        print("Top uncertain squares:")
        for i in top_idxs: print(f"  sq{i}: ent={entropy[i]:.4f}")

    # Gather candidates
    choices = [probs[i].topk(n_candidates).indices.tolist() for i in top_idxs]

    if log:
        for idx, opt in zip(top_idxs, choices):
            print(f"Sq{idx} opts: {[PIECE_ORDER[o] for o in opt]}")

    return top_idxs, choices

def _stage1_filter_candidates(board_tensor, top_idxs, choices, constraints, log):
    """Generates all combinations and filters them using quick_validate."""
    t_stage1_start = time.time()
    _, _, mat_min, mat_max, _, _ = constraints

    valid_boards = []
    total_combos = 0

    for combo in product(*choices):
        total_combos += 1
        b = board_tensor.clone()
        for idx, ch in zip(top_idxs, combo):
            b[idx] = ch

        fen, status, wk, bk, mat = quick_validate(b)
        if fen is None:
            continue
        if not (mat_min <= mat <= mat_max):
            continue

        valid_boards.append((b, fen, mat, status, wk, bk))

    if log:
        kept = len(valid_boards)
        print(f"Pruned {total_combos} → {kept} boards after quick_validate & material filter")
        t_stage1_end = time.time()
        print(f"Stage 1 (product+filter) took {t_stage1_end - t_stage1_start:.3f}s")

    return valid_boards

def _stage2_evaluate_candidates(valid_boards, engine, best, cp_mid, constraints,
                              early_return, timed_out, log):
    """Runs Stockfish evaluation on the filtered candidates."""
    t_stage2_start = time.time()

    best_board, best_cp = best
    best_diff = abs(best_cp - cp_mid) if best_cp is not None else float('inf')

    cp_min, cp_max, _, _, strict_min, strict_max = constraints

    for b, fen, mat, status, wk, bk in valid_boards:
        if timed_out():
            if log: print(f"Timeout in stage 2, stopping early")
            break

        cp = evaluate_cp(fen, engine)
        dist = abs(cp - cp_mid)

        if dist < best_diff:
            best_diff = dist
            best_board = b.clone()
            best_cp = cp

        # early-return if this candidate fully satisfies the interval
        valid_low  = (cp >  cp_min) if strict_min else (cp >= cp_min)
        valid_high = (cp <  cp_max) if strict_max else (cp <= cp_max)

        if early_return and valid_low and valid_high:
            if log:
                print(f"Candidate (early return): {fen}")
                print(f"  cp={cp:+.2f}, mat={mat}, status={status}, kings W{wk} B{bk}")
            return best_board, best_cp

    t_stage2_end = time.time()
    if log:
        print(f"Stage 2 (Stockfish eval) took {t_stage2_end - t_stage2_start:.3f}s")

    return best_board, best_cp

def _finalize_result(best, base_board, base_cp, sample_valid, cp_mid, log):
    """Logs the final result and returns the chosen board."""
    best_board, best_cp = best

    if log:
        if best_board is not None:
            print(f"Best cp close to mid {cp_mid:+.2f}: {best_cp:+.2f}")
        else:
            print("No candidate met the constraints.")

    if best_board is None:
        if sample_valid:
            if log: print("→ Falling back to original valid board.")
            return base_board, base_cp
        else:
            if log: print("→ No valid board found, returning None.")
            return None, None

    return best_board, best_cp

def refine_board(model, board_tensor, label, engine, device,
                 label_name=None,
                 n_uncertain=3, n_candidates=3, log=True,
                 early_return=True,
                 time_limit=None):
    """
    Attempts to "fix" a sampled board by modifying its most
    uncertain squares to better match the target label constraints.
    Returns (best_board_tensor, best_cp_score).
    """
    t_refine_start = time.time()
    def timed_out():
        return time_limit is not None and (time.time() - t_refine_start) > time_limit

    # 1. Get constraints
    cp_min, cp_max, mat_min, mat_max, strict_min, strict_max = LABEL_CONSTRAINTS[label]
    cp_mid = (cp_min + cp_max) / 2
    constraints = (cp_min, cp_max, mat_min, mat_max, strict_min, strict_max)

    # 2. Setup tensors
    board = board_tensor.clone().detach().to(device)
    t = torch.tensor([1.0 - 1e-3], device=device)
    lbl = torch.tensor([label], device=device)

    # 3. Analyze base board
    sample_valid, base_cp, base_mat, _, base_best = _analyze_base_board(
        board, engine, log, label_name
    )

    # 4. Check for early return if base board is already perfect
    if early_return and _check_early_return(sample_valid, base_cp, base_mat, constraints):
        return base_best

    # 5. Get candidate choices from model
    top_idxs, choices = _get_candidate_choices(
        model, board, t, lbl, n_uncertain, n_candidates, log
    )

    # 6. Initialize best
    if sample_valid:
        best = base_best
    else:
        best = (None, None)

    # 7. Stage 1: Quick Filter
    valid_boards = _stage1_filter_candidates(
        board, top_idxs, choices, constraints, log
    )

    # 8. Stage 2: Stockfish Evaluation
    best_board, best_cp = _stage2_evaluate_candidates(
        valid_boards, engine, best, cp_mid, constraints,
        early_return, timed_out, log
    )

    # 9. Log and Finalize
    t_refine_end = time.time()
    if log:
        print(f"Total refine_board took {t_refine_end - t_refine_start:.3f}s\n")

    return _finalize_result(
        (best_board, best_cp), board, base_cp, sample_valid, cp_mid, log
    )
