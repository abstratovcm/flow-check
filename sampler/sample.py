import argparse, json, os, time, csv
import torch
import chess
import chess.engine
from tqdm import tqdm

from data.chess_dataset import PIECE_ORDER
from model.denoiser import CFGProbabilityDenoiser, ProbabilityDenoiser
from model.vt_discrete_flow import VisionTransformerDiscreteFlow as DiscreteFlow
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.solver import MixtureDiscreteEulerSolver

from sampler.refine import refine_board
from utils.chess_utils import tensor_to_fen, compute_material_advantage, classify, EngineManager, get_piece_counts
from utils.constants import BAD_FLAGS, INVALID_FLAGS, TARGET_FLAGS

def analyze_raw_board(grid_tensor, x0_tensor, mgr: EngineManager):
    """
    Analyzes the 'raw' sampled board and returns a dict of results.
    Assumes grid_tensor and x0_tensor are on CPU.
    """
    data = {}

    raw_fen = tensor_to_fen(grid_tensor)
    board = chess.Board(raw_fen)
    data["fen4"] = " ".join(raw_fen.split()[:4])
    data["initial_x0"] = tensor_to_fen(x0_tensor)

    status = board.status()
    for mask, name in INVALID_FLAGS.items():
        data[name] = bool(status & mask)
    for mask, name in TARGET_FLAGS.items():
        data[name] = bool(status & mask)

    wc, bc = get_piece_counts(board)
    data["white_piece_count"] = wc
    data["black_piece_count"] = bc
    data["material_advantage"] = compute_material_advantage(board)

    is_invalid = bool(status & BAD_FLAGS)
    data["cp0"] = ""
    data["predicted_label"] = ""
    data["cp"] = ""
    data["mate"] = ""
    data["engine_time_ms"] = ""

    if is_invalid:
        return data # Return early if board is fundamentally broken

    # Depth-0 analysis
    try:
        info0 = mgr.analyse(board, chess.engine.Limit(depth=0))
        sc0 = info0["score"].white()
        data["cp0"] = sc0.score(mate_score=10**6) or ""
    except Exception:
        pass

    # Deep analysis
    try:
        t1 = time.time()
        info = mgr.analyse(board, chess.engine.Limit(depth=20, time=10))
        eng_ms = (time.time() - t1) * 1000

        sc = info["score"].white()
        cp = sc.score(mate_score=10**6)
        mate = sc.mate()

        if cp is None and mate is None:
            pass # Fields already blank
        elif mate is None: # Centipawn
            data["predicted_label"] = classify(cp / 100, None)
            data["cp"] = cp
        else: # Mate
            data["predicted_label"] = classify(None, mate)
            data["mate"] = mate

        data["engine_time_ms"] = f"{eng_ms:.1f}"

    except Exception as e:
        print(f"\n⚠️  Engine error on {data['fen4']}… skipping: {e}")

    return data

def analyze_corrected_board(grid_tensor_dev, model, mgr: EngineManager, label_id, device, label_str):
    """
    Runs 'refine_board' and performs a deep analysis on the *corrected* board.
    Returns a dict of results. Assumes grid_tensor_dev is on DEVICE.
    """
    data = {}

    t0 = time.time()
    corr_tensor, corr_cp = refine_board(
        model, grid_tensor_dev, label_id, mgr, device,
        label_name=label_str,
        n_uncertain=4,
        n_candidates=4,
        log=False,
        early_return=True,
        time_limit=5.0
    )
    corr_time_ms = (time.time() - t0) * 1000

    if corr_tensor is None:
        corr_tensor = grid_tensor_dev.clone()
        corr_cp = None

    data["correction_time_ms"] = f"{corr_time_ms:.1f}"
    data["corrected_cp"] = int(corr_cp * 100) if corr_cp is not None else ""
    data["corrected_mate"] = ""

    corr_fen = tensor_to_fen(corr_tensor.cpu())
    board = chess.Board(corr_fen)
    status = board.status()

    data["corrected_fen4"] = " ".join(corr_fen.split()[:4])
    wc, bc = get_piece_counts(board)
    data["corrected_white_piece_count"] = wc
    data["corrected_black_piece_count"] = bc
    data["corrected_material_advantage"] = compute_material_advantage(board)

    is_invalid = bool(status & BAD_FLAGS)
    data["corrected_engine_time_ms"] = ""
    data["corrected_cp_deep"] = ""
    data["corrected_mate_deep"] = ""
    data["corrected_label"] = ""

    if is_invalid:
        return data

    try:
        t1 = time.time()
        info = mgr.analyse(board, chess.engine.Limit(depth=20, time=10))
        eng_ms = (time.time() - t1) * 1000

        sc = info["score"].white()
        cp = sc.score(mate_score=10**6)
        mate = sc.mate()

        data["corrected_label"] = classify(
            cp / 100 if cp is not None else None,
            mate
        )
        data["corrected_cp_deep"] = cp if cp is not None else ""
        data["corrected_mate_deep"] = mate if mate is not None else ""
        data["corrected_engine_time_ms"] = f"{eng_ms:.1f}"

    except Exception as e:
        print(f"\n⚠️  Engine error on corrected {data['corrected_fen4']}… skipping: {e}")

    return data

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config","-c",required=True,help="path to JSON config")
    args = p.parse_args()
    cfg = json.load(open(args.config))

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    SAMPLES    = cfg["samples_per_label"]
    BATCH_SIZE = cfg["batch_size"]
    use_cfg    = cfg.get("use_cfg",True) and cfg["num_labels"]>1
    labels     = cfg["labels"]

    mgr = EngineManager(
        cfg["engine_path"],
        Threads=8,
        Hash=512,
        UCI_LimitStrength=False
    )

    if use_cfg:
        model = DiscreteFlow(label_classes=cfg["num_labels"]+1, d_model=cfg["d_model"], ff_dim=cfg["ff_dim"]).to(device)
    else:
        model = DiscreteFlow(label_classes=cfg["num_labels"], d_model=cfg["d_model"], ff_dim=cfg["ff_dim"]).to(device)

    ckpt = torch.load(cfg["output_pt"], map_location=device,weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sched = PolynomialConvexScheduler(n=1.0)
    path  = MixtureDiscreteProbPath(scheduler=sched)
    guidance_scales = [1.0, 1.5, 2.0]
    base = os.path.splitext(os.path.basename(cfg["output_pt"]))[0]

    fieldnames = [
        "guidance_scale", "conditioning_label", "fen4", "initial_x0",
        *INVALID_FLAGS.values(), *TARGET_FLAGS.values(),
        "white_piece_count", "black_piece_count", "material_advantage",
        "cp0", "cp", "mate", "predicted_label", "engine_time_ms", "sampling_time_ms",

        "corrected_fen4", "correction_time_ms",
        "corrected_cp", "corrected_mate", "corrected_engine_time_ms",
        "corrected_cp_deep", "corrected_mate_deep", "corrected_label",
        "corrected_material_advantage",
        "corrected_white_piece_count", "corrected_black_piece_count"
    ]

    out_fname = f"{base}_positions_all_terms_corrected.csv"
    print(f"Starting sampling. Output will be saved to {out_fname}")

    with open(out_fname, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        torch.manual_seed(0)

        for gs in guidance_scales:
            print(f"\n=== Sampling for guidance_scale={gs} ===")
            if use_cfg:
                den = CFGProbabilityDenoiser(model=model, guidance_scale=gs, null_label=cfg["null_label"])
            else:
                den = ProbabilityDenoiser(model=model)
            solver = MixtureDiscreteEulerSolver(model=den, path=path, vocabulary_size=len(PIECE_ORDER))

            for label_id, label_str in enumerate(labels):

                total_done = 0
                pbar = tqdm(total=SAMPLES, desc=f"→ Sampling “{label_str}”", leave=False)

                while total_done < SAMPLES:
                    curr_batch = min(BATCH_SIZE, SAMPLES - total_done)
                    x0_batch = torch.randint(model.vocab_size,(curr_batch,64),device=device)
                    lbls = torch.full((curr_batch,),label_id,device=device)

                    t0 = time.time()
                    with torch.no_grad():
                        grids = solver.sample(
                            x_init=x0_batch,step_size=1/100,
                            time_grid=torch.tensor([0.0,1.0-1e-3],device=device),
                            extras={"label":lbls}
                        ).cpu()
                    sampling_time_ms = (time.time()-t0)*1000 / curr_batch
                    for idx, grid_cpu in enumerate(grids):
                        if total_done >= SAMPLES:
                            break

                        row = {
                            "guidance_scale": gs,
                            "conditioning_label": label_str,
                            "sampling_time_ms": f"{sampling_time_ms:.1f}",
                        }

                        raw_data = analyze_raw_board(grid_cpu, x0_batch[idx].cpu(), mgr)
                        row.update(raw_data)

                        corr_data = analyze_corrected_board(
                            grid_cpu.to(device), model, mgr, label_id, device, label_str
                        )
                        row.update(corr_data)

                        writer.writerow(row)

                        total_done += 1
                        pbar.update(1)

                pbar.close()

        print(f"\n✅ Wrote all positions to {out_fname}")

    mgr.quit()

if __name__=="__main__":
    main()
