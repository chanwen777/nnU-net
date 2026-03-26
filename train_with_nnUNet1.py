"""
nnU-Net Training Orchestration Script

Two-Stage Pipeline:
  1. Experiment planning and preprocessing: nnUNetv2_plan_and_preprocess
  2. Training: nnUNetv2_train

Logic:
- When only -d/--dataset is provided, configurations, folds, and train_args 
  are loaded from 'configs/train_config.py'.
- Command-line arguments override config file values.
- Multiple configurations are trained sequentially (ensures 3d_lowres precedes cascade).
- Multiple folds are trained in parallel across available GPUs.
- Supports trainer/plans as lists to train all (num_trainers * num_plans) combinations.
"""

import os
import re
import logging
import subprocess
import argparse
from pathlib import Path
import time

# Local utility imports
from .utils.logging_utils import setup_logging
from .utils.train_utils import *
from .configs.train_config import TRAIN_DATASET_CONFIGS, DEFAULT_FOLDS


# Environment Variables (Consistent with predict_with_nnUNet.py)
os.environ['nnUNet_raw'] = "/your/path/to/nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "/your/path/to/nnUNet_preprocessed" 
os.environ['nnUNet_results'] = "/your/path/to/nnUNet_results"


# Dependency Order: 3d_lowres MUST be trained before 3d_cascade_fullres
CONFIG_ORDER = ["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"]


# =============================================================================
# Stage 1: Experiment planning and preprocessing
# =============================================================================


def run_plan_and_preprocess(dataset_id, verify_integrity=True):
    """
    Executes 'nnUNetv2_plan_and_preprocess' for the given dataset ID.
    """
    cmd = ["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id)]
    if verify_integrity:
        cmd.append("--verify_dataset_integrity")
    
    logging.info("[Plan] Executing: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, text=True)
        logging.info("[Plan] Completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logging.error("[Plan] Failed: %s", e.stderr or str(e))
        raise


# =============================================================================
# Stage 2: Training
# =============================================================================


def run_training(
    dataset_id,
    configurations,
    folds,
    train_args,
    trainer_plan_combinations,
    num_gpus_available=None,
    wait_for_extraction_seconds=300,
):
    """
    Orchestrates training across configurations and folds.
    
    - Configurations are sorted to respect cascade dependencies.
    - Each (trainer, plan) combination is trained for all specified folds.
    - Single GPU: Folds are trained sequentially.
    - Multi-GPU: Folds are distributed across GPUs for parallel execution.
    """
    configs_sorted = sort_configurations(configurations)
    
    # Identify available GPU resources
    try:
        import torch
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        gpu_count = 0
    
    if num_gpus_available is not None:
        gpu_count = min(num_gpus_available, gpu_count) if gpu_count else num_gpus_available
    
    if gpu_count <= 0:
        logging.warning("No GPU detected. Falling back to CPU training.")
        gpu_count = 1

    for configuration in configs_sorted:
        logging.info("[Train] Starting configuration: %s", configuration)
        if not folds or not trainer_plan_combinations:
            continue

        for combo_idx, (trainer, plan) in enumerate(trainer_plan_combinations):
            logging.info("[Train] tr=%s p=%s | Target folds: %s", trainer, plan, folds)

            # --- Sequential Mode (Single GPU) ---
            if gpu_count == 1:
                for fold_idx, fold in enumerate(folds):
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = "0"
                    cmd = build_train_cmd(dataset_id, configuration, fold, train_args, trainer=trainer, plan=plan)
                    
                    if combo_idx == 0 and fold_idx == 0:
                        logging.info("[Train] Single-GPU mode. Running fold %s (Initial data extraction).", fold)
                    else:
                        logging.info("[Train] Processing fold %s", fold)
                    subprocess.run(cmd, env=env, check=True)
            
            # --- Parallel Mode (Multi-GPU) ---
            else:
                fold0 = folds[0]
                rest_folds = folds[1:]
                is_first_combo = (combo_idx == 0)
                
                env0 = os.environ.copy()
                env0["CUDA_VISIBLE_DEVICES"] = "0"
                cmd0 = build_train_cmd(dataset_id, configuration, fold0, train_args, trainer=trainer, plan=plan)
                
                # Critical: Avoid race conditions during data extraction by letting the first fold start alone
                if is_first_combo:
                    logging.info("[Train] Multi-GPU mode. Starting fold %s. Waiting %s seconds for extraction.", 
                                 fold0, wait_for_extraction_seconds)
                
                # Use Popen for non-blocking execution; redirect output to avoid terminal clutter
                proc0 = subprocess.Popen(cmd0, env=env0, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                if is_first_combo:
                    time.sleep(wait_for_extraction_seconds)

                # Check if first fold crashed early
                if proc0.poll() is not None:
                    proc0.wait()
                    if proc0.returncode != 0:
                        raise RuntimeError(f"Training failed for tr={trainer} p={plan} fold {fold0}")
                else:
                    logging.info("[Train] Fold %s initialized. Launching remaining folds in parallel.", fold0)

                # Launch remaining folds across GPUs
                procs_info = [(proc0, fold0, 0)]
                for i, fold in enumerate(rest_folds):
                    gpu_id = (i + 1) % gpu_count
                    env_i = os.environ.copy()
                    env_i["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    cmd_i = build_train_cmd(dataset_id, configuration, fold, train_args, trainer=trainer, plan=plan)
                    
                    proc = subprocess.Popen(cmd_i, env=env_i, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    procs_info.append((proc, fold, gpu_id))
                    logging.info("[Train] Launched fold %s on GPU %s", fold, gpu_id)

                # Wait for all processes in this trainer/plan combination to complete
                for proc, fold_val, _ in procs_info:
                    proc.wait()
                    if proc.returncode != 0:
                        raise RuntimeError(f"Training failed for tr={trainer} p={plan} fold {fold_val}")
            
            logging.info("[Train] tr=%s p=%s | All %d folds completed.", trainer, plan, len(folds))

    logging.info("[Train] All training configurations finished.")


# =============================================================================
# Main: Argument Parsing & Execution
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="nnU-Net Training Orchestrator. Uses 'train_config.py' if only -d is specified.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use full parameters from config
  python -m train_code.train_with_nnUNet -d Dataset043_Osteosclerosis

  # Preprocessing only
  python -m train_code.train_with_nnUNet -d Dataset043 --plan_only

  # Training only with argument overrides
  python -m train_code.train_with_nnUNet -d Dataset043 --train_only -c 2d -f 0 1 -tr nnUNetTrainerNoMirroring

  # Combinatorial exploration (2 Trainers x 2 Plans = 4 jobs)
  python -m train_code.train_with_nnUNet -d Dataset043 -tr nnUNetTrainer nnUNetTrainerNoMirroring -p nnUNetPlans CustomPlans
        """,
    )

    parser.add_argument("-d", "--dataset", type=str, required=True, help="Dataset name or ID.")

    # Execution Stages
    parser.add_argument("--plan_only", action="store_true", help="Run plan_and_preprocess only.")
    parser.add_argument("--no_verify_dataset_integrity", action="store_true", help="Skip dataset integrity verification.")
    parser.add_argument("--train_only", action="store_true", help="Run training only (requires existing preprocessed data).")

    # Training Parameters (Mapped to nnUNetv2_train CLI)
    parser.add_argument("-c", "--configurations", type=str, nargs="+", help="Configurations (e.g., 2d 3d_fullres).")
    parser.add_argument("-f", "--folds", type=int, nargs="+", help="Folds to train (e.g., 0 1 2 3 4).")
    parser.add_argument("-tr", "--trainer", type=str, nargs="+", help="nnU-Net trainer(s).")
    parser.add_argument("-p", "--plans", type=str, nargs="+", help="nnU-Net plan(s).")
    parser.add_argument("--pretrained_weights", type=str, help="Path to pretrained weights.")
    parser.add_argument("--num_gpus", type=int, help="Number of GPUs for parallel fold training.")
    parser.add_argument("--npz", action="store_true", help="Save softmax probabilities for ensembling.")
    parser.add_argument("--continue_train", action="store_true", help="Continue training from the latest checkpoint.")
    parser.add_argument("--val", dest="val_only", action="store_true", help="Run validation only.")
    parser.add_argument("--val_best", action="store_true", help="Validate using the best checkpoint.")
    parser.add_argument("--disable_checkpointing", action="store_true", help="Disable saving checkpoints.")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "mps"], default=None, help="Device selection.")
    parser.add_argument("--wait_extraction", type=int, default=300, help="Seconds to wait for the first fold to extract data.")

    args = parser.parse_args()
    setup_logging()

    # Dataset ID Parsing
    dataset_name_or_id = (args.dataset or "").strip()
    if dataset_name_or_id.isdigit():
        dataset_id = int(dataset_name_or_id)
    else:
        m = re.match(r"^Dataset(\d+)_", dataset_name_or_id, re.IGNORECASE)
        if m:
            dataset_id = int(m.group(1))
        else:
            raise ValueError(f"Could not parse Dataset ID from: {dataset_name_or_id}")

    # Configuration Resolution: CLI > Config File
    config = TRAIN_DATASET_CONFIGS.get(args.dataset)
    if not config and dataset_name_or_id.isdigit():
        prefix = f"Dataset{dataset_id:03d}_"
        for k, v in TRAIN_DATASET_CONFIGS.items():
            if k.startswith(prefix):
                config = v
                break
    
    if not config:
        raise ValueError(f"No configuration found for {args.dataset} in configs/train_config.py")

    configurations = args.configurations or config.get("configurations", ["2d", "3d_fullres"])
    folds = args.folds if args.folds is not None else config.get("folds", list(DEFAULT_FOLDS))
    
    # Normalize folds to List[int]
    if folds is None: folds = []
    elif isinstance(folds, int): folds = [folds]
    elif isinstance(folds, (tuple, set)): folds = list(folds)

    verify_integrity = not args.no_verify_dataset_integrity and config.get("verify_dataset_integrity", True)

    # Merge training arguments
    train_args_from_config = config.get("train_args", {})
    train_args_from_cli = parse_train_args_from_cli(args)
    if args.npz: train_args_from_cli["npz"] = True
    train_args = {**train_args_from_config, **train_args_from_cli}

    # Generate Cartesian product of Trainers and Plans
    trainer_plan_combinations = get_trainer_plan_combinations(
        train_args.get("tr"), train_args.get("p")
    )
    
    logging.info("[Train] Trainer x Plan combinations: %d", len(trainer_plan_combinations))
    for tr, p in trainer_plan_combinations:
        logging.info("  - tr=%s p=%s", tr, p)

    # Execution Logic
    if not args.train_only:
        run_plan_and_preprocess(dataset_id, verify_integrity=verify_integrity)

    if not args.plan_only:
        run_training(
            dataset_id,
            configurations=configurations,
            folds=folds,
            train_args=train_args,
            trainer_plan_combinations=trainer_plan_combinations,
            num_gpus_available=args.num_gpus,
            wait_for_extraction_seconds=args.wait_extraction,
        )

    logging.info("nnU-Net training pipeline completed.")


if __name__ == "__main__":
    main()
