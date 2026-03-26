"""
Batch prediction using trained nnUNet models with real-time progress monitoring.
Maintains the same directory structure for output as the input.

nnUNet Inference Workflow (3 Steps):
1. Automatically determine the best configuration (via nnUNetv2_find_best_configuration).
2. Run inference.
3. Apply postprocessing (only when using 'best configuration' mode).

Supported Modes:
- --use_best_config: Parses inference and postprocessing commands from 'inference_instructions.txt'.
  (Triggers find_best_configuration automatically if the file is missing).
- Custom Mode: User specifies model_type, fold, trainer, plans, etc. Run inference only (no postprocessing).
"""

import os
import re
import logging
import shutil
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple, List
import threading
import subprocess

# Local utility imports
from .utils.logging_utils import setup_logging
from .utils.run_utils import monitor_output_folder
from .utils.predict_utils import *
from .configs.predict_data_config import PREDICT_CONFIGS

# Environment Variable Setup
os.environ['nnUNet_raw'] = "/your/path/to/nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "/your/path/to/nnUNet_preprocessed" 
os.environ['nnUNet_results'] = "/your/path/to/nnUNet_results"

# Valid nnUNet configuration types for filtering
VALID_CONFIGURATIONS = ["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"]

# =============================================================================
# Step 1: Automatically determine the best configuration
# =============================================================================

def run_find_best_configuration(
    dataset_name_or_id: str,
    configurations: Optional[List[str]] = None,
) -> bool:
    """
    Invokes 'nnUNetv2_find_best_configuration' to determine the optimal model setup.
    If 'configurations' is None, it scans 'nnUNet_results/DatasetXXX/' for trained models.
    Generates 'inference_instructions.txt' upon success.
    """
    if configurations is None:
        configurations = discover_trained_configurations(dataset_name_or_id)
        if not configurations:
            raise RuntimeError("No trained models found in nnUNet_results. Please complete training first.")

    cmd = ["nnUNetv2_find_best_configuration", str(get_dataset_name(dataset_name_or_id))]
    cmd.extend(["-c"] + configurations)
    logging.info("Executing Step 1: %s", " ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True, text=True)
        logging.info("find_best_configuration completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logging.error("find_best_configuration failed: %s", e.stderr or str(e))
        return False

# =============================================================================
# Step 3: Apply postprocessing (used only with best configuration)
# =============================================================================

def run_postprocessing(
    output_folder: Path,
    pp_parts: List[str],
    output_folder_pp: Optional[Path] = None,
) -> bool:
    """
    Executes 'nnUNetv2_apply_postprocessing' on the inference output directory.
    'output_folder_pp' defaults to output_folder + '_postprocessed'.
    """
    output_folder = Path(output_folder)
    if output_folder_pp is None:
        output_folder_pp = Path(str(output_folder) + "_postprocessed")
    else:
        output_folder_pp = Path(output_folder_pp)

    cmd = build_postprocessing_cmd_from_instructions(
        pp_parts,
        str(output_folder),
        str(output_folder_pp),
    )
    logging.info("Executing Postprocessing: %s", " ".join(cmd))
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logging.info("Postprocessing completed. Output directory: %s", output_folder_pp)
        return True
    except subprocess.CalledProcessError as e:
        logging.error("Postprocessing failed: %s", e.stderr)
        return False

# =============================================================================
# Step 2: Run inference (Supports Best Config & Custom Parameters)
# =============================================================================

def predict_with_nnunet(input_folder, 
                        output_folder, 
                        output_csv_path,
                        model_type, 
                        task_name, 
                        fold, 
                        file_type="nrrd",
                        folds=None,
                        prev_stage_predictions=None,
                        continue_prediction=False,
                        num_processes_preprocessing=1,
                        num_processes_saving=1,
                        trainer="nnUNetTrainer",
                        plans="nnUNetPlans",
                        use_best_config=False):
    """
    Performs batch prediction and displays real-time progress.
    """
    try:
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        if not input_folder.exists():
            raise ValueError(f"Input folder does not exist: {input_folder}")

        # Search for input files based on specified file type
        if file_type:
            ext = file_type.lower()
            input_files = list(input_folder.glob(f"*_0000.{ext}"))
            if not input_files:
                input_files = list(input_folder.rglob(f"*_0000.{ext}"))
            if input_files:
                logging.info(f"Found {len(input_files)} files with {ext.upper()} format.")
        else:
            raise ValueError("File type not specified. Use --file_type to define input format.")

        logging.info(f"Total files identified for prediction: {len(input_files)}")

        # Collect files and define expected output paths
        files_to_process = []
        output_paths = []
        for input_file in input_files:
            rel_path = input_file.relative_to(input_folder)
            input_stem = input_file.stem
            output_stem = input_stem[:-5] if input_stem.endswith("_0000") else input_stem
            
            output_ext = f".{file_type}"
            output_rel_path = rel_path.with_name(output_stem + output_ext)
            expected_output_path = output_folder / output_rel_path
            expected_output_path.parent.mkdir(parents=True, exist_ok=True)

            files_to_process.append(input_file)
            output_paths.append(expected_output_path)

        if not files_to_process:
            logging.info(f"No files found for prediction (missing *_0000.{file_type} in input directory).")
            return True

        # Group input files by directory to optimize nnUNetv2_predict calls
        grouped_input_folders = {}
        for input_file in files_to_process:
            grouped_input_folders.setdefault(input_file.parent, 0)
            grouped_input_folders[input_file.parent] += 1
        
        input_folders_sorted = sorted(grouped_input_folders.keys(), key=lambda p: str(p))
        logging.info(f"Grouped into {len(input_folders_sorted)} input directories.")

        # Best Config Logic: Parse inference_instructions.txt
        inference_parts_list = None
        ensemble_parts = None
        pp_parts = None
        if use_best_config:
            inference_parts_list, ensemble_parts, pp_parts = parse_inference_instructions(task_name)
            if inference_parts_list is None:
                logging.info("inference_instructions.txt missing. Running find_best_configuration automatically.")
                if not run_find_best_configuration(task_name):
                    raise ValueError("Automatic find_best_configuration failed.")
                inference_parts_list, ensemble_parts, pp_parts = parse_inference_instructions(task_name)
            
            if inference_parts_list is None:
                raise ValueError("use_best_config=True but failed to parse inference_instructions.txt.")
            
            mode_desc = "ensemble" if ensemble_parts else "single model"
            logging.info(f"Loaded best configuration ({mode_desc}) from instructions.")

        # Start background thread for progress monitoring
        stop_event = threading.Event()
        total_files = len(files_to_process)
        monitor_thread = threading.Thread(
            target=monitor_output_folder,
            args=(output_paths, stop_event, total_files)
        )
        monitor_thread.daemon = True
        monitor_thread.start()

        start_time = time.time()
        any_group_failed = False
        
        # Batch inference by directory
        for idx_group, in_dir in enumerate(input_folders_sorted, start=1):
            rel_dir = in_dir.relative_to(input_folder) if in_dir != input_folder else Path(".")
            out_dir = output_folder / rel_dir
            out_dir.mkdir(parents=True, exist_ok=True)

            prev_stage_path = str(prev_stage_predictions) if prev_stage_predictions else None
            cmds_to_run, extra_tmp_dirs = build_case_commands(
                use_best_config=use_best_config,
                inference_parts_list=inference_parts_list,
                ensemble_parts=ensemble_parts,
                task_name=str(task_name),
                model_type=str(model_type),
                trainer=trainer,
                plans=plans,
                fold=fold,
                folds=folds,
                input_folder=in_dir,
                output_folder=out_dir,
                prev_stage_runtime_path=prev_stage_path,
                continue_prediction=continue_prediction,
                num_processes_preprocessing=num_processes_preprocessing,
                num_processes_saving=num_processes_saving,
            )

            try:
                for cmd_idx, cmd in enumerate(cmds_to_run, start=1):
                    logging.info(f"[Group {idx_group}/{len(input_folders_sorted)}] Executing Command ({cmd_idx}/{len(cmds_to_run)}): {' '.join(cmd)}")
                    subprocess.run(cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                any_group_failed = True
                logging.error(f"Inference failed: {e.stderr}\nInput: {in_dir}\nOutput: {out_dir}")
            finally:
                for p in extra_tmp_dirs:
                    if p.exists():
                        shutil.rmtree(p, ignore_errors=True)

        # Post-inference summary
        ok_count = sum(1 for p in output_paths if p.exists())
        logging.info(f"Output Check: {ok_count}/{len(output_paths)} files exist.")
        
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        logging.info(f"Total Inference Runtime: {elapsed_str}")
        
        stop_event.set()
        monitor_thread.join(timeout=1.0)

        if any_group_failed:
            logging.error("Inference failed for some groups. Marking task as failed.")
            return False

        # Post-processing (only for best config mode)
        if use_best_config and pp_parts:
            pp_root = Path(str(output_folder) + "_postprocessed")
            logging.info("Starting batch post-processing...")
            for in_dir in input_folders_sorted:
                rel_dir = in_dir.relative_to(input_folder) if in_dir != input_folder else Path(".")
                out_dir = output_folder / rel_dir
                pp_out = pp_root / rel_dir
                pp_out.mkdir(parents=True, exist_ok=True)
                if not run_postprocessing(out_dir, pp_parts, output_folder_pp=pp_out):
                    logging.error(f"Postprocessing failed: {out_dir} -> {pp_out}")
                    return False
            logging.info(f"Inference and post-processing completed. Results: {pp_root}")

        return True
    except Exception as e:
        logging.error(f"An error occurred during prediction: {str(e)}", exc_info=True)
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Batch prediction using nnUNet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow Steps:
  1. Determine best configuration (auto-scan if missing when using --use_best_config).
  2. Run inference.
  3. Apply postprocessing (automatic with --use_best_config).

Examples:
  # Best configuration (automatic ensemble/single model)
  python -m train_code.predict_with_nnUNet --task Dataset038_Spine_Fracture --input_folder /path/to/in --output_folder /path/to/out --use_best_config --file_type nrrd --continue_prediction --npp 3 --nps 3

  # Custom mode
  python -m train_code.predict_with_nnUNet --task Dataset038_Spine_Fracture --input_folder /path/to/in --output_folder /path/to/out --model 3d_fullres --folds 0 1 2 3 4 --file_type nrrd
        """,
    )
    # Step 1 Args
    parser.add_argument('--task', type=str, required=True, help='Task name (e.g., Dataset038_Spine)')
    parser.add_argument('--use_best_config', action='store_true', help='Use optimal configuration instructions.')
    parser.add_argument('--data_mode', type=str, default='train_data', choices=['train_data', 'valid_data'])
    
    # Step 2 Args
    parser.add_argument('--input_folder', type=str, help='Input directory path.')
    parser.add_argument('--output_folder', type=str, help='Output directory path.')
    parser.add_argument('--model', type=str, default=None, help='Model type (e.g., 3d_fullres).')
    parser.add_argument('--fold', type=int, default=None, help='Specific fold index.')
    parser.add_argument('--folds', type=int, nargs='+', default=None, help='List of folds (e.g., 0 1 2 3 4).')
    parser.add_argument('--trainer', type=str, default=None, help='Trainer name.')
    parser.add_argument('--plans', type=str, default=None, help='Plans name.')
    parser.add_argument('--file_type', type=str, default=None, help='File format (nrrd/bmp).')
    parser.add_argument('--prev_stage_predictions', type=str, default=None, help='Path for previous stage (cascade models).')
    parser.add_argument('--continue_prediction', action='store_true', help='Skip existing files.')
    parser.add_argument('--npp', type=int, default=1, help='Preprocessing processes.')
    parser.add_argument('--nps', type=int, default=1, help='Saving/Postprocessing processes.')
    
    # Metadata
    parser.add_argument('--output_csv', type=str, default='', help='Path to save output CSV (optional).')

    args = parser.parse_args()
    setup_logging()

    # Configuration Priority: CLI > Config File > Default
    input_path = None
    output_path = None
    DATASET_NAME = get_dataset_name(args.task)
    
    # Load from config if available
    cfg = {}
    dataset_key_for_config = args.task
    if dataset_key_for_config and dataset_key_for_config not in PREDICT_CONFIGS:
        if dataset_key_for_config.isdigit():
            dataset_id = int(dataset_key_for_config)
        else:
            m = re.match(r"^Dataset(\d+)_?", dataset_key_for_config, re.IGNORECASE)
            dataset_id = int(m.group(1)) if m else None
        
        if dataset_id is not None:
            prefix = f"Dataset{dataset_id:03d}_"
            for k in PREDICT_CONFIGS.keys():
                if k.startswith(prefix):
                    dataset_key_for_config = k
                    break
                    
    if dataset_key_for_config in PREDICT_CONFIGS:
        cfg = PREDICT_CONFIGS[dataset_key_for_config].get(args.data_mode, {})
    
    # Override with CLI
    input_path = Path(args.input_folder) if args.input_folder else Path(cfg.get("input_path"))
    output_path = Path(args.output_folder) if args.output_folder else Path(cfg.get("output_path"))
    
    model_type = args.model or cfg.get("model_type") or "3d_fullres"
    fold = args.fold if args.fold is not None else cfg.get("fold", 0)
    file_type = args.file_type or cfg.get("file_type") or "nrrd"
    trainer = args.trainer or cfg.get("trainer") or "nnUNetTrainer"
    plans = args.plans or cfg.get("plans") or "nnUNetPlans"
    prev_stage_predictions = args.prev_stage_predictions or cfg.get("pred_stage_path")

    if not input_path or not output_path or not DATASET_NAME:
        raise ValueError("Missing required arguments: --input_folder, --output_folder, or --task.")

    # Execute Prediction
    success = predict_with_nnunet(
        input_path, output_path, args.output_csv or '',
        model_type, DATASET_NAME, fold, file_type,
        folds=args.folds,
        prev_stage_predictions=prev_stage_predictions,
        continue_prediction=args.continue_prediction,
        num_processes_preprocessing=args.npp,
        num_processes_saving=args.nps,
        trainer=trainer,
        plans=plans,
        use_best_config=args.use_best_config
    )

    if success:
        logging.info("Prediction pipeline finished successfully.")
    else:
        logging.error("Prediction pipeline failed.")

if __name__ == "__main__":
    main()
