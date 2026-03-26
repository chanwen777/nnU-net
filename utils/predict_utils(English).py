import os
import re
import logging
import shlex
from pathlib import Path
from typing import List, Optional, Tuple
import shutil
import tempfile

# Valid nnU-Net configuration types for filtering
VALID_CONFIGURATIONS = ["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"]

# =============================================================================
# Dataset Name Parsing
# =============================================================================

def get_dataset_name(task_name: str) -> str:
    """
    Converts a task_name into the standard nnU-Net dataset format (DatasetXXX_Name).
    
    If provided with an ID (e.g., '38') or an incomplete name (e.g., 'Dataset038'), 
    it searches nnUNet_results for a matching folder starting with 'Dataset038_'.
    """
    task_name = str(task_name).strip()
    
    # Check if already in standard format
    if task_name.startswith("Dataset") and "_" in task_name and re.match(r"^Dataset\d{3}_.+", task_name, re.I):
        return task_name
        
    dataset_id = None
    if task_name.isdigit():
        dataset_id = int(task_name)
    else:
        m = re.match(r"^Dataset(\d+)$", task_name, re.IGNORECASE)
        if m:
            dataset_id = int(m.group(1))
            
    if dataset_id is not None:
        prefix = f"Dataset{dataset_id:03d}_"
        nnunet_results = os.environ.get("nnUNet_results")
        if nnunet_results:
            results_dir = Path(nnunet_results)
            if results_dir.exists():
                candidates = [d.name for d in results_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
                if len(candidates) == 1:
                    return candidates[0]
                if len(candidates) > 1:
                    logging.warning("Multiple datasets match ID %s, using the first: %s", task_name, candidates[0])
                    return candidates[0]
        return f"Dataset{dataset_id:03d}"
    
    return task_name


# =============================================================================
# Trained Configuration Discovery
# =============================================================================

# nnU-Net requirement for find_best_configuration/ensemble: 
# All 5 folds must exist, and each fold/validation must contain .npz files (trained with --npz).
REQUIRED_FOLDS = [0, 1, 2, 3, 4]


def discover_trained_configurations(dataset_name_or_id: str) -> List[str]:
    """
    Scans nnUNet_results/DatasetXXX/ for trained model folders (Format: Trainer__Plans__Configuration).
    
    Filters for valid configurations (2d, 3d_fullres, etc.) and ensures:
    1. Fold structures 0-4 are complete.
    2. At least one .npz file exists in each fold's validation directory.
    3. Skips folders with the 'ensemble___' prefix.
    """
    nnunet_results = os.environ.get("nnUNet_results")
    if not nnunet_results:
        return []

    dataset_name = get_dataset_name(dataset_name_or_id)
    dataset_dir = Path(nnunet_results) / dataset_name
    if not dataset_dir.exists():
        return []

    configs = set()
    for item in dataset_dir.iterdir():
        if not item.is_dir() or item.name.startswith("ensemble___"):
            continue
            
        matched_cfg = None
        for valid_cfg in VALID_CONFIGURATIONS:
            if valid_cfg in item.name:
                matched_cfg = valid_cfg
                break
        
        if matched_cfg is None:
            continue
            
        try:
            # Validate presence of 5 folds and .npz validation files
            for f in REQUIRED_FOLDS:
                val_dir = item / f"fold_{f}" / "validation"
                if not val_dir.is_dir() or not list(val_dir.glob("*.npz")):
                    break
            else:
                # Executes if the loop finishes without 'break' (all folds valid)
                configs.add(matched_cfg)
        except OSError:
            continue

    return sorted(configs, key=lambda c: VALID_CONFIGURATIONS.index(c) if c in VALID_CONFIGURATIONS else 999)


# =============================================================================
# Inference Instructions Parsing (Step 1 Output for Step 2/3)
# =============================================================================

def parse_inference_instructions(
    task_name: str,
) -> Tuple[Optional[List[List[str]]], Optional[List[str]], Optional[List[str]]]:
    """
    Parses nnUNet's 'inference_instructions.txt' to extract prediction and post-processing command templates.
    
    Returns:
        tuple: (inference_cmd_parts_list, ensemble_cmd_parts, postprocessing_cmd_parts)
        - cmd_parts_list: shlex-parsed predict commands with placeholders.
        - ensemble_parts: Parsed ensemble command (if exists).
        - postprocessing_parts: Parsed post-processing command.
    """
    nnunet_results = os.environ.get("nnUNet_results")
    if not nnunet_results:
        logging.warning("nnUNet_results env variable not set.")
        return None, None, None

    dataset_name = get_dataset_name(task_name)
    candidates = [
        Path(nnunet_results) / dataset_name / "inference_instructions.txt",
        Path(nnunet_results) / dataset_name / "inference_instruction.txt",
    ]
    
    instructions_path = next((p for p in candidates if p.exists()), None)
    if not instructions_path:
        logging.warning(f"Instructions file not found in: {candidates}")
        return None, None, None

    content = instructions_path.read_text(encoding="utf-8", errors="replace")
    inference_cmds = [line.strip() for line in content.splitlines() if line.strip().startswith("nnUNetv2_predict ")]
    ensemble_cmd = next((line.strip() for line in content.splitlines() if line.strip().startswith("nnUNetv2_ensemble ")), None)
    postprocessing_cmd = next((line.strip() for line in content.splitlines() if line.strip().startswith("nnUNetv2_apply_postprocessing ")), None)

    if not inference_cmds:
        logging.warning("No nnUNetv2_predict commands found in instructions.")
        return None, None, None

    try:
        inference_parts_list = [shlex.split(cmd) for cmd in inference_cmds]
        ensemble_parts = shlex.split(ensemble_cmd) if ensemble_cmd else None
        pp_parts = shlex.split(postprocessing_cmd) if postprocessing_cmd else None
        return inference_parts_list, ensemble_parts, pp_parts
    except Exception as e:
        logging.warning(f"Error parsing commands: {e}")
        return None, None, None

def replace_instruction_placeholders(parts: List[str], replacement_map: dict) -> List[str]:
    """Replaces command placeholders (e.g., INPUT_FOLDER) with actual system paths."""
    return [str(replacement_map.get(p, p)) for p in parts]

def build_postprocessing_cmd_from_instructions(
    pp_parts: List[str],
    output_folder: str,
    output_folder_pp: str,
) -> List[str]:
    """Populates post-processing placeholders with actual output and post-processed paths."""
    return replace_instruction_placeholders(
        pp_parts,
        {"OUTPUT_FOLDER": str(output_folder), "OUTPUT_FOLDER_PP": str(output_folder_pp)}
    )

def get_cmd_option_value(cmd_parts: List[str], option: str) -> Optional[str]:
    """Retrieves the value of a specific CLI option (e.g., '-c')."""
    if option in cmd_parts:
        idx = cmd_parts.index(option)
        if idx + 1 < len(cmd_parts):
            return cmd_parts[idx + 1]
    return None

def set_or_append_cmd_option(cmd_parts: List[str], option: str, value: str) -> List[str]:
    """Updates an existing CLI option or appends it if missing."""
    result = list(cmd_parts)
    if option in result:
        idx = result.index(option)
        if idx + 1 < len(result):
            result[idx + 1] = str(value)
        else:
            result.append(str(value))
    else:
        result.extend([option, str(value)])
    return result

def append_flag_if_missing(cmd_parts: List[str], flag: str) -> List[str]:
    """Appends a boolean flag (e.g., '--save_probabilities') if not present."""
    result = list(cmd_parts)
    if flag not in result:
        result.append(flag)
    return result

def is_predict_command(cmd_parts: List[str]) -> bool:
    """Checks if the command is a nnUNetv2_predict call."""
    return len(cmd_parts) > 0 and cmd_parts[0] == "nnUNetv2_predict"

def predict_cmd_requires_prev_stage(cmd_parts: List[str]) -> bool:
    """Determines if the prediction command belongs to a cascade model requiring previous stages."""
    config = get_cmd_option_value(cmd_parts, "-c")
    return (
        config == "3d_cascade_fullres"
        or "-prev_stage_predictions" in cmd_parts
        or "OUTPUT_FOLDER_PREV_STAGE" in cmd_parts
    )

def build_best_config_base_commands(
    inference_parts_list: List[List[str]],
    ensemble_parts: Optional[List[str]],
    input_folder: Path,
    output_folder: Path,
    prev_stage_runtime_path: Optional[str],
) -> Tuple[List[List[str]], List[Path]]:
    """
    Maps ensemble placeholders (OUTPUT_FOLDER_MODEL_X) to physical subdirectories 
    within the output_folder to support --continue_prediction across runs.
    """
    replacement_map = {
        "INPUT_FOLDER": str(input_folder),
        "OUTPUT_FOLDER": str(output_folder),
    }
    if prev_stage_runtime_path:
        replacement_map["OUTPUT_FOLDER_PREV_STAGE"] = str(prev_stage_runtime_path)

    extra_tmp_dirs = []
    model_placeholder_re = re.compile(r"^OUTPUT_FOLDER_MODEL_(\d+)$")

    def ensure_output_placeholder(placeholder: str):
        if placeholder in replacement_map:
            return
        m = model_placeholder_re.match(placeholder)
        if m:
            # Map model-specific ensemble outputs to fixed subfolders
            replacement_map[placeholder] = str(output_folder / f"_model_{m.group(1)}")
            return
        # Use temp directories for unidentified placeholders
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"nnunet_{placeholder.lower()}_"))
        replacement_map[placeholder] = str(tmp_dir)
        extra_tmp_dirs.append(tmp_dir)

    for cmd_parts in (inference_parts_list + ([ensemble_parts] if ensemble_parts else [])):
        for token in cmd_parts:
            if token.startswith("OUTPUT_FOLDER") and token not in ("OUTPUT_FOLDER_PP", "OUTPUT_FOLDER_PREV_STAGE"):
                ensure_output_placeholder(token)

    base_cmds = [replace_instruction_placeholders(cmd, replacement_map) for cmd in inference_parts_list]
    if ensemble_parts:
        base_cmds.append(replace_instruction_placeholders(ensemble_parts, replacement_map))
    return base_cmds, extra_tmp_dirs


def build_custom_base_command(
    task_name: str,
    model_type: str,
    trainer: str,
    plans: str,
    fold: int,
    folds: Optional[List[int]],
    input_folder: Path,
    output_folder: Path,
) -> List[str]:
    """Constructs a manual (custom) predict command with specified model parameters."""
    cmd = [
        "nnUNetv2_predict",
        "-d", str(task_name),
        "-i", str(input_folder),
        "-o", str(output_folder),
        "-c", str(model_type),
        "-tr", trainer,
        "-p", plans,
    ]
    if folds and isinstance(folds, (list, tuple)):
        cmd.extend(["-f"] + [str(f) for f in folds])
    else:
        cmd.extend(["-f", str(fold)])
    return cmd


def append_runtime_predict_options(
    cmd_parts: List[str],
    prev_stage_runtime_path: Optional[str],
    continue_prediction: bool,
    num_processes_preprocessing: int,
    num_processes_saving: int,
    require_probabilities: bool,
) -> List[str]:
    """
    Appends runtime-specific execution arguments to the predict command template.
    Handles GPU device setting, parallel processing, and cascade dependencies.
    """
    if not is_predict_command(cmd_parts):
        return list(cmd_parts)

    result = list(cmd_parts)
    result = set_or_append_cmd_option(result, "-device", "cuda")
    
    if require_probabilities:
        result = append_flag_if_missing(result, "--save_probabilities")
    if continue_prediction:
        result = append_flag_if_missing(result, "--continue_prediction")
    if num_processes_preprocessing > 0:
        result = set_or_append_cmd_option(result, "-npp", str(num_processes_preprocessing))
    if num_processes_saving > 0:
        result = set_or_append_cmd_option(result, "-nps", str(num_processes_saving))

    if predict_cmd_requires_prev_stage(result):
        if prev_stage_runtime_path:
            result = set_or_append_cmd_option(result, "-prev_stage_predictions", str(prev_stage_runtime_path))
        else:
            logging.warning("Cascade command detected, but no prev_stage_predictions provided for this case.")
    return result


def build_case_commands(
    use_best_config: bool,
    inference_parts_list: Optional[List[List[str]]],
    ensemble_parts: Optional[List[str]],
    task_name: str,
    model_type: str,
    trainer: str,
    plans: str,
    fold: int,
    folds: Optional[List[int]],
    input_folder: Path,
    output_folder: Path,
    prev_stage_runtime_path: Optional[str],
    continue_prediction: bool,
    num_processes_preprocessing: int,
    num_processes_saving: int,
) -> Tuple[List[List[str]], List[Path]]:
    """
    Generates the final list of executable commands (predict/ensemble) for a given case.
    
    1. Builds base commands (template replacement or custom definition).
    2. Appends runtime-specific flags (GPU, multiprocessing, resume options).
    """
    if use_best_config and inference_parts_list:
        base_cmds, extra_tmp_dirs = build_best_config_base_commands(
            inference_parts_list, ensemble_parts, input_folder, output_folder, prev_stage_runtime_path
        )
        require_probabilities = ensemble_parts is not None
    else:
        base_cmds = [
            build_custom_base_command(
                task_name, model_type, trainer, plans, fold, folds, input_folder, output_folder
            )
        ]
        extra_tmp_dirs = []
        require_probabilities = False

    final_cmds = [
        append_runtime_predict_options(
            cmd, prev_stage_runtime_path, continue_prediction, 
            num_processes_preprocessing, num_processes_saving, require_probabilities
        )
        for cmd in base_cmds
    ]
    return final_cmds, extra_tmp_dirs
