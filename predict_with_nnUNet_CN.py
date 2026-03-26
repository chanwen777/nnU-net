"""
使用训练好的nnUNet模型对数据进行批量预测，并实时显示进度
保持输出目录结构与输入一致。

nnUNet 推理流程三步骤：
1. Automatically determine the best configuration（调用 nnUNetv2_find_best_configuration）
2. Run inference
3. Apply postprocessing（仅在使用 best configuration 时）

支持两种模式：
- --use_best_config: 从 inference_instructions.txt 解析推理和后处理命令（txt 不存在时自动执行 find_best_configuration）
- 自定义模式: 用户/ config 指定 model_type、fold、trainer、plans 等，仅推理不后处理
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
from .utils.logging_utils import setup_logging
from .utils.run_utils import monitor_output_folder
from .utils.predict_utils import *
from .configs.predict_data_config import PREDICT_CONFIGS


os.environ['nnUNet_raw'] = "/your/path/to/nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "/your/path/to/nnUNet_preprocessed" 
os.environ['nnUNet_results'] = "/your/path/to/nnUNet_results"


# 有效的 nnUNet configuration 类型（用于过滤）
VALID_CONFIGURATIONS = ["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"]


# =============================================================================
# Step 1: Automatically determine the best configuration
# =============================================================================


def run_find_best_configuration(
    dataset_name_or_id: str,
    configurations: Optional[List[str]] = None,
) -> bool:
    """
    调用 nnUNetv2_find_best_configuration 确定 best configuration。
    若未传入 configurations，则从 nnUNet_results/DatasetXXX/ 扫描已训练模型文件夹自动推断（仅 -c 参数）。
    成功后会在 nnUNet_results/DatasetXXX/inference_instructions.txt 生成推理与后处理指令。
    """
    if configurations is None:
        configurations = discover_trained_configurations(dataset_name_or_id)
        if not configurations:
            raise RuntimeError("未在 nnUNet_results 中发现已训练的模型，无法确定configuration，请先完成模型训练。")

    cmd = ["nnUNetv2_find_best_configuration", str(get_dataset_name(dataset_name_or_id))]
    cmd.extend(["-c"] + configurations)
    logging.info("执行 Step 1: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, text=True)
        logging.info("find_best_configuration 完成")
        return True
    except subprocess.CalledProcessError as e:
        logging.error("find_best_configuration 失败: %s", e.stderr or str(e))
        return False


# =============================================================================
# Step 3: Apply postprocessing（仅在使用 best configuration 时）
# =============================================================================


def run_postprocessing(
    output_folder: Path,
    pp_parts: List[str],
    output_folder_pp: Optional[Path] = None,
) -> bool:
    """
    对推理输出目录执行 nnUNetv2_apply_postprocessing。
    output_folder_pp 默认为 output_folder + "_postprocessed"。
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
    logging.info("执行后处理: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logging.info("后处理完成，输出目录: %s", output_folder_pp)
        return True
    except subprocess.CalledProcessError as e:
        logging.error("后处理失败: %s", e.stderr)
        return False


# =============================================================================
# Step 2: Run inference（支持 best config 与自定义参数）
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
    使用训练好的nnUNet模型对数据进行批量预测，并实时显示进度
    保持输出目录结构与输入一致。
    
    参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        output_csv_path: 输出CSV路径（可选）
        model_type: 模型类型（如 '3d_fullres', '3d_cascade_fullres'）
        task_name: 任务名称（如 'Dataset038_Spine_Fracture'）
        fold: fold编号（单个fold，如果folds不为None则忽略此参数）
        file_type: 文件类型（'nrrd' 或 'bmp'）
        folds: fold列表（如 [0,1,2,3,4]），如果提供则使用此参数而不是fold
        prev_stage_predictions: 前一阶段的预测结果路径（用于cascade模型）
        continue_prediction: 是否继续预测（跳过已存在的预测文件）
        num_processes_preprocessing: 预处理进程数（-npp参数）
        num_processes_saving: 后处理进程数（-nps参数）
        trainer: 训练器名称（默认: 'nnUNetTrainer'）
        plans: 计划名称（默认: 'nnUNetPlans'）
        use_best_config: 若为 True，从 inference_instructions.txt 解析推理与后处理命令（txt 不存在时自动执行 find_best_configuration）
    """
    try:
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        if not input_folder.exists():
            raise ValueError(f"输入文件夹不存在: {input_folder}")

        input_files = []
        
        # 如果指定了 file_type，优先按指定格式查找
        if file_type:
            ext = file_type.lower()
            input_files = list(input_folder.glob(f"*_0000.{ext}"))
            if not input_files:
                input_files = list(input_folder.rglob(f"*_0000.{ext}"))
            if input_files:
                logging.info(f"按配置查找 {ext.upper()} 格式，找到 {len(input_files)} 个文件")
        else:
            raise ValueError(f"未指定文件类型，请使用 --file_type 指定文件类型")
        

        logging.info(f"找到 {len(input_files)} 个待预测的文件")

        # 收集需要处理的文件和输出路径
        files_to_process = []
        output_paths = []
        for input_file in input_files:
            rel_path = input_file.relative_to(input_folder)
            input_stem = input_file.stem
            if input_stem.endswith("_0000"):
                output_stem = input_stem[:-5]
            else:
                output_stem = input_stem
            
            # 输出格式与检测到的输入格式一致
            output_ext = f".{file_type}"
            output_rel_path = rel_path.with_name(output_stem + output_ext)

            expected_output_path = output_folder / output_rel_path
            expected_output_path.parent.mkdir(parents=True, exist_ok=True)

            files_to_process.append(input_file)
            output_paths.append(expected_output_path)

        if not files_to_process:
            logging.info("未找到待预测文件（输入目录下无 *_0000.%s 格式文件）", file_type)
            return True
        logging.info(f"需要处理 {len(files_to_process)} 个文件（跳过由 nnUNetv2_predict --continue_prediction 控制）")
        
        # 将待预测文件按其所在目录分组；对每个目录调用一次 nnUNetv2_predict（避免逐病例临时目录与 move）
        grouped_input_folders = {}
        for input_file in files_to_process:
            grouped_input_folders.setdefault(input_file.parent, 0)
            grouped_input_folders[input_file.parent] += 1
        input_folders_sorted = sorted(grouped_input_folders.keys(), key=lambda p: str(p))
        logging.info("按输入目录分组: %d 组", len(input_folders_sorted))
        for p in input_folders_sorted:
            rel_dir = p.relative_to(input_folder) if p != input_folder else Path(".")
            logging.info("  - %s（%d 文件） -> 输出子目录 %s", str(p), grouped_input_folders[p], str(output_folder / rel_dir))


        # use_best_config: 解析 inference_instructions.txt
        inference_parts_list = None
        ensemble_parts = None
        pp_parts = None
        if use_best_config:
            inference_parts_list, ensemble_parts, pp_parts = parse_inference_instructions(task_name)
            if inference_parts_list is None:
                logging.info("inference_instructions.txt 不存在，自动执行 find_best_configuration")
                if not run_find_best_configuration(task_name):
                    raise ValueError("自动执行 find_best_configuration 失败")
                inference_parts_list, ensemble_parts, pp_parts = parse_inference_instructions(task_name)
            if inference_parts_list is None:
                raise ValueError("use_best_config=True 但无法解析 inference_instructions.txt")
            if ensemble_parts is not None:
                logging.info("已从 inference_instructions.txt 加载 best configuration（ensemble）")
            else:
                logging.info("已从 inference_instructions.txt 加载 best configuration（single model）")
        
        # 监控输出目录，显示进度
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
        # 按目录批量推理：nnUNet 原生 continue_prediction 生效，且避免大量临时目录/子进程开销
        idx_group = 0
        for in_dir in input_folders_sorted:
            idx_group += 1
            rel_dir = in_dir.relative_to(input_folder) if in_dir != input_folder else Path(".")
            out_dir = output_folder / rel_dir
            out_dir.mkdir(parents=True, exist_ok=True)

            prev_stage_path = str(prev_stage_predictions) if prev_stage_predictions is not None else None
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
                    logging.info(
                        f"[Group {idx_group}/{len(input_folders_sorted)}] 执行命令({cmd_idx}/{len(cmds_to_run)}): {' '.join(cmd)}"
                    )
                    subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=True
                    )
            except subprocess.CalledProcessError as e:
                any_group_failed = True
                log_path = None
                try:
                    from .utils.logging_utils import LOGS_DIR
                    import __main__
                    log_path = LOGS_DIR / f"{Path(__main__.__file__).stem}.log"
                except Exception:
                    pass

                error_msg = f"预测命令执行失败: {e.stderr}"
                logging.error(error_msg)
                logging.error(f"失败的输入目录: {in_dir}")
                logging.error(f"失败的输出目录: {out_dir}")
                if log_path and log_path.exists():
                    logging.error(f"详细错误信息已记录到日志文件: {log_path}")
                    print(f"\n❌ 预测失败！日志文件位置: {log_path}")
            finally:
                for p in extra_tmp_dirs:
                    if p.exists():
                        try:
                            shutil.rmtree(p)
                        except Exception:
                            pass

        # 批量跑完后，按期望输出路径汇总日志（不再逐病例打印）
        ok_count = sum(1 for p in output_paths if p.exists())
        logging.info("输出文件检查: %d/%d 已存在（continue_prediction 时可能跳过生成）", ok_count, len(output_paths))
        elapsed = time.time() - start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        logging.info(f"推理总运行时间: {elapsed_str}")
        stop_event.set()
        monitor_thread.join(timeout=1.0)

        if any_group_failed:
            logging.error("部分分组推理失败，整体视为失败")
            return False

        # use_best_config 时执行后处理（按分组执行，与推理的 out_dir 一一对应，避免多分组时根目录无文件导致后处理落空）
        if use_best_config and pp_parts is not None:
            pp_root = Path(str(output_folder) + "_postprocessed")
            logging.info("开始执行后处理（按分组）...")
            for in_dir in input_folders_sorted:
                rel_dir = in_dir.relative_to(input_folder) if in_dir != input_folder else Path(".")
                out_dir = output_folder / rel_dir
                pp_out = pp_root / rel_dir
                pp_out.mkdir(parents=True, exist_ok=True)
                if not run_postprocessing(out_dir, pp_parts, output_folder_pp=pp_out):
                    logging.error("后处理失败: %s -> %s", out_dir, pp_out)
                    return False
            logging.info("推理与后处理流程完成，后处理结果目录: %s", pp_root)

        return True
    except Exception as e:
        logging.error(f"预测过程发生错误: {str(e)}", exc_info=True)
        return False

def main():
    parser = argparse.ArgumentParser(
        description='使用nnUNet进行批量预测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
流程步骤：
  1. Determine best configuration（--use_best_config 时，txt 不存在则自动执行）
  2. Run inference
  3. Apply postprocessing（--use_best_config 时自动执行）

推荐运行示例（在 code 目录下运行）:
  # best config（单模型或 ensemble 自动解析）
  python -m train_code.predict_with_nnUNet --task Dataset038_Spine_Fracture --input_folder /path/to/input --output_folder /path/to/output --use_best_config --file_type nrrd --continue_prediction --npp 3 --nps 3

  # 自定义模式
  python -m train_code.predict_with_nnUNet --task Dataset038_Spine_Fracture --input_folder /path/to/input --output_folder /path/to/output --model 3d_fullres --folds 0 1 2 3 4 --trainer nnUNetTrainer --plans nnUNetPlans --file_type nrrd --npp 3 --nps 3

  # cascade（需提供前一阶段预测目录）
  python -m train_code.predict_with_nnUNet --task Dataset038_Spine_Fracture --input_folder /path/to/stage2_input --output_folder /path/to/output --use_best_config --prev_stage_predictions /path/to/stage1_pred --file_type nrrd --npp 3 --nps 3
        """,
    )
    # Step 1: Determine best configuration
    parser.add_argument('--task', type=str, required=True, help='[Step 1] 任务名称（如 Dataset035_Spine）')
    parser.add_argument('--use_best_config', action='store_true',
                        help='[Step 1] 使用 best configuration（txt 不存在时自动扫描已训练模型并执行 find_best_configuration）')
    parser.add_argument('--data_mode', type=str, default='train_data', choices=['train_data', 'valid_data'],
                        help='[Step 1] 从 config 加载时使用的数据模式（默认: train_data）')
    # Step 2: Run inference
    parser.add_argument('--input_folder', type=str, help='[Step 2] 输入文件夹路径')
    parser.add_argument('--output_folder', type=str, help='[Step 2] 输出文件夹路径')
    parser.add_argument('--model', type=str, default=None, help='[Step 2] 模型类型（默认从 config 或 3d_fullres）')
    parser.add_argument('--fold', type=int, default=None, help='[Step 2] fold编号（默认从 config 或 0），若指定--folds则忽略')
    parser.add_argument('--folds', type=int, nargs='+', default=None, help='[Step 2] fold列表（如: 0 1 2 3 4）')
    parser.add_argument('--trainer', type=str, default=None, help='[Step 2] 训练器名称（默认从 config 或 nnUNetTrainer）')
    parser.add_argument('--plans', type=str, default=None, help='[Step 2] 计划名称（默认从 config 或 nnUNetPlans）')
    parser.add_argument('--file_type', type=str, default=None, help='[Step 2] 文件类型（默认从 config 或 nrrd）')
    parser.add_argument('--prev_stage_predictions', type=str, default=None, help='[Step 2] 前一阶段预测路径（cascade 模型）')
    parser.add_argument('--continue_prediction', action='store_true', help='[Step 2] 继续预测（跳过已存在文件）')
    parser.add_argument('--npp', type=int, default=1, help='[Step 2] 预处理进程数（默认: 1）')
    parser.add_argument('--nps', type=int, default=1, help='[Step 2] 后处理进程数（默认: 1）')
    # 其他
    parser.add_argument('--output_csv', type=str, default='', help='输出CSV路径（可选）')

    args = parser.parse_args()

    setup_logging()

    # 参数来源：config 为底，命令行覆盖（与 train_with_nnUNet 一致）
    input_path = None
    output_path = None
    DATASET_NAME = get_dataset_name(args.task)
    output_csv_path = args.output_csv or ''
    continue_prediction = args.continue_prediction
    num_processes_preprocessing = args.npp
    num_processes_saving = args.nps
    use_best_config = args.use_best_config

    # 从 config 加载：PREDICT_CONFIGS 的 key 为完整名（如 Dataset038_Spine_Fracture）
    cfg = {}
    dataset_key_for_config = args.task
    if dataset_key_for_config and dataset_key_for_config not in PREDICT_CONFIGS:
        # 纯数字（如 38）或 Dataset038：按 DatasetXXX_ 前缀匹配第一个 config key
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
    if dataset_key_for_config and dataset_key_for_config in PREDICT_CONFIGS:
        cfg = PREDICT_CONFIGS[dataset_key_for_config].get(args.data_mode, {})
    if cfg:
        if input_path is None:
            input_path = cfg.get("input_path")
        if output_path is None:
            output_path = cfg.get("output_path")
        output_csv_path = output_csv_path or str(cfg.get("input_csv_path", ""))

    # 命令行覆盖（用户传入的优先）
    if args.input_folder:
        input_path = Path(args.input_folder)
    if args.output_folder:
        output_path = Path(args.output_folder)

    # 模型参数：config 为底，命令行覆盖，最后用默认值
    model_type = (args.model if args.model is not None else (cfg.get("model_type") if cfg else None)) or "3d_fullres"
    fold = args.fold if args.fold is not None else (cfg.get("fold", 0) if cfg else 0)
    folds = args.folds
    file_type = (args.file_type if args.file_type is not None else (cfg.get("file_type") if cfg else None)) or "nrrd"
    trainer = (args.trainer if args.trainer is not None else (cfg.get("trainer") if cfg else None)) or "nnUNetTrainer"
    plans = (args.plans if args.plans is not None else (cfg.get("plans") if cfg else None)) or "nnUNetPlans"
    prev_stage_predictions = args.prev_stage_predictions if args.prev_stage_predictions is not None else (cfg.get("pred_stage_path") if cfg else None)

    if not input_path or not output_path or not DATASET_NAME:
        raise ValueError("必须提供 --input_folder、--output_folder、--task，或通过 config 可解析的 --task")

    input_path = Path(input_path)
    output_path = Path(output_path)

    # 运行预测
    success = predict_with_nnunet(
        input_path,
        output_path,
        output_csv_path,
        model_type,
        DATASET_NAME,
        fold,
        file_type,
        folds=folds,
        prev_stage_predictions=prev_stage_predictions,
        continue_prediction=continue_prediction,
        num_processes_preprocessing=num_processes_preprocessing,
        num_processes_saving=num_processes_saving,
        trainer=trainer,
        plans=plans,
        use_best_config=use_best_config
    )

    if success:
        logging.info("预测流程成功完成")
    else:
        logging.error("预测流程失败")


if __name__ == "__main__":
    main() 
