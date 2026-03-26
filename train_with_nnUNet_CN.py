"""
nnUNet 训练脚本

两阶段流程:
  1. Experiment planning and preprocessing: nnUNetv2_plan_and_preprocess
  2. Training: nnUNetv2_train

当用户仅传入 -d/--dataset 时，从 configs/train_config.py 的 TRAIN_DATASET_CONFIGS 读取
configurations、folds、train_args。命令行参数可覆盖 config 中的值。

多 configuration 按顺序训练（cascade 需在 lowres 之后）。
多 fold 使用不同 GPU 并行训练。
tr/plans 可为 list，将训练 num_trainers * num_plans 种组合。
"""

import os
import re
import logging
import subprocess
import argparse
from pathlib import Path
import time

from .utils.logging_utils import setup_logging
from .utils.train_utils import *
from .configs.train_config import TRAIN_DATASET_CONFIGS, DEFAULT_FOLDS


# nnUNet 环境变量（与 predict_with_nnUNet.py 一致）
os.environ['nnUNet_raw'] = "/your/path/to/nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "/your/path/to/nnUNet_preprocessed" 
os.environ['nnUNet_results'] = "/your/path/to/nnUNet_results"


# cascade 模型依赖顺序：3d_lowres 必须在 3d_cascade_fullres 之前
CONFIG_ORDER = ["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres", ]


# =============================================================================
# Stage 1: Experiment planning and preprocessing
# =============================================================================


def run_plan_and_preprocess(dataset_id, verify_integrity=True):
    """运行 nnUNetv2_plan_and_preprocess。只传 dataset id。"""
    cmd = ["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id)]
    if verify_integrity:
        cmd.append("--verify_dataset_integrity")
    logging.info("[Plan] 执行: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, text=True)
        logging.info("[Plan] 完成")
        return True
    except subprocess.CalledProcessError as e:
        logging.error("[Plan] 失败: %s", e.stderr or str(e))
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
    多 configuration 按顺序训练，cascade 在 lowres 之后。
    每个 configuration 内，对每个 (trainer, plan) 组合，在其下训练所有 fold。
    单卡：fold 顺序训练，输出不混叠。
    多卡：不同 fold 在不同 GPU 上并行训练。
    只传 dataset id。
    """
    configs_sorted = sort_configurations(configurations)
    try:
        import torch
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        gpu_count = 0
    if num_gpus_available is not None:
        gpu_count = min(num_gpus_available, gpu_count) if gpu_count else num_gpus_available
    if gpu_count <= 0:
        logging.warning("未检测到 GPU，将使用 CPU 训练")
        gpu_count = 1

    for configuration in configs_sorted:
        logging.info("[Train] 开始配置: %s", configuration)
        if not folds or not trainer_plan_combinations:
            continue

        for combo_idx, (trainer, plan) in enumerate(trainer_plan_combinations):
            logging.info("[Train] tr=%s p=%s，训练 folds %s", trainer, plan, folds)

            if gpu_count == 1:
                for fold_idx, fold in enumerate(folds):
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = "0"
                    cmd = build_train_cmd(dataset_id, configuration, fold, train_args, trainer=trainer, plan=plan)
                    if combo_idx == 0 and fold_idx == 0:
                        logging.info("[Train] 单卡模式，顺序训练。先运行 fold %s 完成数据解压", fold)
                    else:
                        logging.info("[Train] fold %s", fold)
                    subprocess.run(cmd, env=env, check=True)
            else:
                fold0 = folds[0]
                rest_folds = folds[1:]
                is_first_combo = combo_idx == 0
                env0 = os.environ.copy()
                env0["CUDA_VISIBLE_DEVICES"] = "0"
                cmd0 = build_train_cmd(dataset_id, configuration, fold0, train_args, trainer=trainer, plan=plan)
                if is_first_combo:
                    logging.info("[Train] 多卡模式，先运行 fold %s 完成数据解压，等待 %s 秒", fold0, wait_for_extraction_seconds)
                proc0 = subprocess.Popen(cmd0, env=env0, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # 并行训练不打印到终端，避免输出混叠
                if is_first_combo:
                    time.sleep(wait_for_extraction_seconds)

                if proc0.poll() is not None:
                    proc0.wait()
                    if proc0.returncode != 0:
                        raise RuntimeError(f"tr={trainer} p={plan} fold {fold0} 训练失败")
                else:
                    logging.info("[Train] fold %s 已启动，并行启动其余 fold", fold0)

                procs_info = [(proc0, fold0, 0)]
                for i, fold in enumerate(rest_folds):
                    gpu_id = (i + 1) % gpu_count
                    env_i = os.environ.copy()
                    env_i["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    cmd_i = build_train_cmd(dataset_id, configuration, fold, train_args, trainer=trainer, plan=plan)
                    proc = subprocess.Popen(cmd_i, env=env_i, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # 并行训练不打印到终端，避免输出混叠
                    procs_info.append((proc, fold, gpu_id))
                    logging.info("[Train] 已启动 fold %s 在 GPU %s", fold, gpu_id)

                for proc, fold_val, _ in procs_info:
                    proc.wait()
                    if proc.returncode != 0:
                        raise RuntimeError(f"tr={trainer} p={plan} fold {fold_val} 训练失败")
            logging.info("[Train] tr=%s p=%s 全部 %d 个 fold 已完成", trainer, plan, len(folds))

    logging.info("[Train] 所有训练完成")


# =============================================================================
# Main: 参数解析与调度
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="nnUNet 训练脚本。仅传 -d 时从 train_config 读取参数。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例（在 code 目录下运行）:
  # 使用 config 全部参数
  python -m train_code.train_with_nnUNet -d Dataset043_Osteosclerosis_small_region

  # 仅 plan
  python -m train_code.train_with_nnUNet -d Dataset043_Osteosclerosis_small_region --plan_only

  # 仅 train，覆盖部分参数
  python -m train_code.train_with_nnUNet -d Dataset043_Osteosclerosis_small_region --train_only -c 2d -f 0 1 -tr nnUNetTrainerNoMirroring

  # 多 trainer×plan 组合探索（2×2=4 种组合）
  python -m train_code.train_with_nnUNet -d Dataset043 -tr nnUNetTrainer nnUNetTrainerNoMirroring -p nnUNetPlans nnUNetResEncUNetPlans_49G
        """,
    )

    parser.add_argument("-d", "--dataset", type=str, required=True, help="数据集名称或 ID，必填")

    # Plan 阶段
    parser.add_argument("--plan_only", action="store_true", help="仅执行 plan_and_preprocess")
    parser.add_argument("--no_verify_dataset_integrity", action="store_true", help="跳过 plan 的 --verify_dataset_integrity")
    parser.add_argument("--train_only", action="store_true", help="仅训练，跳过 plan（需已预处理）")

    # Train 阶段：大部分参数与 nnUNetv2_train 对齐（-tr -p -pretrained_weights --npz --c --val --val_best --disable_checkpointing -device）
    parser.add_argument("-c", "--configurations", type=str, nargs="+", help="配置列表，如 2d 3d_fullres（默认从 config）")
    parser.add_argument("-f", "--folds", type=int, nargs="+", help="fold 列表，如 0 1 2 3 4（默认从 config）")
    parser.add_argument("-tr", "--trainer", type=str, nargs="+", help="nnUNet -tr，可为多个，与 -p 组合训练")
    parser.add_argument("-p", "--plans", type=str, nargs="+", help="nnUNet -p，可为多个，与 -tr 组合训练")
    parser.add_argument("--pretrained_weights", type=str, help="nnUNet -pretrained_weights，预训练权重路径")
    parser.add_argument(
        "--num_gpus",
        type=int,
        help="用于本脚本并行训练 folds 的可用 GPU 数（调度用），不会传递给 nnUNetv2_train 的 -num_gpus",
    )
    parser.add_argument("--npz", action="store_true", help="nnUNet --npz，保存 softmax 用于 ensemble")
    parser.add_argument("--continue_train", action="store_true", help="nnUNet --c，从最新 checkpoint 继续")
    parser.add_argument("--val", dest="val_only", action="store_true", help="nnUNet --val，仅运行 validation")
    parser.add_argument("--val_best", action="store_true", help="nnUNet --val_best")
    parser.add_argument("--disable_checkpointing", action="store_true", help="nnUNet --disable_checkpointing")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "mps"], default=None, help="nnUNet -device")

    parser.add_argument("--wait_extraction", type=int, default=300, help="首 fold 启动后等待数据解压的秒数")

    args = parser.parse_args()

    setup_logging()
    dataset_name_or_id = (args.dataset or "").strip()
    if not dataset_name_or_id:
        raise ValueError("必须提供 --dataset")

    if dataset_name_or_id.isdigit():
        dataset_id = int(dataset_name_or_id)
    else:
        m = re.match(r"^Dataset(\d+)_", dataset_name_or_id, re.IGNORECASE)
        if m:
            dataset_id = int(m.group(1))
        else:
            raise ValueError(f"无法解析 dataset id: {dataset_name_or_id}，需为数字或 DatasetXXX_ 格式")

    config = TRAIN_DATASET_CONFIGS.get(args.dataset)
    if not config and dataset_name_or_id.isdigit():
        for k, v in TRAIN_DATASET_CONFIGS.items():
            if k.startswith(f"Dataset{dataset_id:03d}_"):
                config = v
                break
    if not config:
        raise ValueError(f"未找到数据集 {args.dataset} 的配置，请在 configs/train_config.py 中添加")

    configurations = args.configurations or config.get("configurations", ["2d", "3d_fullres"])
    folds = args.folds if args.folds is not None else config.get("folds", list(DEFAULT_FOLDS))
    # configs/train_config.py 里 folds 既可能是 int 也可能是 list，这里统一归一化为 list[int]
    if folds is None:
        folds = []
    elif isinstance(folds, int):
        folds = [folds]
    elif isinstance(folds, (tuple, set)):
        folds = list(folds)
    verify_integrity = not args.no_verify_dataset_integrity and config.get("verify_dataset_integrity", True)

    train_args_from_config = config.get("train_args", {})
    train_args_from_cli = parse_train_args_from_cli(args)
    if args.npz:
        train_args_from_cli["npz"] = True
    train_args = {**train_args_from_config, **train_args_from_cli}

    trainer_plan_combinations = get_trainer_plan_combinations(
        train_args.get("tr"), train_args.get("p")
    )
    logging.info("[Train] trainer×plan 组合数: %d", len(trainer_plan_combinations))
    for tr, p in trainer_plan_combinations:
        logging.info("  - tr=%s p=%s", tr, p)

    if not args.train_only:
        try:
            run_plan_and_preprocess(dataset_id, verify_integrity=verify_integrity)
        except Exception as e:
            logging.error("plan_and_preprocess 失败: %s", e)
            raise

    if not args.plan_only:
        try:
            run_training(
                dataset_id,
                configurations=configurations,
                folds=folds,
                train_args=train_args,
                trainer_plan_combinations=trainer_plan_combinations,
                num_gpus_available=args.num_gpus,
                wait_for_extraction_seconds=args.wait_extraction,
            )
        except Exception as e:
            logging.error("训练失败: %s", e)
            raise

    logging.info("nnUNet 训练流程完成")


if __name__ == "__main__":
    main()
