# nnunet-pipeline

[![English](https://img.shields.io/badge/README-English-blue)](README.md)
[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-brightgreen)](README.zh-CN.md)

Research training & inference scripts based on **nnU-Net v2**, supporting:

- **Training**: `nnUNetv2_plan_and_preprocess` + `nnUNetv2_train`
- **Inference**: custom inference / automatic best configuration (parses `inference_instructions.txt`, supports single/ensemble)
- **Cascade**: when required by configuration, automatically appends `-prev_stage_predictions` to `nnUNetv2_predict` at runtime

---

## Quick start (run after forking)

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/nnunet-pipeline.git
   cd nnunet-pipeline
   ```

2. **Create an environment and install dependencies**
   ```bash
   conda create -n nnunet_env python=3.11 -y
   conda activate nnunet_env
   pip install -r requirements.txt
   ```
   > Install PyTorch/CUDA separately according to your machine (e.g. `pip install torch` or via conda).

3. **Install nnU-Net v2** (if not installed yet)
   ```bash
   pip install nnunetv2
   ```

4. **Configure nnU-Net data paths (required)**  
   Edit the following three lines near the top of `train_with_nnUNet.py` and `predict_with_nnUNet.py` to match your paths:
   ```python
   os.environ['nnUNet_raw'] = "/your/path/to/nnUNet_raw"
   os.environ['nnUNet_preprocessed'] = "/your/path/to/nnUNet_preprocessed"
   os.environ['nnUNet_results'] = "/your/path/to/nnUNet_results"
   ```
   Or use environment variables: `export nnUNet_raw=... nnUNet_preprocessed=... nnUNet_results=...` before running.

5. **How to run**  
   These scripts use relative imports, so you should run them as a Python package (`python -m ...`). This repo already includes `__init__.py`. Rename the repo folder to a valid Python package name (e.g. `nnunet_pipeline`), then run commands from the **parent directory**:
   ```bash
   # Clone with a valid directory name (avoid hyphens)
   git clone https://github.com/<your-username>/nnunet-pipeline.git nnunet_pipeline
   cd nnunet_pipeline
   # ... after installing deps and setting paths ...
   cd ..
   python -m nnunet_pipeline.train_with_nnUNet -d Dataset038_Spine_Fracture --plan_only
   python -m nnunet_pipeline.predict_with_nnUNet --task Dataset038_Spine_Fracture --input_folder ... --output_folder ... --use_best_config
   ```
   If you place this repo under `code/train_code/`, then run `python -m train_code.train_with_nnUNet ...` after `cd code`.

---

## Environment & dependencies

### 1) Create an environment

You can reuse an existing `nnunet_env`, or create a new Python 3.11 environment:

```bash
conda create -n nnunet_env python=3.11 -y
conda activate nnunet_env
```

### 2) Install dependencies

This repo provides `requirements.txt` (excluding `nvidia-*` for easier installation across different CUDA setups).

```bash
pip install -r requirements.txt
```

> It is recommended to install PyTorch/CUDA separately based on your machine/CUDA version (e.g. official wheels/conda).

### 3) Install/verify nnU-Net v2

Make sure `nnunetv2` is installed and the CLI commands are available:

```bash
nnUNetv2_predict -h
nnUNetv2_train -h
nnUNetv2_find_best_configuration -h
```

---

## nnU-Net paths (required)

nnU-Net v2 uses the following environment variables to locate data and results:

- `nnUNet_raw`
- `nnUNet_preprocessed`
- `nnUNet_results`

**Recommendation**: hardcode these paths at the top of the scripts to keep the environment consistent and avoid exporting every time.

- **Training**: in `train_with_nnUNet.py`, update the three `os.environ["nnUNet_raw"]`-like lines near the top.
- **Inference**: in `predict_with_nnUNet.py`, update the same three lines.

If you prefer environment variables, run before execution:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

---

## Config files

- `configs/train_config.py`
  - `TRAIN_DATASET_CONFIGS`: per-dataset training configs (configurations/folds/train_args)
- `configs/predict_data_config.py`
  - `PREDICT_CONFIGS`: per-dataset default inference paths and parameters (input/output/model/fold/file_type, etc.)

These scripts support: **config provides defaults, CLI args override config**.

---

## Training (`train_with_nnUNet`)

Entry: `train_with_nnUNet.py`

### 1) Plan & preprocess only

```bash
cd /path/to/code   # or the parent dir if using the `nnunet_pipeline` package name
conda activate nnunet_env

python -m train_code.train_with_nnUNet \
  -d Dataset038_Spine_Fracture \
  --plan_only
```

### 2) Train only (skip plan)

Use this when preprocessing has already been done:

```bash
cd /path/to/code
conda activate nnunet_env

python -m train_code.train_with_nnUNet \
  -d Dataset038_Spine_Fracture \
  --train_only \
  -c 3d_fullres \
  -f 0
```

### 3) Multiple folds / trainers / plans

- `-f/--folds` supports multiple folds
- `-tr/--trainer` and `-p/--plans` accept multiple values and will run the Cartesian product (trainer × plans)

Example:

```bash
python -m train_code.train_with_nnUNet \
  -d Dataset043_Osteosclerosis_small_region \
  --train_only \
  -c 2d \
  -f 0 1 \
  -tr nnUNetTrainer nnUNetTrainerNoMirroring \
  -p nnUNetPlans nnUNetResEncUNetPlans_49G
```

---

## Inference (`predict_with_nnUNet`)

Entry: `predict_with_nnUNet.py`

### Mode A: best configuration (recommended)

This mode reads/generates `nnUNet_results/DatasetXXX_*/inference_instructions.txt` and executes in order:

- one or more `nnUNetv2_predict`
- If the best is an ensemble: additionally runs `nnUNetv2_ensemble` (and automatically adds `--save_probabilities` to predict commands)
- Finally runs `nnUNetv2_apply_postprocessing`

```bash
cd /path/to/code
conda activate nnunet_env

python -m train_code.predict_with_nnUNet \
  --task Dataset038_Spine_Fracture \
  --input_folder /path/to/input \
  --output_folder /path/to/output \
  --use_best_config \
  --file_type nrrd \
  --continue_prediction \
  --npp 3 --nps 3
```

### Mode B: custom inference (manually specify model params)

```bash
python -m train_code.predict_with_nnUNet \
  --task Dataset038_Spine_Fracture \
  --input_folder /path/to/input \
  --output_folder /path/to/output \
  --model 3d_fullres \
  --folds 0 1 2 3 4 \
  --trainer nnUNetTrainer \
  --plans nnUNetPlans \
  --file_type nrrd \
  --npp 3 --nps 3
```

### Cascade inference (requires previous stage predictions)

If using `3d_cascade_fullres` (or if the command requires a previous stage), provide `--prev_stage_predictions`:

```bash
python -m train_code.predict_with_nnUNet \
  --task Dataset038_Spine_Fracture \
  --input_folder /path/to/stage2_input \
  --output_folder /path/to/output \
  --use_best_config \
  --prev_stage_predictions /path/to/stage1_pred \
  --file_type nrrd \
  --npp 3 --nps 3
```

---

## Input data naming convention

The inference input folder must contain `*_0000.<ext>` files (e.g. `case_0000.nrrd`).

- With `--file_type nrrd`, the script matches `*_0000.nrrd` in the input folder.

---

## FAQ

### How is the best configuration chosen?

`nnUNetv2_find_best_configuration` selects the best single/ensemble based on cross-validation results (Dice) and generates `inference_instructions.txt`.

---

## Citation

If you use this code or nnU-Net in your research, please cite:

1. Hanwen Cheng et al.Fracture-Adaptive Deep Learning for Opportunistic Volumetric Bone Mineral Density Assessment and Vertebral Fracture Prediction. submmitted.
2. Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature methods*, 18(2), 203-211.

---

## File index

- `train_with_nnUNet.py`: training entry (plan/preprocess + train)
- `predict_with_nnUNet.py`: inference entry (best config / custom)
- `utils/train_utils.py`: training command building & arg parsing
- `utils/predict_utils.py`: instruction parsing & command building (ensemble/cascade support)
- `utils/run_utils.py`: output monitor progress bar
- `configs/train_config.py`: training config
- `configs/predict_data_config.py`: inference config

