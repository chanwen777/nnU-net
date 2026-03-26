"""
nnU-Net Training Configurations
train_args: Mapping of arguments passed to 'nnUNetv2_train'. 
Used primarily when the user provides only the dataset name.
Supported keys: tr, p, pretrained_weights, npz, c, val, val_best, disable_checkpointing, device

'tr' (Trainer) and 'p' (Plans) can be either a string or a list. 
If provided as a list, the pipeline will execute the Cartesian product of (num_trainers * num_plans).
Example: "tr": ["nnUNetTrainer", "nnUNetTrainerNoMirroring"], "p": ["nnUNetPlans", "nnUNetResEncUNetPlans_49G"]
"""

# Training Configuration Mapping: Dataset Name -> Config Dictionary
TRAIN_DATASET_CONFIGS = {
    "Dataset001_ExampleA": {
        "verify_dataset_integrity": True,
        "configurations": ["3d_fullres"],
        "folds": 0,
        "train_args": {
            "npz": True,
            "device": "cuda",
        },
    },
    "Dataset002_ExampleB": {
        "verify_dataset_integrity": True,
        "configurations": ["3d_fullres"],
        "folds": 0,
        "train_args": {
            "npz": True,
            "device": "cuda",
        },
    },
    "Dataset003_ExampleC": {
        "verify_dataset_integrity": True,
        "configurations": ["2d"],
        "folds": 0,
        "train_args": {
            "npz": True, 
            "device": "cuda"
        },
    },
    "Dataset004_ExampleD": {
        "verify_dataset_integrity": True,
        "configurations": ["3d_fullres", "3d_lowres", "3d_cascade_fullres"],
        "folds": [0, 1, 2, 3, 4],
        "train_args": {
            # Example for hyperparameter exploration:
            "tr": ["nnUNetTrainer", "nnUNetTrainerNoMirroring"],
            "p": ["nnUNetPlans", "nnUNetResEncUNetPlans_49G"],
            "npz": True, 
            "device": "cuda"
        },
    },
}

# Default dataset for training (can be overridden via CLI)
DEFAULT_DATASET_NAME = "DatasetXXX_YourDatasetName"

# Default fold list for 5-fold cross-validation (can be overridden via CLI)
DEFAULT_FOLDS = [0, 1, 2, 3, 4]
