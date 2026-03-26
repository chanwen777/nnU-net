from pathlib import Path

# Prediction and Inference Configurations
PREDICT_CONFIGS = {
    "Dataset001_ExampleA": {
        "train_data": {
            "input_path": Path("/path/to/your/input/data"),
            "output_path": Path("/path/to/your/output/data"),
            "model_type": "3d_fullres",  # Valid nnU-Net configuration type
        },
        "valid_data": {
            "input_path": Path("/path/to/your/valid/data"),
            "output_path": Path("/path/to/your/output/data"),
            "model_type": "3d_fullres",  # Valid nnU-Net configuration type
        }
    },
    "Dataset002_ExampleB": {
        "train_data": {
            "input_path": Path("/path/to/your/train/data"),
            "output_path": Path("/path/to/your/output/data"),
            "model_type": "3d_fullres",  # Valid nnU-Net configuration type
        },
        "valid_data": {
            "input_path": Path("/path/to/your/valid/data"),
            "output_path": Path("/path/to/your/output/data"),
            "model_type": "3d_fullres",  # Valid nnU-Net configuration type
        }
    },
    "Dataset003_ExampleC": {
        "train_data": {
            "input_path": Path("/path/to/your/train/data"),
            "output_path": Path("/path/to/your/output/data"),
            "pred_stage_path": Path("/path/to/your/pred/stage/data"),
            "input_csv_path": "/",
            "model_type": "3d_cascade_fullres",
            "fold": 0,
            "force_all": False,
            "file_type": "nrrd",  # Set to None for auto-detection, or specify "bmp"/"nrrd" to force format
        }
    },
}
