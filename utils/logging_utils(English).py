import logging
from pathlib import Path
import sys

# Log directory: 'log' folder at the repository root (shared by training and inference scripts)
LOGS_DIR = Path(__file__).resolve().parent.parent / "log"

def setup_logging(log_name=None, log_dir=None, level=logging.INFO, overwrite=False):
    """
    Unified logging configuration. Defaults to LOGS_DIR with the caller script's name.
    
    Args:
        log_name (str): Log filename (e.g., 'train.log'). If None, uses the calling script's stem.
        log_dir (Path/str): Directory to store logs. Defaults to LOGS_DIR.
        level (int): Logging level (e.g., logging.INFO).
        overwrite (bool): If True, overwrites existing log file ('w'). If False, appends ('a').
        
    Returns:
        Path: The absolute path to the initialized log file.
    """
    import __main__
    
    # Initialize log directory
    if log_dir is None:
        log_dir = LOGS_DIR
    else:
        log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Determine log filename
    if log_name is None:
        # Use the name of the script that executed this function
        log_name = f"{Path(__main__.__file__).stem}.log"
    
    log_path = log_dir / log_name
    
    # Define handlers: File and Console (Stdout)
    handlers = [
        logging.FileHandler(log_path, mode='w' if overwrite else 'a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
    
    # Configure logging format and handlers
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logging.info(f"Logging initialized at: {log_path}")
    return log_path
