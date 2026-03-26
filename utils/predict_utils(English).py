import logging
from pathlib import Path
import sys

# Logs directory: Located at 'log/' within the repository root
# (shared across training and inference modules)
LOGS_DIR = Path(__file__).resolve().parent.parent / "log"

def setup_logging(log_name=None, log_dir=None, level=logging.INFO, overwrite=False):
    """
    Unified logging configuration. Defaults to LOGS_DIR using the caller script's name.
    
    Args:
        log_name (str, optional): Name of the log file (e.g., 'inference.log'). 
            If None, the stem of the calling script is used.
        log_dir (Path or str, optional): Directory for log storage. 
            Defaults to the global LOGS_DIR.
        level (int): Logging threshold (e.g., logging.INFO or logging.DEBUG).
        overwrite (bool): If True, initializes a fresh log file ('w'). 
            If False, appends to the existing file ('a').
        
    Returns:
        Path: The absolute path to the generated log file.
    """
    import __main__
    
    # Resolve log directory
    if log_dir is None:
        log_dir = LOGS_DIR
    else:
        log_dir = Path(log_dir)
        
    # Create directory if it doesn't exist (including parent folders)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Automatically determine log filename if not provided
    if log_name is None:
        # Fetch the name of the script that invoked this utility
        log_name = f"{Path(__main__.__file__).stem}.log"
    
    log_path = log_dir / log_name
    
    # Configure dual-stream handlers: File and System Console (Stdout)
    handlers = [
        logging.FileHandler(log_path, mode='w' if overwrite else 'a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
    
    # Set global logging parameters
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logging.info(f"Logging session initialized at: {log_path}")
    return log_path
