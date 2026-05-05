import logging
from pathlib import Path


def setup_logging(output_dir):
    """
    Simple logging setup - logs to file only.
    
    Args:
        output_dir: Directory where training.log will be saved
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    handler = logging.FileHandler(output_dir / "training.log")
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
