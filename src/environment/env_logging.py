import logging
import os


def initialize_logger(log_file="app.log", log_level=logging.DEBUG):
    """
    Initializes and configures a logger that tracks all important information.

    Parameters:
    log_file (str): Path to the log file.
    log_level (int): Logging level (e.g., logging.DEBUG, logging.INFO).

    Returns:
    logging.Logger: Configured logger instance.
    """
    # Create a logger
    logger = logging.getLogger("AppLogger")
    logger.setLevel(log_level)

    # Prevent duplicate log entries
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define log format
    log_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_format)

    # File handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


