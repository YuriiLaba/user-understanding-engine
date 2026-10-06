import logging

def setup_logging(log_filename, logger_name):
    """Setup error logging to file"""
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    logger.handlers = []
    
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger