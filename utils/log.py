# coding: utf-8
import logging

def get_logger(name, path=None, level=logging.INFO):
    """Get the specified name logger, log to path if path specified. 
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s]%(message)s')

    if path:
        fh = logging.FileHandler(path, 'a', encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger