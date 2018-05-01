import os
import logging
import logging.handlers


def _get_logger(logdir, logname, loglevel=logging.INFO):
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    formatter = logging.Formatter(fmt)
    logger = logging.getLogger("")

    if not logger.handlers:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(logdir, logname),
            maxBytes=10 * 1024 * 1024,
            backupCount=10)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    logger.setLevel(loglevel)

    return logger
