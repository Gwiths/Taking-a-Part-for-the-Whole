import logging
import os

class _LoggerMixin(object):
    """Log only when running in the root process."""
    def __init__(self, logger):
        self.logger = logger

    def __getattr__(self, attr):
        if not attr.startswith('_') and attr not in dir(self):
            return getattr(self.logger, attr)
        return super().__getattr__(attr)


class _Logger(_LoggerMixin):
    """Wrapped Logger."""
    def __init__(self):

        super().__init__(logging.getLogger('segmentation'))
        self.logger.setLevel(logging.DEBUG)
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setLevel(logging.INFO)
        self.logger.addHandler(self.stream_handler)

        self.file_handler = None

        self.set_formatter(
            '%(message)s'
        )

    def set_path(self, path):
        if self.file_handler:
            self.logger.removeHandler(self.file_handler)

        self.file_handler = logging.FileHandler(path)
        self.file_handler.setLevel(logging.INFO)
        self.logger.addHandler(self.file_handler)
        self.file_handler.setFormatter(self.formatter)

    def set_formatter(self, s):
        self.formatter = logging.Formatter(s)

        self.stream_handler.setFormatter(self.formatter)
        if self.file_handler:
            self.file_handler.setFormatter(self.formatter)

    def set_stream_level(self, level):
        self.stream_handler.setLevel(level)

    def set_file_level(self, level):
        self.file_handler.setLevel(getattr(logging, level, logging.DEBUG))

    def set_level(self, level):
        self.logger.setLevel(level)

logger = _Logger()
def set_logger_path(args, path):
    """Set dirichlet logger path.

    Args:
        path: str, path to the logger file.
    """
    global logger
    logger.set_level(args.logging.level)
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        logger.info(f'[logger] PATH {dirname} NOT FOUND! AUTOMATICALLY CREATED FOR YOU!')
        os.makedirs(dirname)
    logger.set_path(path)