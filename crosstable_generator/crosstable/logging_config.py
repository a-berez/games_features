import logging


def setup_logging(verbose: int = 0) -> None:
    if verbose == 0:
        level = logging.INFO

        class InfoFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                if record.levelno >= logging.WARNING:
                    return f"{record.levelname}: {record.getMessage()}"
                return record.getMessage()

        formatter: logging.Formatter = InfoFormatter()
    elif verbose == 1:
        level = logging.INFO
        formatter = logging.Formatter("%(levelname)s: %(message)s")
    else:
        level = logging.DEBUG
        formatter = logging.Formatter("%(levelname)s: %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]
    logging.getLogger("urllib3").setLevel(logging.WARNING if verbose < 3 else logging.DEBUG)
