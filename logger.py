
WHITE = 37
GREEN = 92
BLUE = 96
YELLOW = 93


def log(level: int, message: str, color: int = WHITE):
    log_levels = {0: "TRACE", 1: "DEBUG", 2: "INFO",
        3: "WARN", 4: "ERROR", 5: "FATAL"}
    print("[{}]:\33[{}m".format(log_levels[level], color), message, "\33[{}m".format(WHITE))