import logging
import sys

import colorama
from colorama import Fore, Style


class CustomFormatter(logging.Formatter):
    PREFIXES = {
        "DEBUG": Fore.RESET + "#",
        "INFO": Fore.GREEN + "+",
        "WARNING": Fore.LIGHTYELLOW_EX + "-",
        "CRITICAL": Fore.LIGHTRED_EX + "!",
        "ERROR": Fore.LIGHTRED_EX + "E",
    }

    def format(self, record):
        prefix = self.PREFIXES.get(record.levelname, record.levelname)
        formatter = logging.Formatter(
            f"{Style.RESET_ALL}"
            f"{Fore.LIGHTBLACK_EX}[{Fore.RESET}%(asctime)s{Fore.LIGHTBLACK_EX}]"
            f"{Fore.LIGHTBLACK_EX}[{Fore.RESET}%(filename)s{Fore.LIGHTBLACK_EX}]"
            f"{Fore.LIGHTBLACK_EX}[{prefix}{Fore.LIGHTBLACK_EX}]"
            f"{Style.RESET_ALL} %(message)s{Style.RESET_ALL}",
            "%d/%m/%Y %H:%M:%S"
        )
        return formatter.format(record)


def init_logger():
    colorama.init()
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    log.addHandler(ch)


def get_logger():
    return logging.getLogger(__name__)
