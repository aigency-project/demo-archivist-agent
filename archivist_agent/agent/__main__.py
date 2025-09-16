"""Main entry point for the reception agent."""

import os
import sys

from aigency.aigency import open_aigency
from aigency.utils.logger import get_logger

logger = get_logger()

def main():

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_config.yaml")
    open_aigency(config_path=config_path)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user. Exiting...")

