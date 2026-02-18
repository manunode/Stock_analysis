"""
CLI entry point for stocklens.
Usage:
    python -m stocklens
    python -m stocklens --output custom_output.xlsx
    python -m stocklens --log-level DEBUG
"""

import argparse
import logging
import os
import sys

from .pipeline import run_pipeline


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with timestamp and severity level."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description='Stock analysis')
    parser.add_argument('--output', default=os.path.join(base_dir, 'stock_analysis.xlsx'))
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        run_pipeline(base_dir, args.output)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        logging.error(str(e))
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(2)


if __name__ == '__main__':
    main()
