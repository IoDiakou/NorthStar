"""Run regression tests; add --model to include real TensorFlow integration tests."""
import argparse
import os
from pathlib import Path
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='store_true', help='Train tiny models and verify checkpoint restoration.')
    args = parser.parse_args()
    if args.model:
        os.environ['NORTHSTAR_RUN_MODEL_TESTS'] = '1'
    root = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(str(root / 'tests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
