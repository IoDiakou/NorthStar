"""Legacy data preparation: extract alternating lines (this is not a test suite)."""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_filename', default='smiles_original.txt')
    parser.add_argument('--output_filename', default='smiles.txt')
    args = parser.parse_args(argv)
    if Path(args.input_filename).resolve() == Path(args.output_filename).resolve():
        parser.error('Input and output paths must differ.')
    try:
        records = Path(args.input_filename).read_text(encoding='utf-8').splitlines()[::2]
        Path(args.output_filename).write_text('\n'.join(records) + ('\n' if records else ''), encoding='utf-8')
    except OSError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
