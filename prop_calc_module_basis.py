"""Calculate raw MW, logP, and TPSA descriptors from one SMILES per line."""
import argparse
from multiprocessing import Pool
from pathlib import Path

from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from utils import canonical_molecule, positive_int


def cal_prop(smiles):
    parsed = canonical_molecule(smiles)
    if parsed is None:
        return None
    canonical, mol = parsed
    return canonical, ExactMolWt(mol), MolLogP(mol), CalcTPSA(mol)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_filename', default='smiles.txt')
    parser.add_argument('--output_filename', default='smiles_prop.txt')
    parser.add_argument('--ncpus', type=positive_int, default=1)
    args = parser.parse_args(argv)
    if Path(args.input_filename).resolve() == Path(args.output_filename).resolve():
        parser.error('Input and output paths must differ.')
    try:
        smiles = [s.strip() for s in Path(args.input_filename).read_text(encoding='utf-8').splitlines() if s.strip()]
        if not smiles:
            raise ValueError('Input contains no SMILES.')
        if args.ncpus == 1:
            results = list(map(cal_prop, smiles))
        else:
            with Pool(args.ncpus) as pool:
                results = pool.map(cal_prop, smiles)
        valid = [row for row in results if row is not None]
        if not valid:
            raise ValueError('No valid, nonempty molecules were found.')
        target = Path(args.output_filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('w', encoding='utf-8') as out:
            for row in valid:
                out.write('\t'.join(map(str, row)) + '\n')
        print(f'Input: {len(smiles)}; written: {len(valid)}; invalid: {len(smiles) - len(valid)}')
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
