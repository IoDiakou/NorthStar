"""Validated data preparation, vocabulary persistence, and candidate handling."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcTPSA

START_TOKEN = 'X'
END_TOKEN = 'E'
PROPERTY_NAMES = ['MW', 'LogP', 'TPSA']


def positive_int(value):
    value = int(value)
    if value < 1:
        raise ValueError('Must be a positive integer.')
    return value


def positive_float(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Must be a finite, positive number.')
    return value


def read_property_records(filename, seq_length=120, num_prop=3, legacy=False):
    """Read headerless SMILES + numeric properties, retaining source line numbers.

    legacy=True reproduces the original final-line slicing for vocabulary
    recovery ONLY; new training must leave this disabled.
    """
    if seq_length < 3:
        raise ValueError('seq_length must be at least 3.')
    raw = Path(filename).read_text(encoding='utf-8')
    lines = raw.split('\n')[:-1] if legacy else raw.splitlines()
    smiles, properties, source_lines = [], [], []
    too_long = 0
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != num_prop + 1:
            raise ValueError(f'{filename}:{line_number}: expected SMILES and {num_prop} properties.')
        try:
            values = np.asarray(fields[1:], dtype=np.float32)
        except ValueError as exc:
            raise ValueError(f'{filename}:{line_number}: properties must be numeric; omit headers.') from exc
        if not np.all(np.isfinite(values)):
            raise ValueError(f'{filename}:{line_number}: properties must be finite.')
        # Retain the original length threshold to preserve legacy vocabularies.
        if len(fields[0]) >= seq_length - 2:
            too_long += 1
            continue
        smiles.append(fields[0])
        properties.append(values)
        source_lines.append(line_number)
    if too_long:
        warnings.warn(f'Skipped {too_long} molecules exceeding the sequence-length limit.', stacklevel=2)
    if not smiles:
        raise ValueError(f'{filename}: no usable molecular records.')
    return smiles, np.asarray(properties, dtype=np.float32), source_lines


def build_vocabulary(smiles):
    """Preserve the original frequency ordering, including first-seen tie order."""
    counts = Counter(''.join(smiles))
    chars = [char for char, _ in sorted(counts.items(), key=lambda item: -item[1])]
    if START_TOKEN in chars or END_TOKEN in chars:
        raise ValueError('Input contains reserved X/E sequence characters; tokenization must be extended.')
    return chars + [END_TOKEN, START_TOKEN]


def validate_vocabulary(chars):
    if (not isinstance(chars, list) or not chars
            or any(not isinstance(c, str) or len(c) != 1 for c in chars)
            or len(chars) != len(set(chars))
            or not {START_TOKEN, END_TOKEN}.issubset(chars)):
        raise ValueError('Invalid saved character vocabulary.')
    return {char: index for index, char in enumerate(chars)}


def encode_records(smiles, chars, seq_length):
    vocab = validate_vocabulary(list(chars))
    unknown = set(''.join(smiles)) - set(vocab)
    if unknown:
        raise ValueError(f'Characters absent from saved vocabulary: {sorted(unknown)}')
    lengths = np.asarray([len(s) + 1 for s in smiles], dtype=np.int32)
    if np.any(lengths > seq_length):
        raise ValueError('Input sequence exceeds seq_length.')
    x = np.asarray([[vocab[c] for c in (START_TOKEN + s).ljust(seq_length, END_TOKEN)]
                    for s in smiles], dtype=np.int32)
    y = np.asarray([[vocab[c] for c in s.ljust(seq_length, END_TOKEN)]
                    for s in smiles], dtype=np.int32)
    return x, y, lengths


def load_datasource(filename, seq_length, chars=None, num_prop=3):
    smiles, properties, _ = read_property_records(filename, seq_length, num_prop)
    chars = build_vocabulary(smiles) if chars is None else list(chars)
    x, y, lengths = encode_records(smiles, chars, seq_length)
    return x, y, tuple(chars), validate_vocabulary(chars), properties, lengths


# Backward-compatible names for callers of the original entry points.
load_data = load_datasource


def conv_to_smiles(vector, chars):
    tokens = np.asarray(vector)
    if tokens.ndim != 1:
        raise ValueError('Expected one 1D token sequence.')
    result = []
    for index in tokens:
        index = int(index)
        if index < 0 or index >= len(chars):
            raise ValueError(f'Token index {index} is outside the saved vocabulary.')
        char = chars[index]
        if char == END_TOKEN:
            break
        if char == START_TOKEN:
            # A generated start marker is invalid; never silently reinterpret it.
            return ''
        result.append(char)
    return ''.join(result).strip()


convert_to_smiles = conv_to_smiles


def canonical_molecule(smiles):
    if not smiles or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True), mol


def candidate_records(smiles):
    """Canonicalize before deduplication; retain first-seen order and statistics."""
    records = []
    seen = set()
    stats = {'attempted': 0, 'valid': 0, 'unique': 0, 'invalid': 0, 'duplicates': 0}
    for value in smiles:
        stats['attempted'] += 1
        parsed = canonical_molecule(value)
        if parsed is None:
            stats['invalid'] += 1
            continue
        canonical, mol = parsed
        stats['valid'] += 1
        if canonical in seen:
            stats['duplicates'] += 1
            continue
        seen.add(canonical)
        records.append((canonical, ExactMolWt(mol), MolLogP(mol), CalcTPSA(mol)))
    stats['unique'] = len(records)
    return records, stats


def split_indices(smiles, validation_fraction=0.25, seed=42):
    """Seeded split by canonical structure, keeping duplicate molecules together."""
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between 0 and 1.')
    groups = {}
    for index, value in enumerate(smiles):
        parsed = canonical_molecule(value)
        if parsed is None:
            raise ValueError(f'Invalid training SMILES at record {index + 1}: {value!r}')
        groups.setdefault(parsed[0], []).append(index)
    keys = list(groups)
    if len(keys) < 2:
        raise ValueError('Need at least two distinct molecules for training and validation.')
    shuffled = np.random.default_rng(seed).permutation(len(keys))
    count = min(len(keys) - 1, max(1, int(round(len(keys) * validation_fraction))))
    validation_keys = {keys[i] for i in shuffled[:count]}
    validation = [i for k, values in groups.items() if k in validation_keys for i in values]
    training = [i for k, values in groups.items() if k not in validation_keys for i in values]
    return np.asarray(sorted(training), dtype=np.int64), np.asarray(sorted(validation), dtype=np.int64)


def iter_batches(indices, batch_size):
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def file_sha256(filename):
    digest = hashlib.sha256()
    with open(filename, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(filename, value):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_metadata(filename):
    data = json.loads(Path(filename).read_text(encoding='utf-8'))
    if data.get('format_version') != 1:
        raise ValueError('Unsupported metadata format version.')
    validate_vocabulary(data.get('vocabulary'))
    if data.get('property_names') != PROPERTY_NAMES:
        raise ValueError('Expected property order MW, LogP, TPSA.')
    if data.get('property_transform') != 'none':
        raise ValueError('Unsupported property transform; do not mix scaled and raw properties.')
    if data.get('embedding_layout') != 'legacy_latent_by_vocab':
        raise ValueError('Unsupported embedding layout.')
    for key in ('latent_size', 'unit_size', 'n_rnn_layer', 'seq_length', 'num_prop'):
        if not isinstance(data.get('model', {}).get(key), int) or data['model'][key] <= 0:
            raise ValueError(f'Missing or invalid saved model setting: {key}')
    for key in ('mean', 'stddev', 'lr'):
        value = data['model'].get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f'Missing or invalid saved distribution/optimizer setting: {key}')
    if data['model']['stddev'] <= 0 or data['model']['lr'] <= 0:
        raise ValueError('Saved stddev and learning rate must be positive.')
    if data['model']['seq_length'] < 3:
        raise ValueError('Saved seq_length must be at least 3.')
    if data['model']['num_prop'] != len(PROPERTY_NAMES):
        raise ValueError('Saved model must have three property conditions.')
    return data
