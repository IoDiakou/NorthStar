"""Generate property-conditioned candidates using a checkpoint's saved vocabulary."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
from utils import (START_TOKEN, PROPERTY_NAMES, build_vocabulary, candidate_records,
                   conv_to_smiles, load_metadata, positive_float, positive_int,
                   read_property_records, save_json, validate_vocabulary)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save_file', required=True, help='Checkpoint prefix or training output directory.')
    parser.add_argument('--target_prop', required=True, help='Quoted raw values: "MW LogP TPSA".')
    parser.add_argument('--result_filename', default='result.txt')
    parser.add_argument('--metadata', help='Explicit metadata JSON; normally inferred from the checkpoint.')
    parser.add_argument('--legacy_checkpoint', action='store_true', help='Opt in to recovery without saved metadata.')
    parser.add_argument('--prop_file', help='Original training property file, only for legacy vocabulary recovery.')
    parser.add_argument('--batch_size', type=positive_int, default=128)
    parser.add_argument('--num_iteration', type=positive_int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_threads', type=positive_int, default=2)
    # New checkpoints supply these settings; explicit conflicting overrides fail.
    for name in ('latent_size', 'unit_size', 'n_rnn_layer', 'seq_length', 'num_prop'):
        parser.add_argument('--' + name, type=positive_int)
    parser.add_argument('--mean', type=float, help='Generation prior mean; defaults to saved value.')
    parser.add_argument('--stddev', type=positive_float, help='Generation prior stddev; defaults to saved value.')
    parser.add_argument('--lr', type=positive_float, help='Legacy graph setting; no training is performed.')
    return parser


def generation_config(args, checkpoint):
    inferred = Path(str(checkpoint) + '.metadata.json')
    metadata_path = Path(args.metadata) if args.metadata else inferred
    defaults = dict(latent_size=200, unit_size=512, n_rnn_layer=3, seq_length=120,
                    num_prop=3, mean=0.0, stddev=1.0, lr=0.0001)
    if metadata_path.is_file():
        metadata = load_metadata(metadata_path)
        settings = dict(metadata['model'])
        chars = metadata['vocabulary']
        for name in ('latent_size', 'unit_size', 'n_rnn_layer', 'seq_length', 'num_prop'):
            override = getattr(args, name)
            if override is not None and override != settings[name]:
                raise ValueError(f'--{name} conflicts with checkpoint metadata.')
    else:
        if args.metadata:
            raise ValueError(f'Metadata file does not exist: {metadata_path}')
        if not args.legacy_checkpoint or not args.prop_file:
            raise ValueError('Checkpoint metadata is missing. For older checkpoints, explicitly provide '
                             '--legacy_checkpoint and --prop_file with the exact original training file.')
        settings = defaults.copy()
        for name in defaults:
            if getattr(args, name) is not None:
                settings[name] = getattr(args, name)
        smiles, _, _ = read_property_records(args.prop_file, settings['seq_length'],
                                             settings['num_prop'], legacy=True)
        chars = build_vocabulary(smiles)
        warnings.warn('Legacy recovery cannot verify vocabulary identity. Use the exact original file '
                      'and original model dimensions. The historical final-line slicing is reproduced.', stacklevel=2)
        metadata = None
    for name in ('mean', 'stddev', 'lr'):
        override = getattr(args, name)
        if override is not None:
            settings[name] = override
    if settings['num_prop'] != 3:
        raise ValueError('This exporter supports three properties: MW, LogP, TPSA.')
    settings.update(batch_size=args.batch_size, seed=args.seed, num_threads=args.num_threads, max_to_keep=5)
    return SimpleNamespace(**settings), chars, metadata


def run_generation(args):
    if args.seed < 0:
        raise ValueError('seed must be nonnegative.')
    try:
        target = np.asarray([float(p) for p in args.target_prop.split()], dtype=np.float32)
    except ValueError as exc:
        raise ValueError('target_prop must contain three numbers: MW LogP TPSA.') from exc
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError('target_prop must contain exactly three finite numbers: MW LogP TPSA.')
    from core_architecture import CVAE
    import tensorflow as tf
    checkpoint = str(args.save_file)
    if Path(checkpoint).is_dir():
        checkpoint = tf.train.latest_checkpoint(checkpoint)
        if checkpoint is None:
            raise ValueError('No checkpoint found in the supplied directory.')
    if not Path(checkpoint + '.index').is_file():
        raise ValueError('Supply a checkpoint prefix (without .index or .data suffixes), or its directory.')
    config, chars, metadata = generation_config(args, checkpoint)
    vocab = validate_vocabulary(chars)
    output = Path(args.result_filename)
    statistics_path = Path(str(output) + '.stats.json')
    if output.exists() or statistics_path.exists():
        raise ValueError('Output already exists; choose a new result_filename to retain previous results.')
    target_batch = np.tile(target, (args.batch_size, 1))
    starts = np.full((args.batch_size, 1), vocab[START_TOKEN], dtype=np.int32)
    rng = np.random.default_rng(args.seed)
    generated_smiles = []
    with CVAE(len(chars), config) as model:
        model.restore(checkpoint)
        for _ in range(args.num_iteration):
            latent = rng.normal(config.mean, config.stddev, (args.batch_size, config.latent_size)).astype(np.float32)
            generated = model.sample(latent, target_batch, starts, config.seq_length)
            generated_smiles.extend(conv_to_smiles(row, chars) for row in generated)
    records, stats = candidate_records(generated_smiles)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as stream:
        stream.write('smiles\tMW\tLogP\tTPSA\n')
        for smiles, mw, logp, tpsa in records:
            stream.write(f'{smiles}\t{mw:.6f}\t{logp:.6f}\t{tpsa:.6f}\n')
    stats.update(checkpoint=checkpoint, seed=args.seed, target_properties=dict(zip(PROPERTY_NAMES, target.tolist())),
                 prior_mean=config.mean, prior_stddev=config.stddev,
                 vocabulary_source='saved metadata' if metadata is not None else 'legacy file recovery',
                 property_filter_applied=False, decoding='greedy argmax',
                 validity_fraction=stats['valid'] / stats['attempted'],
                 uniqueness_among_valid=stats['unique'] / stats['valid'] if stats['valid'] else None)
    save_json(statistics_path, stats)
    print(json.dumps(stats, indent=2))
    print(f'Wrote {output}')
    return stats


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_generation(args)
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
