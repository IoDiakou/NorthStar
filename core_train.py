"""Train the CVAE and persist vocabulary, model settings, and data split."""
import argparse
import csv
import importlib.metadata
from pathlib import Path
import time

import numpy as np
from utils import (PROPERTY_NAMES, build_vocabulary, encode_records, file_sha256,
                   iter_batches, positive_float, positive_int, read_property_records,
                   save_json, split_indices)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prop_file', required=True)
    parser.add_argument('--save_dir', default='save')
    parser.add_argument('--batch_size', type=positive_int, default=128)
    parser.add_argument('--latent_size', type=positive_int, default=200)
    parser.add_argument('--unit_size', type=positive_int, default=512)
    parser.add_argument('--n_rnn_layer', type=positive_int, default=3)
    parser.add_argument('--seq_length', type=positive_int, default=120)
    parser.add_argument('--num_prop', type=int, choices=[3], default=3)
    parser.add_argument('--mean', type=float, default=0.0)
    parser.add_argument('--stddev', type=positive_float, default=1.0)
    parser.add_argument('--num_epochs', type=positive_int, default=50)
    parser.add_argument('--lr', type=positive_float, default=0.0005)
    parser.add_argument('--validation_fraction', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_to_keep', type=positive_int, default=5)
    parser.add_argument('--num_threads', type=positive_int, default=2)
    return parser


def run_training(args):
    if args.seed < 0 or not np.isfinite(args.mean):
        raise ValueError('Seed must be nonnegative; mean must be finite.')
    destination = Path(args.save_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError('save_dir must be empty or new, to avoid mixing checkpoints and vocabularies.')
    smiles, properties, source_lines = read_property_records(args.prop_file, args.seq_length, args.num_prop)
    chars = build_vocabulary(smiles)
    x, y, lengths = encode_records(smiles, chars, args.seq_length)
    training, validation = split_indices(smiles, args.validation_fraction, args.seed)
    from core_architecture import CVAE
    model_keys = ['latent_size', 'unit_size', 'n_rnn_layer', 'seq_length', 'num_prop', 'mean', 'stddev', 'lr']
    packages = {}
    for package in ('tensorflow', 'tensorflow-cpu', 'tensorflow-addons', 'numpy', 'rdkit'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata = {
        'format_version': 1, 'vocabulary': chars, 'property_names': PROPERTY_NAMES,
        'property_transform': 'none', 'embedding_layout': 'legacy_latent_by_vocab',
        'model': {key: getattr(args, key) for key in model_keys},
        'seed': args.seed, 'training_batch_size': args.batch_size,
        'data_file': str(Path(args.prop_file).resolve()), 'data_sha256': file_sha256(args.prop_file),
        'retained_source_lines': source_lines,
        'training_indices': training.tolist(), 'validation_indices': validation.tolist(),
        'split_method': 'seeded canonical-molecule groups; indices address retained records',
        'validation_method': 'posterior-mean reconstruction plus KL; not a sampled ELBO',
        'packages': packages,
    }
    rng = np.random.default_rng(args.seed)
    with CVAE(len(chars), args) as model:
        destination.mkdir(parents=True, exist_ok=True)
        save_json(destination / 'metadata.json', metadata)
        print(f'Training: {len(training)}; validation: {len(validation)}; parameters: {model.num_parameters}')
        fields = ['epoch', 'mean_training_batch_loss', 'validation_reconstruction',
                  'validation_kl', 'validation_total', 'seconds']
        with (destination / 'history.csv').open('w', newline='', encoding='utf-8') as out:
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            for epoch in range(args.num_epochs):
                start = time.monotonic()
                losses = []
                for batch in iter_batches(rng.permutation(training), args.batch_size):
                    losses.append(model.train(x[batch], y[batch], lengths[batch], properties[batch]))
                rec_sum, kl_sum, token_count, record_count = 0.0, 0.0, 0, 0
                for batch in iter_batches(validation, args.batch_size):
                    rec, kl = model.loss_components(x[batch], y[batch], lengths[batch], properties[batch])
                    tokens = int(lengths[batch].sum())
                    rec_sum += rec * tokens
                    kl_sum += kl * len(batch)
                    token_count += tokens
                    record_count += len(batch)
                rec, kl = rec_sum / token_count, kl_sum / record_count
                train_loss = float(np.mean(losses))
                if not np.all(np.isfinite([train_loss, rec, kl])):
                    raise ValueError('Non-finite training or validation loss; checkpoint was not saved for this epoch.')
                row = dict(zip(fields, [epoch + 1, train_loss, rec, kl, rec + kl, time.monotonic() - start]))
                writer.writerow(row)
                out.flush()
                checkpoint = model.save(destination / 'model.ckpt', epoch + 1)
                save_json(checkpoint + '.metadata.json', metadata)
                print(f'Epoch {epoch + 1}: train={train_loss:.4f}, validation={rec + kl:.4f}; {checkpoint}')
    return metadata


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_training(args)
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
