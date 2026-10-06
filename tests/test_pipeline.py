"""Fast regression tests; no TensorFlow session is required."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from constellating import build_parser, generation_config
from prop_calc_module_basis import main as prepare
from utils import (build_vocabulary, candidate_records, canonical_molecule, conv_to_smiles,
                   encode_records, iter_batches, load_datasource, load_metadata,
                   read_property_records, save_json, split_indices)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def properties(self, text):
        path = self.folder / 'properties.tsv'
        path.write_text(text, encoding='utf-8')
        return path

    def test_final_record_is_kept_with_or_without_newline(self):
        for ending in ('', '\n'):
            path = self.properties('CC 30 1 2\nCO 32 2 3' + ending)
            data = load_datasource(path, 20)
            self.assertEqual(len(data[0]), 2)
            self.assertEqual(data[4].dtype, np.float32)

    def test_blank_lines_are_skipped_but_bad_properties_fail(self):
        path = self.properties('\nCC 30 1 2\n \nCO 32 2 3\n')
        self.assertEqual(read_property_records(path)[2], [2, 4])
        for value in ('bad', 'NaN', 'inf'):
            with self.assertRaises(ValueError):
                read_property_records(self.properties(f'CC 30 {value} 2'))
        with self.assertRaises(ValueError):
            read_property_records(self.properties('CC 30 1'))

    def test_vocabulary_order_matches_original_frequency_order(self):
        self.assertEqual(build_vocabulary(['CN', 'CO']), ['C', 'N', 'O', 'E', 'X'])

    def test_encoding_and_decoding_preserve_sequence(self):
        chars = build_vocabulary(['CCO', 'CN'])
        x, y, lengths = encode_records(['CCO', 'CN'], chars, 8)
        self.assertEqual(x.shape, (2, 8))
        self.assertEqual(lengths.tolist(), [4, 3])
        self.assertEqual([conv_to_smiles(row, chars) for row in y], ['CCO', 'CN'])
        self.assertEqual(conv_to_smiles([chars.index('X')], chars), '')
        with self.assertRaises(ValueError):
            encode_records(['Cl'], chars, 8)

    def test_molecular_deduplication_and_empty_rejection(self):
        rows, stats = candidate_records(['CCO', 'OCC', '', 'not_smiles', 'CC'])
        self.assertEqual([row[0] for row in rows], ['CCO', 'CC'])
        self.assertEqual(stats, dict(attempted=5, valid=3, unique=2, invalid=2, duplicates=1))

    def test_reproducible_split_keeps_duplicate_molecules_together(self):
        smiles = ['CCO', 'OCC', 'CC', 'CO', 'CN', 'CCC', 'CCN', 'C']
        training, validation = split_indices(smiles, seed=7)
        self.assertEqual(sorted(np.r_[training, validation].tolist()), list(range(len(smiles))))
        train_keys = {canonical_molecule(smiles[i])[0] for i in training}
        validation_keys = {canonical_molecule(smiles[i])[0] for i in validation}
        self.assertFalse(train_keys & validation_keys)
        second = split_indices(smiles, seed=7)
        np.testing.assert_array_equal(training, second[0])
        np.testing.assert_array_equal(validation, second[1])

    def test_partial_final_batch_is_retained(self):
        self.assertEqual([batch.tolist() for batch in iter_batches(np.arange(5), 2)], [[0, 1], [2, 3], [4]])

    def test_iteration_argument_is_integer(self):
        args = build_parser().parse_args(['--save_file', 'model', '--target_prop', '30 1 2', '--num_iteration', '10'])
        self.assertEqual(list(range(args.num_iteration)), list(range(10)))

    def metadata(self):
        return dict(format_version=1, vocabulary=['C', 'O', 'E', 'X'],
                    property_names=['MW', 'LogP', 'TPSA'], property_transform='none',
                    embedding_layout='legacy_latent_by_vocab',
                    model=dict(latent_size=20, unit_size=8, n_rnn_layer=2, seq_length=20,
                               num_prop=3, mean=0.0, stddev=1.0, lr=0.001))

    def test_saved_vocabulary_is_used_without_reopening_data(self):
        checkpoint = self.folder / 'model.ckpt-1'
        save_json(str(checkpoint) + '.metadata.json', self.metadata())
        args = build_parser().parse_args(['--save_file', str(checkpoint), '--target_prop', '30 1 2'])
        config, chars, data = generation_config(args, checkpoint)
        self.assertEqual(chars, self.metadata()['vocabulary'])
        self.assertEqual(config.n_rnn_layer, 2)
        args.latent_size = 21
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            generation_config(args, checkpoint)

    def test_legacy_vocabulary_recovery_requires_opt_in_and_preserves_old_slicing(self):
        checkpoint = self.folder / 'old.ckpt'
        args = build_parser().parse_args(['--save_file', str(checkpoint), '--target_prop', '30 1 2'])
        with self.assertRaises(ValueError):
            generation_config(args, checkpoint)
        args.legacy_checkpoint = True
        args.prop_file = str(self.properties('CC 30 1 2\nCN 31 1 2'))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, chars, _ = generation_config(args, checkpoint)
        self.assertEqual(chars, ['C', 'E', 'X'])

    def test_metadata_validation_rejects_wrong_property_order(self):
        path = self.folder / 'metadata.json'
        data = self.metadata()
        data['property_names'] = ['LogP', 'MW', 'TPSA']
        save_json(path, data)
        with self.assertRaises(ValueError):
            load_metadata(path)

    def test_preparation_cli_keeps_last_line_and_honors_multiple_workers(self):
        source = self.folder / 'smiles.txt'
        source.write_text('CC\n\nCO', encoding='utf-8')
        outputs = []
        for workers in (1, 2):
            target = self.folder / f'out{workers}.tsv'
            process = subprocess.run([sys.executable, str(ROOT / 'prop_calc_module_basis.py'),
                '--input_filename', str(source), '--output_filename', str(target),
                '--ncpus', str(workers)], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            outputs.append(target.read_text())
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(len(outputs[0].splitlines()), 2)

    def test_bad_cli_input_does_not_create_output(self):
        source = self.folder / 'smiles.txt'
        source.write_text('\n', encoding='utf-8')
        out = self.folder / 'out.tsv'
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            prepare(['--input_filename', str(source), '--output_filename', str(out)])
        self.assertFalse(out.exists())

    def test_imports_do_not_parse_command_line_or_write_files(self):
        process = subprocess.run([sys.executable, '-c',
            'import core_train, constellating, prop_calc_module_basis, test; print("ok")', '--unexpected'],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout.strip(), 'ok')


if __name__ == '__main__':
    unittest.main()
