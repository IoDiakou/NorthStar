"""Opt-in TensorFlow integration tests: NORTHSTAR_RUN_MODEL_TESTS=1.

Run on a host capable of creating TensorFlow sessions. These tests train only
small fixture models; they do not establish research performance.
"""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ENABLED = os.environ.get('NORTHSTAR_RUN_MODEL_TESTS') == '1'


@unittest.skipUnless(ENABLED, 'Set NORTHSTAR_RUN_MODEL_TESTS=1 on a TensorFlow-capable host.')
class ModelTests(unittest.TestCase):
    @staticmethod
    def settings(layers=2):
        return SimpleNamespace(batch_size=2, latent_size=20, unit_size=8, n_rnn_layer=layers,
            seq_length=16, num_prop=3, mean=0.0, stddev=1.0, lr=0.001, seed=42,
            num_threads=1, max_to_keep=2)

    def test_training_save_restore_dynamic_batches_and_layer_counts(self):
        from core_architecture import CVAE
        from utils import build_vocabulary, encode_records
        chars = build_vocabulary(['CCO', 'CN', 'CC'])
        x, y, lengths = encode_records(['CCO', 'CN', 'CC'], chars, 16)
        properties = np.array([[46, 0, 20], [31, 0, 26], [30, 1, 0]], dtype=np.float32)
        for layers in (1, 3):
            args = self.settings(layers)
            with tempfile.TemporaryDirectory() as tmp:
                with CVAE(len(chars), args) as original:
                    loss = original.train(x, y, lengths, properties)
                    self.assertTrue(np.isfinite(loss))
                    expected = original.loss_components(x, y, lengths, properties)
                    repeated = original.loss_components(x, y, lengths, properties)
                    np.testing.assert_allclose(expected, repeated, rtol=1e-6)
                    checkpoint = original.save(Path(tmp) / 'model', 1)
                    z = np.zeros((1, args.latent_size), dtype=np.float32)
                    start = np.array([[chars.index('X')]], dtype=np.int32)
                    generated = original.sample(z, properties[:1], start, args.seq_length)
                    self.assertEqual(generated.shape, (1, args.seq_length))
                    saved_variables = dict(original._checkpoint_variables)
                with CVAE(len(chars), args) as restored:
                    self.assertEqual(saved_variables, restored._checkpoint_variables)
                    restored.restore(checkpoint)
                    actual = restored.loss_components(x, y, lengths, properties)
                    np.testing.assert_allclose(actual, expected, rtol=1e-6)
                    np.testing.assert_array_equal(restored.sample(z, properties[:1], start, args.seq_length), generated)

    def test_checkpoint_created_by_original_architecture(self):
        # Fixture is the public original at commit db4c7bac47b42171043ecb379272da90f3c5cd14.
        import importlib.util
        from unittest import mock
        import tensorflow as tf
        from core_architecture import CVAE
        from utils import build_vocabulary, encode_records
        fixture = ROOT / 'tests' / 'fixtures' / 'legacy_core_architecture.py'
        spec = importlib.util.spec_from_file_location('legacy_core_fixture', fixture)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        args = self.settings(3)
        chars = build_vocabulary(['CCO', 'CN'])
        x, y, lengths = encode_records(['CCO', 'CN'], chars, args.seq_length)
        properties = np.array([[46, 0, 20], [31, 0, 26]], dtype=np.float32)
        z = np.zeros((2, args.latent_size), dtype=np.float32)
        starts = np.full((2, 1), chars.index('X'), dtype=np.int32)
        sessions = []
        real_session = tf.compat.v1.Session
        def tracked_session(*positional, **keywords):
            keywords.setdefault('config', tf.compat.v1.ConfigProto(
                intra_op_parallelism_threads=1, inter_op_parallelism_threads=1))
            session = real_session(*positional, **keywords)
            sessions.append(session)
            return session
        with tempfile.TemporaryDirectory() as tmp:
            try:
                with tf.Graph().as_default(), mock.patch.object(tf.compat.v1, 'Session', side_effect=tracked_session):
                    old = legacy.CVAE(len(chars), args)
                    self.assertTrue(np.isfinite(old.train(x, y, lengths, properties)))
                    expected_loss = old.sess.run([old.reconstr_loss, old.latent_loss], feed_dict={
                        old.X: x, old.Y: y, old.L: lengths, old.C: properties, old.eps['eps']: z})
                    expected_tokens = old.sample(z, properties, starts, args.seq_length)
                    prefix = str(Path(tmp) / 'legacy.ckpt')
                    old.save(prefix, 1)
            finally:
                for session in sessions:
                    session.close()
            with CVAE(len(chars), args) as new:
                new.restore(prefix + '-1')
                np.testing.assert_allclose(new.loss_components(x, y, lengths, properties),
                                           expected_loss, rtol=1e-5, atol=1e-6)
                np.testing.assert_array_equal(new.sample(z, properties, starts, args.seq_length), expected_tokens)

    def test_cli_end_to_end(self):
        from core_train import main as train
        from constellating import main as generate
        from prop_calc_module_basis import main as prepare
        import json
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            props = folder / 'props.tsv'
            prepare(['--input_filename', str(ROOT / 'examples' / 'smiles.txt'), '--output_filename', str(props)])
            run = folder / 'run'
            train(['--prop_file', str(props), '--save_dir', str(run), '--num_epochs', '1',
                   '--batch_size', '7', '--unit_size', '8', '--latent_size', '24', '--n_rnn_layer', '2',
                   '--seq_length', '32', '--num_threads', '1'])
            props.unlink()  # Saved vocabulary must make generation independent of training input.
            output = folder / 'candidates.tsv'
            generate(['--save_file', str(run), '--target_prop', '46 0 20',
                      '--batch_size', '1', '--num_iteration', '2', '--result_filename', str(output)])
            self.assertTrue(output.read_text().startswith('smiles\tMW\tLogP\tTPSA\n'))
            stats = json.loads(Path(str(output) + '.stats.json').read_text())
            self.assertEqual(stats['attempted'], 2)


if __name__ == '__main__':
    unittest.main()
