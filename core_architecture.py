"""NorthStar's graph-mode CVAE, retaining legacy variable names and shapes.

The legacy embedding layout is intentionally preserved for checkpoint reuse.
Changing it is a separate model migration requiring checkpoint conversion or
retraining. Batch dimensions and decoder state depth are now dynamic.
"""
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

tf.compat.v1.disable_v2_behavior()


class CVAE:
    def __init__(self, vocab_size, args):
        self.vocab_size = int(vocab_size)
        self.batch_size = int(args.batch_size)
        self.latent_size = int(args.latent_size)
        self.num_prop = int(args.num_prop)
        self.stddev = float(args.stddev)
        self.prior_mean = float(args.mean)
        self.unit_size = int(args.unit_size)
        self.n_rnn_layer = int(args.n_rnn_layer)
        if min(self.vocab_size, self.batch_size, self.latent_size, self.num_prop,
               self.unit_size, self.n_rnn_layer) < 1:
            raise ValueError('Model dimensions must be positive.')
        if self.vocab_size > self.latent_size:
            raise ValueError(
                f'Legacy embedding layout needs latent_size >= vocab_size '
                f'({self.latent_size} < {self.vocab_size}). For an existing checkpoint, '
                'use its original architecture and vocabulary; do not change dimensions.')
        if not np.isfinite(self.prior_mean) or not np.isfinite(self.stddev) or self.stddev <= 0:
            raise ValueError('Normal distribution parameters must be finite; stddev must be positive.')
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.compat.v1.set_random_seed(getattr(args, 'seed', 42))
            self.lr = tf.Variable(float(args.lr), trainable=False)
            self._create_network()
            self.num_parameters = sum(int(np.prod(v.shape.as_list()))
                                      for v in tf.compat.v1.trainable_variables())
            self._checkpoint_variables = {v.op.name: v.shape.as_list()
                                          for v in tf.compat.v1.global_variables()}
            self.saver = tf.compat.v1.train.Saver(max_to_keep=getattr(args, 'max_to_keep', 5))
            initializer = tf.compat.v1.global_variables_initializer()
        threads = getattr(args, 'num_threads', 2)
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=threads,
                                          inter_op_parallelism_threads=threads)
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(graph=self.graph, config=config)
        self.sess.run(initializer)

    def _create_network(self):
        self.X = tf.compat.v1.placeholder(tf.int32, [None, None])
        self.Y = tf.compat.v1.placeholder(tf.int32, [None, None])
        self.C = tf.compat.v1.placeholder(tf.float32, [None, self.num_prop])
        self.L = tf.compat.v1.placeholder(tf.int32, [None])
        with tf.compat.v1.variable_scope('decode'):
            self.decoder = tf.compat.v1.nn.rnn_cell.MultiRNNCell([
                tf.compat.v1.nn.rnn_cell.LSTMCell(self.unit_size)
                for _ in range(self.n_rnn_layer)])
        with tf.compat.v1.variable_scope('encode'):
            self.encoder = tf.compat.v1.nn.rnn_cell.MultiRNNCell([
                tf.compat.v1.nn.rnn_cell.LSTMCell(self.unit_size)
                for _ in range(self.n_rnn_layer)])
        self.eps = {'eps': tf.random.normal([tf.shape(self.X)[0], self.latent_size],
                                            mean=self.prior_mean, stddev=self.stddev)}
        variance = tf.compat.v1.keras.initializers.VarianceScaling
        self.weights = {
            'softmax': tf.compat.v1.get_variable('softmaxw', initializer=tf.random.uniform(
                [self.unit_size, self.vocab_size], minval=-0.1, maxval=0.1)),
            'out_mean': tf.compat.v1.get_variable('outmeanw', shape=[self.unit_size, self.latent_size],
                initializer=variance(scale=1.0, mode='fan_avg', distribution='uniform')),
            'out_log_sigma': tf.compat.v1.get_variable('outlogsigmaw', shape=[self.unit_size, self.latent_size],
                initializer=variance(scale=1.0, mode='fan_avg', distribution='uniform')),
        }
        self.biases = {
            'softmax': tf.compat.v1.get_variable('softmaxb', initializer=tf.zeros([self.vocab_size])),
            'out_mean': tf.compat.v1.get_variable('outmeanb', shape=[self.latent_size],
                                                  initializer=tf.compat.v1.zeros_initializer()),
            'out_log_sigma': tf.compat.v1.get_variable('outlogsigmab', shape=[self.latent_size],
                                                       initializer=tf.compat.v1.zeros_initializer()),
        }
        # Preserve both variables and the shared lookup used by the old decoder.
        # The unused decode_embedding variable remains in the checkpoint schema.
        self.embedding_encode = tf.compat.v1.get_variable('encode_embedding',
            shape=[self.latent_size, self.vocab_size],
            initializer=tf.compat.v1.random_uniform_initializer(minval=-0.1, maxval=0.1))
        self.embedding_decode = tf.compat.v1.get_variable('decode_embedding',
            shape=[self.latent_size, self.vocab_size],
            initializer=tf.compat.v1.random_uniform_initializer(minval=-0.1, maxval=0.1))
        self.latent_vector, self.mean, self.log_sigma = self.encode()
        self.decoded, logits = self.decode(self.latent_vector)
        weights = tf.sequence_mask(self.L, tf.shape(self.X)[1], dtype=tf.float32)
        self.reconstr_loss = tf.reduce_mean(tfa.seq2seq.sequence_loss(
            logits=logits, targets=self.Y, weights=weights))
        self.latent_loss = self.cal_latent_loss(self.mean, self.log_sigma)
        self.loss = self.reconstr_loss + self.latent_loss
        self.opt = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.mol_pred = tf.argmax(self.decoded, axis=2)

    def encode(self):
        x = tf.nn.embedding_lookup(self.embedding_encode, self.X)
        c = tf.tile(tf.expand_dims(self.C, 1), [1, tf.shape(x)[1], 1])
        _, state = tf.compat.v1.nn.dynamic_rnn(self.encoder, tf.concat([x, c], axis=-1),
            dtype=tf.float32, scope='encode', sequence_length=self.L)
        _, hidden = state[-1]
        mean = tf.matmul(hidden, self.weights['out_mean']) + self.biases['out_mean']
        log_sigma = tf.matmul(hidden, self.weights['out_log_sigma']) + self.biases['out_log_sigma']
        z = mean + tf.exp(log_sigma / 2.0) * self.eps['eps']
        return z, mean, log_sigma

    def decode(self, z):
        batch = tf.shape(self.X)[0]
        sequence = tf.shape(self.X)[1]
        z = tf.tile(tf.expand_dims(z, 1), [1, sequence, 1])
        c = tf.tile(tf.expand_dims(self.C, 1), [1, sequence, 1])
        x = tf.nn.embedding_lookup(self.embedding_encode, self.X)
        self.initial_decoded_state = tuple(
            tf.compat.v1.nn.rnn_cell.LSTMStateTuple(
                tf.zeros([batch, self.unit_size]), tf.zeros([batch, self.unit_size]))
            for _ in range(self.n_rnn_layer))
        y, self.output_decoded_state = tf.compat.v1.nn.dynamic_rnn(
            self.decoder, tf.concat([z, x, c], axis=-1), dtype=tf.float32,
            scope='decode', sequence_length=self.L, initial_state=self.initial_decoded_state)
        y = tf.reshape(y, [-1, self.unit_size])
        logits = tf.reshape(tf.matmul(y, self.weights['softmax']) + self.biases['softmax'],
                            [batch, sequence, self.vocab_size])
        return tf.nn.softmax(logits), logits

    @staticmethod
    def cal_latent_loss(mean, log_sigma):
        return tf.reduce_mean(-0.5 * (1 + log_sigma - tf.square(mean) - tf.exp(log_sigma)))

    def train(self, x, y, lengths, properties):
        _, reconstruction, latent = self.sess.run(
            [self.opt, self.reconstr_loss, self.latent_loss],
            feed_dict={self.X: x, self.Y: y, self.L: lengths, self.C: properties})
        return float(reconstruction + latent)

    def loss_components(self, x, y, lengths, properties, deterministic=True):
        feed = {self.X: x, self.Y: y, self.L: lengths, self.C: properties}
        if deterministic:
            # Evaluate the posterior mean; this is not a Monte Carlo ELBO estimate.
            feed[self.eps['eps']] = np.zeros((len(x), self.latent_size), dtype=np.float32)
        values = self.sess.run([self.reconstr_loss, self.latent_loss], feed_dict=feed)
        return tuple(float(value) for value in values)

    def test(self, x, y, lengths, properties):
        return sum(self.loss_components(x, y, lengths, properties))

    def get_latent_vector(self, x, c, lengths):
        return self.sess.run(self.latent_vector, feed_dict={self.X: x, self.C: c, self.L: lengths})

    def sample(self, latent_vector, properties, start_codon, seq_length):
        latent_vector = np.asarray(latent_vector, dtype=np.float32)
        properties = np.asarray(properties, dtype=np.float32)
        x = np.asarray(start_codon, dtype=np.int32)
        if latent_vector.ndim != 2 or latent_vector.shape[1] != self.latent_size:
            raise ValueError('Expected latent vectors shaped (batch, latent_size).')
        batch = len(latent_vector)
        if batch < 1 or properties.shape != (batch, self.num_prop) or x.shape != (batch, 1):
            raise ValueError('Inconsistent sampling batch dimensions.')
        if seq_length < 1:
            raise ValueError('seq_length must be positive.')
        lengths = np.ones(batch, dtype=np.int32)
        predictions, state = [], None
        for _ in range(seq_length):
            feed = {self.X: x, self.latent_vector: latent_vector, self.L: lengths, self.C: properties}
            if state is not None:
                feed[self.initial_decoded_state] = state
            x, state = self.sess.run([self.mol_pred, self.output_decoded_state], feed_dict=feed)
            predictions.append(x)
        return np.concatenate(predictions, axis=1).astype(np.int32)

    def save(self, ckpt_path, global_step=None):
        return self.saver.save(self.sess, str(ckpt_path), global_step=global_step)

    def restore(self, ckpt_path):
        available = dict(tf.train.list_variables(str(ckpt_path)))
        problems = [f'{name}: expected {shape}, found {available.get(name)}'
                    for name, shape in self._checkpoint_variables.items()
                    if available.get(name) != shape]
        if problems:
            raise ValueError('Checkpoint architecture mismatch:\n' + '\n'.join(problems[:10]))
        self.saver.restore(self.sess, str(ckpt_path))

    def assign_lr(self, learning_rate):
        self.lr.load(float(learning_rate), session=self.sess)

    def close(self):
        self.sess.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
