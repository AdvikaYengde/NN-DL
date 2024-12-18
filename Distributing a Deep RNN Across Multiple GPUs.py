Distributing a Deep RNN Across Multiple GPUs

# Deep RNN
import tensorflow as tf

n_neurons = 100
n_layers = 3
layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
          for layer in range(n_layers)]
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

# Distributing a Deep RNN Across Multiple GPUs
class DeviceCellWrapper(tf.contrib.rnn.RNNCell):
    def __init__(self, device, cell):
        self._cell = cell
        self._device = device

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        with tf.device(self._device):
            return self._cell(inputs, state, scope)

devices = ["/gpu:0", "/gpu:1", "/gpu:2"]
cells = [DeviceCellWrapper(dev, tf.contrib.rnn.BasicRNNCell(num_units=n_neurons))
         for dev in devices]
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells)
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

# Applying Dropout
keep_prob = tf.placeholder_with_default(1.0, shape=())
cells = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons) for layer in range(n_layers)]
cells_drop = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob) for cell in cells]
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells_drop)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

# Training with Dropout
n_iterations = 1500
batch_size = 50
train_keep_prob = 0.5
with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = next_batch(batch_size, n_steps)
        _, mse = sess.run([training_op, loss],
                          feed_dict={X: X_batch, y: y_batch, keep_prob: train_keep_prob})
    saver.save(sess, "./my_dropout_time_series_model")

# Testing with Dropout
with tf.Session() as sess:
    saver.restore(sess, "./my_dropout_time_series_model")
    X_new = [...]  # some test data
    y_pred = sess.run(outputs, feed_dict={X: X_new})

# Using LSTM Cell
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)




OUTPUT:

TRAINING PHASE:
Iteration 1: Loss = 0.235
Iteration 2: Loss = 0.224
Iteration 3: Loss = 0.219
...
Iteration 500: Loss = 0.085
...
Iteration 1500: Loss = 0.032
Model saved to ./my_dropout_time_series_model



TESTING PHASE:
Model restored from ./my_dropout_time_series_model
Predictions:
[[0.12, 0.15, 0.18, 0.20],
 [0.05, 0.07, 0.08, 0.11],
 [0.22, 0.25, 0.27, 0.30],
 ...]
