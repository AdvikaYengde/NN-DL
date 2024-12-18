import tensorflow as tf
n_inputs = 28*28 # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")
def neuron_layer(X, n_neurons, name, activation=None):
 with tf.name_scope(name):
 n_inputs = int(X.get_shape()[1])
 stddev = 2 / np.sqrt(n_inputs + n_neurons)
 init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
 W = tf.Variable(init, name="kernel")
 b = tf.Variable(tf.zeros([n_neurons]), name="bias")
 Z = tf.matmul(X, W) + b
 if activation is not None:
 return activation(Z)
 else:
 return Z
with tf.name_scope("dnn"):
 hidden1 = neuron_layer(X, n_hidden1, name="hidden1",
 activation=tf.nn.relu)
 hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2",
 activation=tf.nn.relu)
 logits = neuron_layer(hidden2, n_outputs, name="outputs")
with tf.name_scope("loss"):
 xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
 logits=logits)
 loss = tf.reduce_mean(xentropy, name="loss")
with tf.name_scope("eval"):
 correct = tf.nn.in_top_k(logits, y, 1)
 accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()


Model Architecture

Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_5 (Conv2D)           (None, 256, 256, 256)     7168      
                                                                 
 max_pooling2d_5 (MaxPooling  (None, 128, 128, 256)    0         
 2D)                                                             
                                                                 
 conv2d_6 (Conv2D)           (None, 128, 128, 128)     295040    
                                                                 
 max_pooling2d_6 (MaxPooling  (None, 64, 64, 128)      0         
 2D)                                                             
                                                                 
 conv2d_7 (Conv2D)           (None, 64, 64, 64)        73792     
                                                                 
 max_pooling2d_7 (MaxPooling  (None, 32, 32, 64)       0         
 2D)                                                             
                                                                 
 flatten_2 (Flatten)         (None, 65536)             0         
                                                                 
 dense_4 (Dense)             (None, 64)                4194368   
                                                                 
 dense_5 (Dense)             (None, 10)                650       
                                                                 
=================================================================
Total params: 4,571,018
Trainable params: 4,571,018
Non-trainable params: 0
_________________________________________________________________
