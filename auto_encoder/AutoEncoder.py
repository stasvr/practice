import tensorflow as tf


class AutoEncoder:
    def __init__(self, params):
        num_input, num_hidden, b_norm = params['num_input'], params['num_hidden'], params['batch_normalisation']

        # Placeholders and variables used in graph
        self._input_tensor = tf.placeholder(shape=(None, 784), dtype=tf.float32)

        weights_hidden = [self.init_VarPair([num_hidden[i - 1], num_hidden[i]]) for i in range(1, len(num_hidden))]
        weights_hidden += [self.init_VarPair([num_hidden[i], num_hidden[i - 1]]) for i in reversed(range(1, len(num_hidden)))]
        weights_hidden = [self.init_VarPair([num_input, num_hidden[0]])] + weights_hidden
        weights_hidden += [self.init_VarPair([num_hidden[0], num_input])]

        bias_hidden = [self.init_VarSingle(num_hidden[i]) for i in range(len(num_hidden))]
        bias_hidden += [self.init_VarSingle(num_hidden[i]) for i in reversed(range(len(num_hidden) - 1))]
        bias_hidden += [self.init_VarSingle(num_input)]

        # Graph structure
        hidden_tensor = self.coder(self._input_tensor, weights_hidden[0], bias_hidden[0])
        for i in zip(weights_hidden[1:-1], bias_hidden[1:-1], num_hidden[1:] + list(reversed(num_hidden))[1:]):
            if b_norm:
                hidden_tensor = self.normalisation(self.coder(hidden_tensor, i[0], i[1]), i[2])
            else:
                hidden_tensor = self.coder(hidden_tensor, i[0], i[1])
        self._output_tensor = self.coder(hidden_tensor, weights_hidden[-1:][0], bias_hidden[-1:][0])

        # Initialisation of loss funstion and optimisation method with constraits for weights changing
        self._loss = tf.reduce_mean(tf.pow(self._input_tensor - self._output_tensor, 2))
        self._optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3, decay=3e-8).minimize(self._loss)

    def coder(self, X, w, b):
        return tf.nn.sigmoid(tf.add(tf.matmul(X, w), b))

    def normalisation(self, tensor, count):
        batch_mean, batch_var = tf.nn.moments(tensor, [0])
        scale = tf.Variable(tf.ones([count]))
        beta = tf.Variable(tf.zeros([count]))
        return tf.nn.batch_normalization(tensor, batch_mean, batch_var, beta, scale, 1e-3)

    def init_VarPair(self, pair):
        return tf.Variable(tf.random_normal([pair[0], pair[1]]))

    def init_VarSingle(self, value):
        return tf.Variable(tf.random_normal([value]))

    @property
    def out(self):
        return self._output_tensor

    @property
    def loss(self):
        return self._loss

    @property
    def optimize(self):
        return self._optimizer