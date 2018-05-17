import tensorflow as tf

class Process:
    def __init__(self, data, params):
        self._epochs, self._batch_size = params['epochs'], params['batch_size']

        init = tf.global_variables_initializer()

        self._sess = tf.Session()
        self._sess.run(init)

        self._data, last, self._iterations = list(), 0, int(data.shape[0] / self._batch_size)
        for i in range(self._iterations):
            if i != 0:
                self._data.append(data[last:i * self._batch_size])
                last = i * self._batch_size

    def run(self, graph):
        for i in range(self._epochs):
            tmp = iter(self._data)
            for j in range(self._iterations - 1):
                batch = next(tmp)
                _, l = self._sess.run([graph.optimize, graph.loss], feed_dict={graph._input_tensor: batch})
            if i % 5 == 0:
                print('Epoch:', i, 'Loss:', l)