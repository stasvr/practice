import tensorflow as tf
import numpy as np

class Process:
    
    def __init__(self, data, params):
        self._epochs, self._batch_size = params['epochs'], params['batch_size']
        init = tf.global_variables_initializer()

        self._sess = tf.Session()
        self._sess.run(init)

        self._data, last, self._iterations = list(), 0, int(data.shape[0] / self._batch_size)
        for i in range(self._iterations-1):
            if i != 0:
                self._data.append(data[last:i * self._batch_size])
                last = i * self._batch_size
    
    def run(self, n, graph):
        PAD, EOS = 0, 1
        for e in range(self._epochs):
            for i in range(self._iterations-1):
                batch = self._data[np.random.randint(len(self._data)-1)]
                feed = {
                    graph.in_encoder : np.array([[PAD] + [j for j in i] for i in batch]),
                    graph.in_decoder : np.array([[EOS] + [j for j in i] for i in batch]),
                    graph.out_decoder : np.array([[j for j in i] + [EOS] for i in batch])
                }
                _,l = self._sess.run([graph.optimze, graph.loss], feed_dict=feed)
            if e % n == 0:
                print('Epoch:', e, 'Loss:', l)
                
        return self._sess