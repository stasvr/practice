import tensorflow as tf

class Model:
    
    def __init__(self, hidden_units, params, optimizer=tf.train.AdamOptimizer()):
        embedding_size, self.vocab_size = params['embedding_size'], params['vocab_size']
        tf.reset_default_graph()
        
        # Placeholders and variables used in graph
        self.in_encoder = tf.placeholder(shape=(None, None), dtype=tf.int32)
        self.in_decoder = tf.placeholder(shape=(None, None), dtype=tf.int32)
        self.out_decoder = tf.placeholder(shape=(None, None), dtype=tf.int32)
        
        input_embedding_size = 20
        embeddings = tf.Variable(tf.random_uniform([self.vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
        encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.in_encoder)
        decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.in_decoder)
        
        # encoder
        tmp_encoder = [
            hidden_units[0],
            encoder_inputs_embedded
        ]
        encoder_outputs, encoder_final_state = self.encoder(tmp_encoder)
        
        # decoder
        tmp_decoder = [
            hidden_units[1],
            decoder_inputs_embedded,
            encoder_final_state
        ]
        decoder_outputs, decoder_final_state = self.decoder(tmp_decoder)
        
        decoder_logits = tf.contrib.layers.linear(decoder_outputs, self.vocab_size)
        self.decoder_prediction = tf.argmax(decoder_logits, 2)
        
        # Initialisation of loss funstion and optimisation method with constraits for weights changing
        self._loss = self.cost([self.out_decoder, decoder_logits])
        self._optimizer = optimizer.minimize(self._loss)
        
        
    def encoder(self, inner):
        encoder_cell = tf.contrib.rnn.LSTMCell(inner[0])
        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, inner[1],
                                                                 dtype=tf.float32,
                                                                 time_major=True)
        
        return encoder_outputs, encoder_final_state

    def decoder(self, inner):
        decoder_cell = tf.contrib.rnn.LSTMCell(inner[0])
        decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_cell, inner[1],
                                                                 initial_state=inner[2],
                                                                 dtype=tf.float32, time_major=True,
                                                                 scope="plain_decoder")
        return decoder_outputs, decoder_final_state
    
    def cost(self, inner):
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(inner[0], depth=self.vocab_size, dtype=tf.float32),
            logits=inner[1])
        
        loss = tf.reduce_mean(stepwise_cross_entropy)
        return loss
    
    @property
    def loss(self):
        return self._loss
    
    @property
    def optimze(self):
        return self._optimizer