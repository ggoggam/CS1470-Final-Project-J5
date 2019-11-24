import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, StackedRNNCells, LSTMCell, Dense, RNN

class Encoder(tf.keras.layers.Layer):

    """
        Encoder Architecture

            - Uses Double-Stacked Bidirectional LSTM

            - Dense (88 -> 128)
            - LSTM (128 -> 256)
            - Dense (256 -> 1)
    """

    def __init__(self):

        super(Encoder, self).__init__()

        self.num_layers = 2
        self.rnn_units = 128

        cells = StackedRNNCells([LSTMCell(self.rnn_units)] * self.num_layers)

        self.lift = Dense(self.rnn_units)
        self.lstm = Bidirectional(RNN(cells, return_state=True))
        self.dense = Dense(1)

    def call(self, inputs):

        outputs = self.lift(inputs)
        outputs, _, h2, _, _ = self.lstm(outputs) 
        outputs = self.dense(outputs)

        final_state = tf.concat([h2[0], h2[1]], axis=1)

        return outputs, final_state



class Decoder(tf.keras.layers):

    """
        Decoder Architecture

            - Uses Double-Stacked Unidirectional LSTM

            - Dense (8 -> 128)
            - LSTM (128 -> 256)
            - Dense (256 -> 88)
    """

    def __init__(self, num_layers=2, rnn_units=128):

        super(Decoder, self).__init__()

        self.num_layers = 2
        self.rnn_units = 128

        cells = StackedRNNCells([LSTMCell(self.rnn_units)] * self.num_layers)

        self.lift = Dense(self.rnn_units)
        self.lstm = RNN(cells, return_state=True)
        self.dense = Dense(88)

    def call(self, inputs):
        
        init_state = tf.zeros(inputs.shape)
        outputs, _, h2, _, _ = self.lstm(inputs, initial_state=init_state)
        outputs = self.dense(outputs)

        final_state = h2

        return outputs, init_state, final_state

class PianoGenie(tf.keras.Model):
    
    """
        Piano Genie Architecture

            - Training
                - Learns as Autoencoder Architecture (Encoder -> Decoder)
                - Goal: Learn the best sounding representation given a compressed input
            - Testing
                - 

            - Encoder
            - Integer Quantization Straight-Through
            - Decoder
            - Losses
                - Reconstruction
                - Contour
                - Marginal
    """

    def __init__(self):

        super(PianoGenie, self).__init__()

        # Layers
        self.encoder = Encoder()
        self.decoder = Decoder()

        # Optimizer
        # self.optimizer = tf.keras.optimizers.Adam()

    def iqst(self, x, nbins):

        eps = 1e-7
        s = float(nbins - 1)
        xp = tf.clip_by_value((x + 1) / 2.0, -eps, 1 + eps)
        xpp = tf.round(s * xp)
        xppp = 2 * (xpp / s) - 1

        return xpp, x + tf.stop_gradient(xppp - x)

    def weighted_avg(self, t, mask=None, axis=None, expand_mask=False):

        if mask is None:
            return tf.reduce_mean(t, axis=axis)
        else:
            if expand_mask:
                mask = tf.expand_dims(mask, axis=-1)
            else:
                return tf.reduce_sum(tf.multiply(t, mask), axis=axis) / tf.reduce_sum(mask, axis=axis)

    def call(self, inputs):

        pass

    def loss(self, inputs):

        pass

