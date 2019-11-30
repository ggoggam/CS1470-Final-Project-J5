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
            - Learns as Autoencoder Architecture 
                (88-key Input ---Encoder---> 8-key Compressed ---Decoder---> 88-key Output)
            - Goal: Learn the best sounding representation given a compressed input
        - Testing
            - Equivalent to Decoding

        - Encoder
        - Integer Quantization Straight-Through
        - Decoder
        - Losses (Sum)
            - Reconstruction
            - Contour
            - Marginal
    """

    def __init__(self):

        super(PianoGenie, self).__init__()

        # Hyperparameters
        self.num_buttons = 8                # Used for binning
        self.seq_embedding_dim = 64         # Used for embedding
        
        self.seq_length = 15
        self.stp_varlen_mask = tf.sequence_mask(self.seq_length, max)

        # Layers
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.enc_pre_dense = Dense(1)
        self.dec_dense = Dense(88)

        # Optimizer
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, )

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

        # Run through LSTM Encoder
        enc_out, enc_seq = self.encoder(inputs)

        # Reduce
        pre_enc_out = self.enc_pre_dense(enc_out)

        # Run through IQST
        stp_emb_disc, stp_emb_disc_rescaled = self.iqst(pre_enc_out, self.num_buttons)
        stp_emb_disc = tf.cast(tf.cast(stp_emb_disc + 1e-4, tf.int32), tf.float32)
        stp_emb_qnt = tf.expand_dims(stp_emb_disc_rescaled, axis=2)
        
        # Check if output is between [-1,1] then mask those in range
        stp_emb_inrange = tf.logical_and(tf.greater_equal(pre_enc_out, -1), tf.less_equal(pre_enc_out, 1))
        stp_emb_inrange_mask = tf.cast(stp_emb_inrange, tf.float32)
        stp_emb_valid_p = self.weighted_avg(stp_emb_inrange_mask, self.stp_varlen_mask)



        pass

    def loss(self, inputs):
        
        # Regularization (Encoder output within range)
        stp_emb_range_penalty = self.weighted_avg(tf.square(tf.maximum(tf.abs(pre_enc_out)-1, 0)), self.stp_varlen_mask)

        # Regularization (Correlating latent finite differences to input)
        stp_emb_latents = pre_enc_out[:, 1:] - pre_enc_out[:, :-1]
        stp_emb_notes = tf.cast(pitches[:, 1:] - pitches[:, :-1], tf.float32)
        stp_emb_contour_penalty = self.weighted_avg(
            tf.square(
                tf.maxmimum(
                    0.0 - tf.multiply(stp_emb_notes, stp_emb_latents), 0)), 
                    None if self.stp_varlen_mask is None else self.stp_varlen_mask[:, 1:])

        # Regularizatoin (Note consistency)
        stp_emb_note_held = tf.cast(tf.equal(pitches[:, 1:] - pitches[:, :-1], 0), tf.float32)

        
        
        pass

