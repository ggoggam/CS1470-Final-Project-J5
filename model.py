import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, StackedRNNCells, LSTMCell, Dense, RNN

import util
import loader_midi_direct

import pretty_midi
import pickle

from magenta.music.midi_io import midi_to_note_sequence


class Encoder(tf.keras.layers.Layer):

    """
    Encoder Architecture

        - Uses Double-Stacked Bidirectional LSTM

        - Dense (88 -> 128)
        - LSTM (128 -> 256)
    """

    def __init__(self):

        super(Encoder, self).__init__()

        self.num_layers = 2
        self.rnn_units = 128

        cells = StackedRNNCells([LSTMCell(self.rnn_units) for _ in range(self.num_layers)])

        self.lift = Dense(self.rnn_units)
        self.lstm = Bidirectional(RNN(cells, return_sequences=True, return_state=True))

    @tf.function
    def call(self, inputs):

        outputs = self.lift(inputs)
        outputs, h_fw, h_bw, _, _ = self.lstm(outputs) 

        final_state = tf.concat([h_fw[1], h_bw[1]], axis=1)

        return outputs, final_state

class Decoder(tf.keras.layers.Layer):

    """
    Decoder Architecture

        - Uses Double-Stacked Unidirectional LSTM

        - Dense (8 -> 128)
        - LSTM (128 -> 256)
    """

    def __init__(self, num_layers=2, rnn_units=128):

        super(Decoder, self).__init__()

        self.num_layers = 2
        self.rnn_units = 128

        cells = StackedRNNCells([LSTMCell(self.rnn_units) for _ in range(self.num_layers)])
        self.state_size = cells.state_size
        
        self.lift = Dense(self.rnn_units)
        self.lstm = RNN(cells, return_sequences=True, return_state=True)

    @tf.function
    def call(self, inputs, run_time=False, input_state=None):

        inputs = self.lift(inputs)
        init_state = [[tf.zeros([32,128]), tf.zeros([32, 128])] for _ in range(self.num_layers)]

        if run_time:
            init_state = input_state
        outputs, h, c = self.lstm(inputs, initial_state=init_state)
        final_state = [h, c]

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

    def __init__(self, is_training=True, randomize_seq_length=False):

        super(PianoGenie, self).__init__()

        # Hyperparameters
        self.num_buttons = 8
        self.seq_embedding_dim = 64
        
        self.batch_size = 32
        self.max_seq_length = 128

        if randomize_seq_length:
            self.seq_length = tf.random.uniform([self.batch_size], 1, self.seq_length+1, dtype=tf.int32)
            self.stp_varlen_mask = tf.sequence_mask(self.seq_length, maxlen=self.max_seq_length, dtype=tf.float32)
        else:
            self.seq_length = tf.ones([self.batch_size], dtype=tf.int32) * self.max_seq_length
            self.stp_varlen_mask = None

        # Layers
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.enc_pre_dense = Dense(1)
        self.dec_dense = Dense(88)

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

    def iqst(self, x, nbins):

        """
            Integer Quantization with Straight-Through (w/ Straight Through used in .call)
        """

        eps = 1e-7
        s = float(nbins - 1)
        xp = tf.clip_by_value((x + 1) / 2.0, -eps, 1 + eps)
        xpp = tf.round(s * xp)
        xppp = 2 * (xpp / s) - 1

        return xpp, x + tf.stop_gradient(xppp - x)

    def weighted_avg(self, t, mask=None, axis=None, expand_mask=False):

        """
            Customized Weighted Average
        """

        if mask is None:
            return tf.reduce_mean(t, axis=axis)
        else:
            if expand_mask:
                mask = tf.expand_dims(mask, axis=-1)
            else:
                return tf.reduce_sum(tf.multiply(t, mask), axis=axis) / tf.reduce_sum(mask, axis=axis)

    def call(self, input_dict):
        
        output_dict = {}

        # Parse Inputs
        pitches = util.midi2piano(input_dict['midi_pitches'])

        # Create Encoder Features (Pitch, Delta, Velocity)
        enc_input = []

        enc_input.append(tf.one_hot(pitches, 88))
        enc_input.append(tf.one_hot(input_dict['delta_times_int'], 33))
        enc_input = tf.concat(enc_input, axis=2)
        
        # Run through LSTM Encoder
        latent = []
        enc_stp, _ = self.encoder(enc_input)

        # Reduce Dimension
        pre_enc_stp = tf.squeeze(self.enc_pre_dense(enc_stp))

        # Run through IQST
        stp_emb_disc_f, stp_emb_disc_rescaled = self.iqst(pre_enc_stp, self.num_buttons)
        stp_emb_disc = tf.cast(tf.cast(stp_emb_disc_f + 1e-4, tf.int32), tf.float32)
        stp_emb_qnt = tf.expand_dims(stp_emb_disc_rescaled, axis=2)
        
        # Check if output is between [-1,1] then mask those in range
        stp_emb_inrange_mask = tf.cast(tf.logical_and(tf.greater_equal(pre_enc_stp, -1), 
                                       tf.less_equal(pre_enc_stp, 1)), 
                                       tf.float32)
        stp_emb_valid_p = self.weighted_avg(stp_emb_inrange_mask, self.stp_varlen_mask)

        # Regularization for output within range (Margin)
        stp_emb_range_penalty = self.weighted_avg(tf.square(
                                                  tf.maximum(
                                                  tf.abs(pre_enc_stp)-1, 0)),
                                                  self.stp_varlen_mask)

        # Regularization for latent finite differences (Contour)
        stp_emb_latents = pre_enc_stp[:,1:] - pre_enc_stp[:,:-1]
        stp_emb_notes = tf.cast(pitches[:,1:] - pitches[:,:-1], tf.float32)
        stp_emb_contour_penalty = self.weighted_avg(tf.square(
                                                    tf.maximum(
                                                    -tf.multiply(stp_emb_notes, stp_emb_latents), 0)),
                                                    None if self.stp_varlen_mask is None else self.stp_varlen_mask[:,1:])

        output_dict['stp_emb_quantized'] = stp_emb_qnt
        output_dict['stp_emb_discrete'] = stp_emb_disc
        output_dict['stp_emb_valid_p'] = stp_emb_valid_p
        output_dict['stp_emb_range_penalty'] = stp_emb_range_penalty
        output_dict['stp_emb_contour_penalty'] = stp_emb_contour_penalty
        output_dict['stp_emb_quantized_lookup'] = tf.expand_dims(2.0 * (stp_emb_disc_f / (self.num_buttons - 1.0)) - 1.0, axis=2)

        latent.append(stp_emb_qnt)

        # Create Decoder Features
        dec_input = latent

        # Autoregression Features
        curr_pitches = pitches
        last_pitches = curr_pitches[:, :-1]
        last_pitches = tf.pad(last_pitches, [[0, 0], [1, 0]], constant_values=-1)
        dec_input.append(tf.one_hot(last_pitches+1, 89))

        # Delta Times Features
        dec_input.append(tf.one_hot(input_dict['delta_times_int'], 33))
        
        # Create Decoder Features
        dec_input = tf.concat(dec_input, axis=2)

        # Decode
        dec_stp, _, dec_final_state = self.decoder(dec_input)
        dec_recon_logits = self.dec_dense(dec_stp)
        dec_recon_loss = self.weighted_avg(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pitches, logits=dec_recon_logits), self.stp_varlen_mask)

        output_dict['dec_final_state'] = dec_final_state
        output_dict['dec_recons_logits'] = dec_recon_logits
        output_dict['dec_recons_scores'] = tf.nn.softmax(dec_recon_logits, axis=-1)
        output_dict['dec_recons_predictions'] = tf.argmax(dec_recon_logits, axis=-1, output_type=tf.int32)
        output_dict['dec_recons_midi_predictions'] = util.piano2midi(output_dict['dec_recons_predictions'])
        output_dict['dec_recons_loss'] = dec_recon_loss

        return output_dict

    def evaluate(self, dec_input, last_state):

        """
            Used for Actual Demo (Converting 8-button Input to Piano Keys)
        """

        dec_stp, _, final_state = self.decoder(dec_input, run_time=True, input_state=last_state)
        dec_recon_logits = self.dec_dense(dec_stp)
        
        return dec_recon_logits, final_state

    def loss(self, output_dict):

        loss = output_dict['dec_recons_loss']
        loss = loss + output_dict['stp_emb_range_penalty']
        loss = loss + output_dict['stp_emb_contour_penalty']

        perp = tf.exp(output_dict['dec_recons_loss'])
        
        return loss, perp

    def train(self, input_dict):
        n, total = self.batch_size, input_dict['midi_pitches'].shape[0]

        for i in range(total//n):
            start, end = i * n, (i+1) * n
            inputs = {'midi_pitches': input_dict['midi_pitches'][start:end],
                      'delta_times_int': input_dict['delta_times_int'][start:end]}

            with tf.GradientTape() as tape:
                outputs = self.call(inputs)
                loss, _ = self.loss(outputs)

            grad = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        return loss

    def test(self, input_dict):
        n, total = self.batch_size, input_dict['midi_pitches'].shape[0]

        avg_loss, avg_perp = 0, 0
        for i in range(total//n):
            start, end = i * n, (i+1) * n
            inputs = {'midi_pitches': input_dict['midi_pitches'][start:end],
                      'delta_times_int': input_dict['delta_times_int'][start:end]}

            outputs = self.call(inputs)
            loss, perp = self.loss(outputs)
            avg_loss += loss
            avg_perp += perp
        
        return avg_loss / (total//n), avg_perp / (total//n)