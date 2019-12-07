import tensorflow as tf

import util
from tensorflow.keras.layers import Bidirectional, StackedRNNCells, LSTMCell, Dense, RNN

import loader

import pretty_midi

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

        cells = StackedRNNCells([LSTMCell(self.rnn_units)] * self.num_layers)

        self.lift = Dense(self.rnn_units)
        self.lstm = Bidirectional(RNN(cells, return_sequences=True, return_state=True))

    @tf.function
    def call(self, inputs):

        outputs = self.lift(inputs)
        outputs, _, h2, _, _ = self.lstm(outputs) 

        final_state = tf.concat([h2[0], h2[1]], axis=1)

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

        cells = StackedRNNCells([LSTMCell(self.rnn_units)] * self.num_layers)

        self.lift = Dense(self.rnn_units)
        self.lstm = RNN(cells, return_state=True)

    @tf.function
    def call(self, inputs):
        
        init_state = tf.zeros(inputs.shape)
        outputs, _, h2, _, _ = self.lstm(inputs, initial_state=init_state)

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
        self.embedding = Dense(64)
        self.dec_dense = Dense(88)
        self.vel_dense = Dense(17)

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
        pitches = input_dict['midi_pitches']

        # Create Encoder Features (Pitch, Delta, Velocity)
        enc_input = []

        enc_input.append(tf.one_hot(pitches, 88))
        enc_input.append(tf.one_hot(input_dict['delta_times_int'], 33))
        
        enc_input = tf.concat(enc_input, axis=2)
        
        # Run through LSTM Encoder
        latent = []
        enc_stp, enc_seq = self.encoder(enc_input)

        # Reduce Dimension
        pre_enc_stp = self.enc_pre_dense(enc_stp)

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

        # Regularization for note consistency 
        stp_emb_note_held = tf.cast(tf.equal(pitches[:,1:] - pitches[:,:-1], 0), tf.float32)
        mask = stp_emb_note_held if self.stp_varlen_mask is None else self.stp_varlen_mask[:,1:] * stp_emb_note_held
        stp_emb_deviate_penalty = self.weighted_avg(tf.square(stp_emb_latents), mask)

        # Perplexity for Encoder
        mask = stp_emb_inrange_mask if self.stp_varlen_mask is None else self.stp_varlen_mask * stp_emb_inrange_mask
        stp_emb_disc_oh = tf.one_hot(stp_emb_disc, self.num_buttons)
        stp_emb_avg_probs = self.weighted_avg(stp_emb_disc_oh, mask, axis=[0,1], expand_mask=True)
        # stp_emb_disc_ppl = tf.exp(-tf.reduce_sum(stp_emb_avg_probs * tf.math.log(stp_emb_avg_probs + 1e-10)))

        output_dict['stp_emb_quantized'] = stp_emb_qnt
        output_dict['stp_emb_discrete'] = stp_emb_disc
        output_dict['stp_emb_valid_p'] = stp_emb_valid_p
        output_dict['stp_emb_range_penalty'] = stp_emb_range_penalty
        output_dict['stp_emb_contour_penalty'] = stp_emb_contour_penalty
        # output_dict['stp_emb_deviate_penalty'] = stp_emb_deviate_penalty
        # output_dict['stp_emb_discrete_ppl'] = stp_emb_disc_ppl
        output_dict['stp_emb_quantized_lookup'] = tf.expand_dims(2.0 * (stp_emb_disc_f / (self.num_buttons - 1.0)) - 1.0, axis=2)

        latent.append(stp_emb_qnt)

        # Embedding
        seq_emb = self.embedding(enc_seq)
        output_dict['seq_emb'] = seq_emb

        # Create Decoder Features
        dec_input = tf.concat(latent, axis=2)

        # Decode
        dec_stp, dec_init_state, dec_final_state = self.decoder(dec_input)
        dec_recon_logits = self.dec_dense(dec_stp)
        dec_recon_loss = self.weighted_avg(tf.nn.sparse_softmax_cross_entropy_with_logits(dec_recon_logits, pitches), self.stp_varlen_mask)

        output_dict['dec_init_state'] = dec_init_state
        output_dict['dec_final_state'] = dec_final_state
        output_dict['dec_reconstruction_logits'] = dec_recon_logits
        output_dict['dec_reconstruction_scores'] = tf.nn.softmax(dec_recon_logits, axis=-1)
        output_dict['dec_reconstruction_predictions'] = tf.argmax(dec_recon_logits, axis=-1, output_type=tf.int32)
        output_dict['dec_reconstruction_midi_predictions'] = util.piano2midi(output_dict['dec_reconstruction_predictions'])
        output_dict['dec_reconstruction_loss'] = dec_recon_loss

        return output_dict

    def loss(self, output_dict):

        loss = output_dict['dec_recons_loss']
        perp = tf.exp(output_dict['dec_recons_loss'])

        loss = loss + output_dict['stp_emb_range_penalty']
        loss = loss + output_dict['stp_emb_contour_penalty']

        return loss, perp

    def train(self, input_dict):
        
        # NEEDS BATCHING !!! (Done in data loader)
        print("bp 3")

        with tf.GradientTape() as tape:
            output_dict = self.call(input_dict)
            loss, _ = self.loss(output_dict)
            print(loss)

        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))  

pg = PianoGenie()
print("bp 1")
# midi_data = pretty_midi.PrettyMIDI('./data/AbdelmoulaJS02.mid')
# note_sequence = midi_to_note_sequence(midi_data)

note_tensors = loader.load_noteseqs("./data/2016beethoven.tfrecord")
print("bp 2")
# print(note_tensors["pb_strs"])
pg.train(note_tensors)

