# Learn more or give us feedback
# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from magenta.music.midi_io import midi_to_note_sequence

import random
import numpy as np
import tensorflow as tf

import pretty_midi
import pickle

def load_noteseqs(seq_len = 128,
                  repeat_sample = 2,
                  max_discrete_times = 32,
                  augment_stretch_bounds = (0.95, 1.05),
                  augment_transpose_bounds = (-6, 6)):

    """
        Modified loading function from the original Piano Genie code.

        Loads random subsequences from MIDI files to a dictionary of numpy arrays.

        The MIDI files must be in ./midi_data directory.

        Args:
            seq_len : Length of subsequences.
            repeat_sample : Number of times going through records
            max_discrete_times : Maximum number of time buckets at 31.25Hz.
            augment_stretch_bounds : Tuple containing speed ratio range.
            augment_transpose_bounds : Tuple containing semitone augmentation range.

        Returns:
            None
            The function should automatically save a pickled file of the dictionary.
    """

    # Deserializes NoteSequences and extracts numeric tensors
    def _str_to_tensor(note_sequence,
                       augment_stretch_bounds=(0.95, 1.05),
                       augment_transpose_bounds=(-6, 6)):
        
        note_sequence_ordered = sorted(list(note_sequence.notes), key=lambda n: (n.start_time, n.pitch))

        # Transposition Data Segmentation
        transpose_factor = np.random.randint(*augment_transpose_bounds)
        for note in note_sequence_ordered:
            note.pitch += transpose_factor
            note_sequence_ordered = [n for n in note_sequence_ordered if (n.pitch >= 21) and (n.pitch <= 108)]

        pitches = np.array([note.pitch for note in note_sequence_ordered])
        start_times = np.array([note.start_time for note in note_sequence_ordered])

        # Tempo Data Augmentation
        stretch_factor = np.random.uniform(*augment_stretch_bounds)
        start_times *= stretch_factor

        # Delta time start high to indicate free decision
        delta_times = np.concatenate([[100000.], start_times[1:] - start_times[:-1]])
        
        return np.stack([pitches, delta_times], axis=1).astype(np.float32)

    # Filter out excessively short examples
    def _filter_short(note_sequence_tensor, seq_len):
        note_sequence_len = tf.shape(note_sequence_tensor)[0]

        return tf.greater_equal(note_sequence_len, seq_len)

    # Take a random crop of a note sequence
    def _random_crop(note_sequence_tensor, seq_len):
        note_sequence_len = tf.shape(note_sequence_tensor)[0]
        start_max = note_sequence_len - seq_len
        start_max = tf.maximum(start_max, 0)

        start = tf.random.uniform([], maxval=start_max + 1, dtype=tf.int32)
        seq = note_sequence_tensor[start:start + seq_len]

        return seq

    # Find sharded filenames
    filenames = tf.io.gfile.glob("test_data/*.midi")

    note_sequences_ls = []
    for _ in range(repeat_sample):
        cnt = 1
        for fn in filenames:
            ns = midi_to_note_sequence(pretty_midi.PrettyMIDI(fn))
            ns_ts = _str_to_tensor(ns, augment_stretch_bounds, augment_transpose_bounds)

            if _filter_short(ns_ts, seq_len):
                note_sequences_ls.append(_random_crop(ns_ts, seq_len))

            print('PROCESSING %d / %d : %s' % (cnt, len(filenames), fn))
            cnt += 1

    note_sequence_tensors = tf.convert_to_tensor(note_sequences_ls)

    # Set lists as tensors of given shape
    note_sequence_tensors.set_shape([None, seq_len, 2])

    # Retrieve tensors
    note_pitches = tf.cast(note_sequence_tensors[:, :, 0] + 1e-4, tf.int32)
    note_delta_times = note_sequence_tensors[:, :, 1]

    # Onsets and frames model samples at 31.25Hz
    note_delta_times_int = tf.cast(tf.round(note_delta_times * 31.25) + 1e-4, tf.int32)

    # Reduce time discretizations to a fixed number of buckets
    note_delta_times_int = tf.minimum(note_delta_times_int, max_discrete_times)

    # Build return dict
    note_tensors = {"midi_pitches": note_pitches, "delta_times_int": note_delta_times_int}

    file = open('pickled_tensors_test.p', 'wb')
    pickle.dump(note_tensors, file)
    file.close()

    pass