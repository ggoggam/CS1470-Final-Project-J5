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

"""Loads music data from TFRecords."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from magenta.music.protobuf import music_pb2
import numpy as np
import tensorflow as tf


def load_noteseqs(fp = "./data/train*.tfrecord",
                batch_size = 32,
                seq_len = 128,
                max_discrete_times = 32,
                max_discrete_velocities = 16,
                augment_stretch_bounds = (0.95, 1.05),
                augment_transpose_bounds = (-6, 6),
                buffer_size=512):
    """
        Loads random subsequences from NoteSequences in TFRecords.

        Args:
            fp: List of shard fps.
            batch_size: Number of sequences in batch.
            seq_len: Length of subsequences.
            max_discrete_times: Maximum number of time buckets at 31.25Hz.
            max_discrete_velocities: Maximum number of velocity buckets.
            augment_stretch_bounds: Tuple containing speed ratio range.
            augment_transpose_bounds: Tuple containing semitone augmentation range.
            randomize_chord_order: If True, list notes of chord in random order.
            repeat: If True, continuously loop through records.
            buffer_size: Size of random queue.

        Returns:
            A dict containing the loaded tensor subsequences.

        Raises:
            ValueError: Invalid file format for shard filepaths.
    """

    # Deserializes NoteSequences and extracts numeric tensors
    def _str_to_tensor(note_sequence_str,
                        augment_stretch_bounds=(0.95, 1.05),
                        augment_transpose_bounds=(-6, 6)):
        
        # Note Sequence 2 Proto
        note_sequence = music_pb2.NoteSequence.FromString(note_sequence_str)
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
        
        return note_sequence_str, np.stack([pitches, delta_times], axis=1).astype(np.float32)

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
    filenames = tf.io.gfile.glob(fp)

    # Create dataset
    dataset = tf.data.TFRecordDataset(filenames)

    # Deserialize protos
    dataset = dataset.map(
        lambda data: tf.py_function(
            lambda x: _str_to_tensor(
                x, augment_stretch_bounds, augment_transpose_bounds),
            [data], (tf.string, tf.float32), stateful=False))


    # Filter sequences that are too short
    dataset = dataset.filter(lambda s, n: _filter_short(n, seq_len))

    # Get random crops
    dataset = dataset.map(lambda s, n: (s, _random_crop(n, seq_len)))

    # Shuffle
    # if repeat:
    dataset = dataset.shuffle(buffer_size=buffer_size)

    # Make batches
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Repeat
    #if repeat:
    dataset = dataset.repeat()

    # Get tensors
    iterator = dataset.__iter__()
    note_sequence_strs, note_sequence_tensors = next(iterator)

    # Set shapes
    note_sequence_strs.set_shape([batch_size])
    note_sequence_tensors.set_shape([batch_size, seq_len, 5])

    # Retrieve tensors
    note_pitches = tf.cast(note_sequence_tensors[:, :, 0] + 1e-4, tf.int32)
    note_delta_times = note_sequence_tensors[:, :, 1]

    # Onsets and frames model samples at 31.25Hz
    note_delta_times_int = tf.cast(tf.round(note_delta_times * 31.25) + 1e-4, tf.int32)

    # Reduce time discretizations to a fixed number of buckets
    note_delta_times_int = tf.minimum(note_delta_times_int, max_discrete_times)

    # Build return dict
    note_tensors = {
        "pb_strs": note_sequence_strs,
        "midi_pitches": note_pitches,
        "delta_times": note_delta_times,
        "delta_times_int": note_delta_times_int,
    }

    return note_tensors

