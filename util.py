import numpy as np
import tensorflow as tf

def midi2piano(pitches):

    """
        Transforms MIDI pitches [21,108] to [0, 88) 
    """

    assertions = [tf.assert_greater_equal(pitches, 21),
                  tf.assert_less_equal(pitches, 108)]

    with tf.control_dependencies(assertions):
        return pitches - 21

def piano2midi(pitches):

    """
        Transforms MIDI pitches [21,108] to [0, 88)
    """

    assertions = [tf.assert_greater_equal(pitches, 0),
                  tf.assert_less_equal(pitches, 87)]

    with tf.control_dependencies(assertions):
        return pitches + 21