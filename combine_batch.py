import pickle
import tensorflow as tf

note_tensors1 = pickle.load(open('pickled_tensors.p', 'rb'))
note_tensors2 = pickle.load(open('pickled_tensors_test.p', 'rb'))

print(note_tensors1['midi_pitches'].shape, note_tensors2['midi_pitches'].shape)

p = tf.concat([note_tensors1['midi_pitches'], note_tensors2['midi_pitches']], axis=0)
d = tf.concat([note_tensors1['delta_times_int'], note_tensors2['delta_times_int']], axis=0)

print(p.shape, d.shape)

note_tensors = {'midi_pitches': p, 'delta_times_int': d}

file = open('pickled_tensors_combined.p', 'wb')
pickle.dump(note_tensors, file)
file.close()
