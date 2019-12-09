from loader_midi_direct import load_noteseqs
from model import *

import pickle

def shuffle(input_dict):

    num_samples = input_dict['midi_pitches'].shape[0]

    shuffled_ind = tf.random.shuffle(range(num_samples))

    input_dict['midi_pitches'] = tf.gather(input_dict['midi_pitches'], shuffled_ind)
    input_dict['delta_times_int'] = tf.gather(input_dict['delta_times_int'], shuffled_ind)

    return input_dict

def main():

    # Used for Pickling Train Data - Only to be used first time
    PICKLE_TRAIN_DATA = False

    # Used for Testing Model (Debugging) with Single Batch Data
    IS_DEBUGGING = False

    # Used for Restoring Checkpoint to Continue Training or Testing
    RESTORE_CHECKPOINT = True

    # Used for Choosing Training / Testing Mode
    MODE = 'TRAIN'

    # Create Piano Genie Model
    model = PianoGenie()

    # Create Checkpoints
    checkpt = tf.train.Checkpoint(model=model)
    checkpt_dir = './checkpoints'
    manager = tf.train.CheckpointManager(checkpt, checkpt_dir, max_to_keep=1)
    available_device = 'GPU:0' if tf.test.is_gpu_available() else 'CPU:0'

    # Training Data Pickling
    if PICKLE_TRAIN_DATA: 
        load_noteseqs()

    # Load Notesequences
    if IS_DEBUGGING: 
        note_tensors = pickle.load(open('pickled_note_test_batch.p', 'rb'))
    else:
        note_tensors = pickle.load(open('pickled_tensors.p', 'rb'))

    # Restore Checkpoint
    if RESTORE_CHECKPOINT:
        checkpt.restore(manager.latest_checkpoint)

    with tf.device('/device:' + available_device):
        if MODE == 'TRAIN':
            cnt = 1

            while True:
                # Shuffle
                note_tensors = shuffle(note_tensors)

                # Train for a single epoch
                train_loss = model.train(note_tensors)
                print('EPOCH %d Training Loss : %.5f' % (cnt, train_loss))
                
                # Save weights after a single epoch
                manager.save()
                cnt += 1
        
        if MODE == 'TEST':
            checkpt.restore(manager.latest_checkpoint)
            # model.evaluate()



if __name__ == '__main__':
    main()