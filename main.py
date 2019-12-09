from loader_midi_direct import load_noteseqs
from model import *

import pickle


def main():

    # Used for Pickling Train Data - Only to be used first time
    PICKLE_TRAIN_DATA = False

    # Used for Testing Model (Debugging) with Single Batch Data
    IS_DEBUGGING = False

    # Used for Choosing Training / Testing Mode
    MODE = 'TRAIN'

    # Training Data Pickling
    if PICKLE_TRAIN_DATA: 
        load_noteseqs()

    # Create Piano Genie Model
    model = PianoGenie()

    # Load Notesequences
    if IS_DEBUGGING: 
        note_tensors = pickle.load(open('pickled_note_test_batch.p', 'rb'))
    else:
        note_tensors = pickle.load(open('pickled_tensors.p', 'rb'))

    print(note_tensors['midi_pitches'].shape)
    # Create Checkpoints
    checkpt = tf.train.Checkpoint(model=model)
    checkpt_dir = './checkpoints'

    # Training (Batching done in model)
    if MODE == 'TRAIN':
        manager = tf.train.CheckpointManager(checkpt, checkpt_dir, max_to_keep=2)
        cnt = 1

        while True:
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