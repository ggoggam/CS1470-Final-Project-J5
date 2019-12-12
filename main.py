from loader_midi_direct import load_noteseqs
from model import *

import pickle
import argparse
from server import run_app
from run_manager import RunManager

parser = argparse.ArgumentParser(description='piano genie model args')

parser.add_argument('--mode', type=str, default='run', help='mode: train, test, or r
parser.add_argument('--restore', type=str, default='true', help='restore last training checkpoint')
parser.add_argument('--debug', type=str, default='false', help='used for debugging on single batch')
parser.add_argument('--create-data', type=str, default='false', help='create pickled train data if true')

args = parser.parse_args()

def shuffle(input_dict):

    num_samples = input_dict['midi_pitches'].shape[0]

    shuffled_ind = tf.random.shuffle(range(num_samples))

    input_dict['midi_pitches'] = tf.gather(input_dict['midi_pitches'], shuffled_ind)
    input_dict['delta_times_int'] = tf.gather(input_dict['delta_times_int'], shuffled_ind)

    return input_dict

def main():

    # Create Piano Genie Model
    model = PianoGenie()

    # Create Checkpoints
    checkpt = tf.train.Checkpoint(model=model)
    checkpt_dir = './checkpoints'
    manager = tf.train.CheckpointManager(checkpt, checkpt_dir, max_to_keep=1)
    available_device = 'GPU:0' if tf.test.is_gpu_available() else 'CPU:0'

    # Training Data Pickling
    if args.create_data == 'true': 
        load_noteseqs()

    # Load Notesequences
    if args.debug == 'true': 
        note_tensors = pickle.load(open('pickled_note_test_batch.p', 'rb'))
    if args.debug == 'false':
        note_tensors = pickle.load(open('pickled_tensors.p', 'rb'))

    # Restore Checkpoint
    if args.restore == 'true':
        checkpt.restore(manager.latest_checkpoint).expect_partial()

    with tf.device('/device:' + available_device):
        if args.mode == 'train':
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
        
        if args.mode == 'test':
            checkpt.restore(manager.latest_checkpoint).expect_partial()
            note_tensors = pickle.load(open('pickled_tensors_test.p', 'rb'))
            test_loss, test_perp = model.test(note_tensors)
            print('Test Loss : %.5f' % (test_loss, test_perp))
        
        if args.mode == 'run':
            checkpt.restore(manager.latest_checkpoint).expect_partial()
            rm = RunManager(model)
            run_app(rm)

if __name__ == '__main__':
    main()