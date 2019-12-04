import numpy as np
import pretty_midi
import simpleaudio as sa
import time

def MIDI_file_to_Notes(MIDI_file_name):
    # Input : MIDI file
    # Output: numpy_array shape(n_notes, 2)
    # Each entry has 'note number' and 'note start time'
    # Only piano notes will be recorded.
    midi_obj  = pretty_midi.PrettyMIDI(MIDI_file_name)
    print(midi_obj); print(midi_obj.instruments)
    note_list = [ ]
    for instrument in midi_obj.instruments:
        if instrument.program == 0: #detect piano; usually = 0
            for note in instrument.notes:
                entry = [note.pitch, note.start]
                note_list.append(entry)
        elif instrument.program == 1: #detect piano; sometimes = 1
            for note in instrument.notes:
                entry = [note.pitch, note.start]
                note_list.append(entry)            
    note_array = np.array(note_list); print(note_array)
    return note_array

def Notes_to_Sound(note_array, delay_correction = 0.3):
    # Input: 1) note_vector prepared by MIDI_file_to_note_vector function
    # Input: 2) delay_correction: if too fast, increase it slightly [0.1-1]
    # It will use simpleaudio's wav objects.
    # All 88 wav objects for piano notes are stored in a file "my_piano.npy"
    # Note that my_piano requires note_number (21 to 108)
    my_piano  = np.load("my_piano.npy", allow_pickle=True)
    n_notes   = note_array.shape[0]
    piano_fig = [" "] *88
    for i in range(n_notes):
        note_number = int(note_array[i][0])
        my_piano[note_number - 21].play()
        if i < n_notes-1: duration = (note_array[i+1][1] - note_array[i][1]) * delay_correction # compter delay correction
        if duration < 0: duration = 0 # chords (not single note) will have duration ~ 0
        piano_fig[note_number - 21] = str(note_number); print("".join(piano_fig)) # If too slow, remove this line.
        if duration/delay_correction > 0.001: piano_fig = [" "] *88 # If too slow, remove this line.
        time.sleep(duration)

note_array = MIDI_file_to_Notes("1.mid")
Notes_to_Sound(note_array)
