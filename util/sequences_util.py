from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import copy
import numpy as np
import librosa

import util.midi_io as midi_io

MIN_MIDI_PITCH = librosa.note_to_midi('A0')
MAX_MIDI_PITCH = librosa.note_to_midi('C8')


    
def midifile2pianoroll(midi_file, frame_per_second=16000/512.0):
    note_sequence = midi_io.midi_file_to_note_sequence(midi_file)
    pianoroll = midi_io.sequence_to_pianoroll(note_sequence, frame_per_second, MIN_MIDI_PITCH, MAX_MIDI_PITCH, onset_window=0)
    onsets = pianoroll.onsets
    frames = pianoroll.active
    velocity = pianoroll.onset_velocities
    return onsets, frames, velocity


def squeeze_pianoroll(pianoroll, min_blank_frame=1):
    '''
    remove those frames that all pitches are inactive 
    but keep the interval between no-blank frames to a minimum distance 1 if there are blank frames originally.
    '''
    
    pr_length, index = pianoroll.shape[0], 0
    squeezed_pianoroll = copy.deepcopy(pianoroll) 
    origin_index = list()
    del_frames = 0

    while index < pr_length:
        if np.sum(pianoroll[index]) != 0:
            origin_index.append(index)
            index += 1
            continue
        
        # search for contiuous blank frames
        start_index = index
        while index < pr_length and np.sum(pianoroll[index]) == 0:
            index += 1
        end_index = index
        
        # less than $(min_blak_frame+1) blank frame, skip
        if end_index - start_index < min_blank_frame+1:
            for i in range(start_index, end_index):   
                origin_index.append(i)
            continue
        
        # remove extra blank frames 
        for del_index in range(start_index+(min_blank_frame), end_index):
            squeezed_pianoroll = np.delete(squeezed_pianoroll, [del_index-del_frames], axis=0)
            del_frames += 1
        for i in range(min_blank_frame):
            origin_index.append(start_index+i)
        
    return squeezed_pianoroll, origin_index 
