import sys
import os
import time
import argparse
from util.sequences_util import midifile2pianoroll
from align.sequence_aligner import DPSequenceAligner

parser = argparse.ArgumentParser()
parser.add_argument('--midi_file', type=str, default='')
args = parser.parse_args()

ref_midi = 'data/Waltz-of-the-Flowers_REF.mid'
reg_midi = 'data/Waltz-of-the-Flowers_REG.mid'


fps = 16000.0 / 512
ref_score = midifile2pianoroll(ref_midi, fps)[0] 
reg_score = midifile2pianoroll(reg_midi, fps)[0] 

print('original ref shape:{}'.format(ref_score.shape))
print('original reg shape:{}'.format(reg_score.shape))

aligner = DPSequenceAligner(min_blank_frame=1, dist_type='e', valid_range=1, jump_penalty=1, half_search_window=200, update_interval=10)

# feed ref score
aligner.set_ref_scores(ref_score)
s = time.time()
interval = 32
for i in range(0, reg_score.shape[0], interval):
    frames = reg_score[i:i+interval]
    aligner.add_reg_scores(frames)
print("alignment cost time: {}s".format(time.time() - s))
print('squeezed ref shape:{}'.format(aligner.ref_pool.squeezed_scores.shape))
print('squeezed reg shape:{}'.format(aligner.reg_pool.squeezed_scores.shape))
aligner.save_intermedia_result('data/alignment_meta_data.npz')
