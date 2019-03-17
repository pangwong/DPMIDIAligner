from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np

from util.sequences_util import squeeze_pianoroll

class ScorePool():
    
    def __init__(self, squeeze=False, min_blank_frame=1):
        self.squeeze = squeeze
        
        #assert min_blank_frame in [0, 1]
        self.min_blank_frame = min_blank_frame
         
        self.full_scores = None 
        self.squeezed_scores = None 
        self.squeezed_original_index = list()
        self.tail_blank_frame = 0

    def add_scores(self, part_scores):
        full_scores_length =  self.full_scores.shape[0] if self.full_scores is not None else 0
        part_length = part_scores.shape[0]
        part_height = part_scores.shape[1] if len(part_scores.shape) > 1 else None
        if part_length < 1:
            return 0 

        # store full scores
        self.full_scores = np.concatenate([self.full_scores, part_scores], axis=0) if self.full_scores is not None else part_scores 
        
        if not self.squeeze:
            squeezed_scores = part_scores
            part_original_index = range(len(squeezed_scores))
        else:
            # squeeze part scores
            concat_shape = list(part_scores.shape)
            concat_shape[0] = self.tail_blank_frame
            squeeze_scores = part_scores if self.tail_blank_frame <= 0 else np.concatenate([np.zeros(concat_shape), part_scores], axis=0)
            squeezed_scores, part_original_index = squeeze_pianoroll(squeeze_scores, min_blank_frame=self.min_blank_frame)
            head_redundant = min(self.min_blank_frame, self.tail_blank_frame)
            squeezed_scores = squeezed_scores[head_redundant:]
            part_original_index = list(np.array(part_original_index)[head_redundant:]-self.tail_blank_frame)
            assert squeezed_scores.shape[0] == len(part_original_index)

            if len(part_original_index) < 1:
                return 0 
            
            # store the number of zero frames at the tail of frames 
            for i, frame in enumerate(part_scores):
                if np.sum(part_scores[i]) != 0:
                    self.tail_blank_frame = 0 
                else:
                    self.tail_blank_frame += 1
            
        # store squeezed scores
        self.squeezed_scores = np.concatenate([self.squeezed_scores, squeezed_scores], axis=0) if self.squeezed_scores is not None else squeezed_scores 
        
        # update reg origin index
        for index in part_original_index:
            self.squeezed_original_index.append(full_scores_length + index)
        return squeezed_scores.shape[0]
    
    def reset(self):
        self.full_scores = None 
        self.squeezed_scores = None 
        self.squeezed_original_index = list()
        self.tail_blank_frame = 0


if __name__ == '__main__':
    min_blank_frame = 2
    pianoroll = np.asarray([0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0])
    print(pianoroll)
    squeeze_res = squeeze_pianoroll(pianoroll, min_blank_frame=min_blank_frame)
    print(squeeze_res[0])
    print(squeeze_res[1])

    pool = ScorePool(squeeze=True, min_blank_frame=min_blank_frame)
    for item in pianoroll:
        pool.add_scores(np.asarray([item]))
    print(pool.full_scores)
    print(pool.squeezed_scores)
    print(pool.squeezed_original_index)
