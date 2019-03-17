from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import collections
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean, hamming, cityblock

from util.score_pool import ScorePool


class DPSequenceAligner():
    
    def __init__(self, min_blank_frame=1, dist_type='e', valid_range=1, jump_penalty=1, align_position=0, half_search_window=300, update_interval=10):
        self.dist_type = dist_type
        self.valid_range = valid_range
        self.jump_penalty = jump_penalty
        self.min_blank_frame = min_blank_frame
        self.align_position = align_position
        self.half_search_window = half_search_window
        self.update_align_pos_interval = update_interval
        self.specify_ref_score = False
        assert self.update_align_pos_interval > 1
        

    def set_ref_scores(self, reference_score):
        '''
            add reference pianoroll, reference scores should be a 2-D ndarray 
        '''
        ## variables initialize
        # reg and ref pool
        self.reg_pool = ScorePool(squeeze=True, min_blank_frame=self.min_blank_frame) 
        self.ref_pool = ScorePool(squeeze=True, min_blank_frame=self.min_blank_frame)
        # ref score status
        self.specify_ref_score = False
        # get alignment_path
        self.alignment_path = None
        self.alignment_path_in_original_sequence = None
        # other
        self.max_cost = 0
        self.zeros_rows = list()
        self.empty_rows = collections.defaultdict(int) 
        self.empty_thres = 200
        self.align_pos_points = [(0,0)]
        self.cand_align_pos_points = [(0,0)]

        # squeeze scores
        self.ref_pool.add_scores(reference_score)
        self.ref_size = self.ref_pool.squeezed_scores.shape[0]

        # cost matrix and sum path costs
        self.estimated_reg_size = self.ref_size
        self.max_float = np.finfo(np.float32).max
        self.cost_matrix = np.full((self.ref_size, self.estimated_reg_size), self.max_float, dtype=np.float32)
        self.path_cost = np.full((self.ref_size, self.estimated_reg_size), self.max_float, dtype=np.float32)
        self.left_rows = np.full((self.ref_size, self.estimated_reg_size), -1, dtype=np.int32)
        self.search_window = (max(self.align_position-self.half_search_window,0), min(self.align_position+self.half_search_window, self.ref_size))
        #self.search_window = [0, self.ref_size/2]

        self.specify_ref_score = True

    def add_reg_scores(self, part_scores):
        '''
            add recognized pianoroll, part_scores should be a 2-D ndarray 
        '''
        if not self.specify_ref_score:
            raise Exception('No Reference Score was specified')
        
        sequence_length = self.reg_pool.add_scores(part_scores)   
        end_index = self.reg_pool.squeezed_scores.shape[0]
        start_index = end_index - sequence_length
        
        # enlarge self.cost_matrix and so on
        if end_index > self.estimated_reg_size:
            inc_size = max(50, end_index-self.estimated_reg_size)
            self.cost_matrix = np.concatenate([self.cost_matrix, np.full((self.ref_size, inc_size), self.max_float, dtype=np.float32)], axis=1)
            self.path_cost   = np.concatenate([self.path_cost,   np.full((self.ref_size, inc_size), self.max_float, dtype=np.float32)], axis=1)
            self.left_rows   = np.concatenate([self.left_rows,   np.full((self.ref_size, inc_size), -1, dtype=np.int32)], axis=1)
            self.estimated_reg_size += inc_size 

        ref_score = self.ref_pool.squeezed_scores
        reg_score = self.reg_pool.squeezed_scores
        for col in range(start_index, end_index):
            print('#{}'.format(col))
            # update align position and search window
            if (col+1) % self.update_align_pos_interval == 0:
                latest_path_cost = self.path_cost[self.search_window[0]:self.search_window[1], col-1]
                min_cost_index = list(np.arange(self.search_window[0], self.search_window[1])[latest_path_cost==np.min(latest_path_cost)])
                self.cand_align_pos_points.extend(map(lambda x:(col-1, x, self.path_cost[x, col-1]), min_cost_index))
                #print('col:{}, {}'.format(col, map(lambda x: self.path_cost[x][col-1], min_cost_index)))
                #print(self.cand_align_pos_points)
                if (len(min_cost_index)//2):
                    min_cost_index += [self.align_position] * (len(min_cost_index)//3)
                self.align_position = int(sum(min_cost_index) / float(len(min_cost_index)))
                self.search_window = (max(self.align_position-self.half_search_window,0), min(self.align_position+self.half_search_window, self.ref_size))
                self.align_pos_points.append((col-1, self.align_position))
            
            # calculate cost matrix
            rows = list()
            for row in range(self.ref_size):
            #for row in range(self.search_window[0], self.search_window[1]):
                cost = self.dist_fun(ref_score[row], reg_score[col], self.dist_type)
                self.cost_matrix[row, col] = cost 
                if cost > self.max_cost:
                    self.max_cost = cost
                if cost < 1e-6:
                    rows.append(row)
            
            if col == 0:
                self.path_cost[:, col] = self.cost_matrix[:, col]
                continue
        
            # filter those cols in cost matrix that most of elements are near 0
            #print("{}/{}".format(len(rows), self.ref_size))
            self.empty_rows[len(rows)] += 1
            if (col+1) % 5 == 0:
                sorted_list = sorted(self.empty_rows.items(), key=lambda x: x[1], reverse=True)
                #print(sorted_list)
                self.empty_thres = max(sorted_list[0][0] - 1, 100)
            if len(rows) > self.empty_thres:
                rows = list() 

            # dynamic programming
            for row in range(self.search_window[0], self.search_window[1]):
                near_rows = list(range(max(row-1-self.valid_range, 0), min(row-1+self.valid_range+1, self.ref_size)))
                for left_row in set(rows + near_rows):
                    curr_path_cost = self.path_cost[left_row][col-1] + self.cost_fun((col-1, left_row), (col, row))
                    if curr_path_cost < self.path_cost[row][col]:
                        self.path_cost[row][col] = curr_path_cost
                        self.left_rows[row][col] = left_row
                
    @staticmethod 
    def dist_fun(ref_frame, reg_frame, dist_type='e'):
        if dist_type == 'e':
            dist = euclidean(reg_frame, ref_frame) 
        elif dist_type == 'h':
            dist = hamming(reg_frame, ref_frame) 
        elif dist_type == 'm':
            dist = cityblock(reg_frame, ref_frame) 
        else:
            raise NotImplementedError
        return dist

    def cost_fun(self, left_pt, curr_pt):
        (x1, y1), (x2, y2)  = left_pt, curr_pt
        slope = y2 - y1
        
        # not jump
        if abs(slope-1) <= self.valid_range:
            path_cost = (1+abs(slope-1)) * self.cost_matrix[y2, x2]
        # jump
        else:
            path_cost = self.max_cost * self.jump_penalty * (abs(slope)/20.0+1)**0.5 
            #path_cost = self.max_cost * self.jump_penalty 

        return path_cost 

    def get_alignment_path(self):
        '''
        traverse forward DTW cost matrix to get path
        '''
        if self.alignment_path is None:
            self.alignment_path, self.original_alignment_path = list(), list()
            min_row = 0
            reg_size = self.reg_pool.squeezed_scores.shape[0]
            for row in range(self.ref_size):
                if self.path_cost[row][reg_size-1] < self.path_cost[min_row][reg_size-1]:
                    min_row = row
            curr_row = min_row 
            for i in range(reg_size-1, 0, -1):
                self.alignment_path.append((i, curr_row)) 
                left_row = self.left_rows[curr_row][i] 
                curr_row = left_row
            self.alignment_path.append((0, left_row))
            self.alignment_path.reverse()

            reg_original_index = self.reg_pool.squeezed_original_index
            ref_original_index = self.ref_pool.squeezed_original_index
            for point in self.alignment_path:
                original_point = (reg_original_index[point[0]], ref_original_index[point[1]])
                self.original_alignment_path.append(original_point)
        return self.alignment_path, self.original_alignment_path

    def get_cost_matrix(self):
        reg_size = self.reg_pool.squeezed_scores.shape[0]
        return self.cost_matrix[:, :reg_size]

    def get_path_cost_matrix(self):
        reg_size = self.reg_pool.squeezed_scores.shape[0]
        return self.path_cost[:, :reg_size]
    
    def save_intermedia_result(self, npz_path):
        cost_matrix = self.get_cost_matrix()
        path_cost = self.get_path_cost_matrix() 
        self.get_alignment_path()
        np.savez(npz_path, 
                    cost_matrix=cost_matrix, 
                    path_cost=path_cost,
                    squeezed_alignment_path=self.alignment_path,
                    unsqueezed_alignment_path=self.original_alignment_path,
                    ref_score=self.ref_pool.full_scores,
                    reg_score=self.reg_pool.full_scores,
                    reg_squeeze_index_in_unsqueeze=self.reg_pool.squeezed_original_index,
                    ref_squeeze_index_in_unsqueeze=self.ref_pool.squeezed_original_index,
                    squeezed_ref_score=self.ref_pool.squeezed_scores,
                    squeezed_reg_score=self.reg_pool.squeezed_scores)

    def display_cost_matrix(self):
        self.get_alignment_path()
        plt.subplot(1,3,1)   
        self.cost_matrix[self.cost_matrix==self.max_float] = 0
        self.path_cost[self.path_cost==self.max_float] = 0
        plt.imshow(self.cost_matrix, origin='lower', cmap='Greys', interpolation='None', alpha=1)
        plt.subplot(1,3,2)
        plt.imshow(self.cost_matrix, origin='lower', cmap='Greys', interpolation='None', alpha=1)
        path_x = [item[0] for item in self.alignment_path]
        path_y = [item[1] for item in self.alignment_path]
        plt.scatter(path_x, path_y, c='r', alpha=0.5) 
        plt.subplot(1,3,3)
        plt.imshow(self.path_cost, origin='lower', cmap='Greys', interpolation='None', alpha=1)
        path_x = [item[0] for item in self.align_pos_points]
        path_y = [item[1] for item in self.align_pos_points]
        plt.scatter(path_x, path_y, c='g', alpha=0.5) 
        path_x = [item[0] for item in self.cand_align_pos_points]
        path_y = [item[1] for item in self.cand_align_pos_points]
        plt.scatter(path_x, path_y, c='r', alpha=0.5) 
        plt.show()
