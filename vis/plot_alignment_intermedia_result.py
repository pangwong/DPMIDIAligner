"""
Python 3.x is required.

Usage:
    python plot_original_alignment_path.py -f $(path of .npz file you download from web) 

Tips:
    You can click buttons on the bottom left of figure to Zoom in and Zoom out.
"""


import os
import argparse
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.patches as mpatches

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default='')
parser.add_argument('-r', action='store_true', help='relative mode, making low cost and high cost more distinguishable')
args = parser.parse_args()

def main():
    meta_data = np.load(args.f)
    ref_score = meta_data['ref_score']
    reg_score = meta_data['reg_score']
    cost_matrix        = meta_data['cost_matrix'] 
    path_cost_matrix   = meta_data['path_cost'] 
    squeezed_alignment_path   = meta_data['squeezed_alignment_path']
    reg_index_to_original_index_mapping = meta_data['reg_squeeze_index_in_unsqueeze']
    ref_index_to_original_index_mapping = meta_data['ref_squeeze_index_in_unsqueeze']

    max_float = np.finfo(np.float32).max
    cost_matrix[cost_matrix==max_float] = 0
    path_cost_matrix[path_cost_matrix==max_float] = 0

    mh, mw = cost_matrix.shape 
    marker_matrix = np.zeros((mh, mw))
    for i in range(mw):
        activate_rate = np.sum(reg_score[reg_index_to_original_index_mapping[i]]) / 88.0
        marker_matrix[:, i] += activate_rate
    for i in range(mh):
        activate_rate = np.sum(ref_score[ref_index_to_original_index_mapping[i]]) / 88.0
        marker_matrix[i, :] += activate_rate

    #Share both X and Y axes with all subplots
    fig, axes = plt.subplots(1, 3, sharex='all', sharey='all')

    axes[0].set_title('cost matrix')
    axes[0].imshow(cost_matrix, origin='lower', cmap='Greys', interpolation='None', alpha=1)
    
    axes[1].set_title('cost matrix with squeezed alignment path')
    axes[1].imshow(cost_matrix, origin='lower', cmap='Greys', interpolation='None', alpha=1)
    path_x = [item[0] for item in squeezed_alignment_path]
    path_y = [item[1] for item in squeezed_alignment_path]
    axes[1].plot(path_x, path_y, alpha=0.2, color='y')
    costs = [cost_matrix[j,i] for (i, j) in squeezed_alignment_path]
    cmap = mcm.get_cmap('cool')
    max_cost = max(costs) if args.r else 4
    for index, pt in enumerate(squeezed_alignment_path):
        cost = costs[index]
        radius = min((cost/max_cost+1) * 0.3, 0.5) if args.r else  0.3
        color = cmap(cost/(max_cost*0.5))
        alpha = 1 if marker_matrix[pt[1], pt[0]] > 0 else 0 
        p = mpatches.Circle(pt, radius, color=color, alpha=alpha)
        axes[1].add_patch(p)
    axes[1].plot()
    
    axes[2].set_title('path cost in search area')
    axes[2].imshow(path_cost_matrix, origin='lower', cmap='Greys', interpolation='None', alpha=1)
    plt.show()
    

if __name__ == '__main__':
    main()
