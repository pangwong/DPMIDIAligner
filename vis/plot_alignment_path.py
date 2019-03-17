"""
Python 3.x is required.

Usage:
    python plot_original_alignment_path.py -f $(path of .npz file you download from web) 

Tips:
    You can click buttons on the bottom left of figure to Zoom in and Zoom out.
"""

import os
import argparse
import collections
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as mcm

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
    unsqueezed_alignment_path = meta_data['unsqueezed_alignment_path']

    max_float = np.finfo(np.float32).max
    cost_matrix[cost_matrix==max_float] = 0
    path_cost_matrix[path_cost_matrix==max_float] = 0
        
    fig, ax = plt.subplots()
    
    # mark active frame as gray and the more notes being active, the more grey the frame in matrix is    
    mw, mh = reg_score.shape[0], ref_score.shape[0] 
    marker_matrix = np.zeros((mh, mw))
    for i in range(mw):
        activate_rate = np.sum(reg_score[i]) / 88.0
        marker_matrix[:, i] += activate_rate
    for i in range(mh):
        activate_rate = np.sum(ref_score[i]) / 88.0
        marker_matrix[i, :] += activate_rate
    plt.imshow(marker_matrix, origin='lower', cmap='Greys', interpolation='None', alpha=0.6)
    
    # draw alignment path
    path_x = [p[0] for p in unsqueezed_alignment_path]
    path_y = [p[1] for p in unsqueezed_alignment_path]
    plt.plot(path_x, path_y, alpha=0.2, color='y')
    costs = [cost_matrix[j,i] for (i, j) in squeezed_alignment_path]
    cmap = mcm.get_cmap('cool')
    max_cost = max(costs) if args.r else 4
    for index, pt in enumerate(unsqueezed_alignment_path):
        cost = costs[index]
        radius = min((cost/max_cost+1) * 0.3, 0.5) * (mw/1000+1) if args.r else  0.3*(mw/1000+1)
        color = cmap(cost/(max_cost*0.5))
        alpha = 1 if marker_matrix[pt[1], pt[0]]>0 else 0
        p = mpatches.Circle(pt, radius, color=color, alpha=alpha)
        ax.add_patch(p)
    ax.plot()
    
    
    # annotation 
    annot_matrix = np.zeros((mh, mw))
    for x, y, cost in zip(path_x, path_y, costs):
        annot_matrix[y, x] = cost+0.01
    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    def update_annot(pos_in_matrix, pos_in_time, cost):
        annot.xy = pos_in_matrix 
        text = "reg time:{}s, ref time:{}s\nframe euclidean dist:{}".format(pos_in_time[0], pos_in_time[1], cost-0.01) 
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)
    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            pos_in_time = (int(event.xdata/31.25*100)/100.0, int(event.ydata/31.25*100)/100.0)
            pos_in_matrix = (int(event.xdata+0.5), int(event.ydata+0.5))
            if pos_in_matrix[0] < mw and pos_in_matrix[1] < mh: 
                cost = annot_matrix[pos_in_matrix[1], pos_in_matrix[0]] 
                if cost > 0:
                    update_annot(pos_in_matrix, pos_in_time, cost)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()
            else:   
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
    fig.canvas.mpl_connect("motion_notify_event", hover)

    # axis setting
    def format_func(value, tick_number):
        return int(value*1000 / 31.25)/1000.0
    plt.xlabel('Time in reg MIDI (second)', fontsize=18)
    plt.ylabel('Time in ref MIDI (second)', fontsize=19)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    
    plt.show()



if __name__ == '__main__':
    main()
