import matplotlib.pyplot as plt
import numpy as np
import os
from collections import deque

LOG_PATH = "/root/files/coding/data_loading_planner/plot_source"
PLOT_PATH = "/root/files/coding/data_loading_planner/taobao_run"

def Read_Steps():
    steps = dict()

    stats_log = open(os.path.join(LOG_PATH, "search-log-level-2.txt"), "rt")
    for line in stats_log:
        nums = line.split()
        if int(nums[4][:-1]) not in steps:
            steps[int(nums[4][:-1])] = (deque(), deque(), deque())                                # cost, best_cost, worst_cost
        steps[int(nums[4][:-1])][0].append(int(nums[8][:-1]))
        steps[int(nums[4][:-1])][1].append(int(nums[10][:-1]))
        steps[int(nums[4][:-1])][2].append(int(nums[12][:-1]))
    stats_log.close()
    
    return steps

def Print_Step_Plot(steps: dict, start_step: int, end_step: int, step_interval: int):
    
    for step in range(start_step, end_step + 1, step_interval):
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.set_ylabel("cost")
        ax.set_xlabel("unused batches")
        sample_points_in_step = [i for i in range(len(steps[step][0]))]
        ax.scatter(sample_points_in_step, steps[step][0], 0.5,)                                         # costs
        ax.plot(sample_points_in_step, steps[step][1], 'g', linewidth=1.0)                              # best costs
        ax.plot(sample_points_in_step, steps[step][2], 'r', linewidth=1.0)                              # worst costs

        plt.savefig(os.path.join(PLOT_PATH, "step_" + str(step) + ".png"), bbox_inches='tight')

        plt.close()

if __name__ == "__main__":
    plt.style.use('_mpl-gallery')
    
    # # Print plots for steps
    steps = Read_Steps()
    Print_Step_Plot(steps, 350, 3000, 50)

