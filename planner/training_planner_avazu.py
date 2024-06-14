import pandas as df
import numpy as np
from typing import Optional
import os
import gc
from collections import deque
import random
from math import ceil
import statistics
import time
from multiprocessing import Process, Queue, get_context
import heapq
import json

LARGE_NUMBER = 10000000
# CACHE_RATIO = 0.50
DATASET = "avazu"
BATCH_FILE_SUFFIX = None #"-1024"
LOG_PATH = "/root/files/coding/data_loading_planner/avazu_run_1"
PLAN_FILE_NAME = "" # "-LFU-15-1024-5"
AVAZU_CACHE_SIZE = int(40428967 * 0.05)
AVAZU_DATA_PATH = "/root/files/coding/RecSys-Training-Planner/DLRM/input/avazu/"
AVAZU_PLAN_PATH = "/root/files/coding/RecSys-Training-Planner/DLRM/input/avazu/training_plan/"

class Planner(object):
    def __init__(
            self, 
            dataset: str,
            plan_path: str, 
            data_path: str, 
            log_path: str, 
            cached_rows: int, 
            warm_up_steps: int = 20, 
            search_ratio: float = 0.3, 
            search_limit: int = None,
            hotness_diff_threshold_update_period: int = 10,
            hotness_diff_threshold_base_relax_ratio: float = 0.95,
            hotness_diff_threshold_increment_relax_ratio: float = 0.001,
            hotness_diff_threshold_relax_ratio_penalty_rate: float = 0.5,
            hotness_diff_threshold_init_value: int = 0,
            hotness_diff_threshold_late_time_cap: float = 1,
            hotness_diff_threshold_startup_cap: int = 0,
            hotness_diff_threshold_recal_steps: int = 0,
            batch_file: str = None
            ) -> None:
        '''
        args:
            dataset (str): Should be "criteo" or "taobao".
            plan_path (str): Directory to read batches from and write generated plan to.
            data_path (str): Path to the dataset file (a numpy xyz file).
            log_path (str): A directory to store planning logs.
            cached_rows (int): Size of the simulated GPU cache (number of rows).
            warm_up_steps (int): Steps in the warm up phase. (for the heuristic searching algorithm)
            search_ratio (float): If search_limit is None, this decides how many possible choices we search. (for the heuristic searching algorithm)
            search_limit (int): Possible choices we search in each searching step. This serves as an efficiency guarantee. (for the heuristic searching algorithm)
            hotness_diff_threshold_update_period (int): Every number of steps after which we update the hotness difference threshold. (for the heuristic searching algorithm)
            hotness_diff_threshold_base_relax_ratio (float): How much variation do we allow about the threshold. (for the heuristic searching algorithm)
            hotness_diff_threshold_increment_relax_ratio (float): Aggregated increment added to the base relax_ratio while threshold hit. (for the heuristic searching algorithm)
            hotness_diff_threshold_relax_ratio_penalty_rate (float): Decrease the aggregated increment by this ratio. (for the heuristic searching algorithm)
            hotness_diff_threshold_init_value (int): Initilized value for hotness_diff_threshold. (for the heuristic searching algorithm)
            hotness_diff_threshold_startup_cap (int): The hotness_diff threshold should only be effective after searching such many choices. (for the heuristic searching algorithm)
            hotness_diff_threshold_recal_steps (int): After searching such many steps, recalibrate the threshold.
        '''
        # Read the batches.
        if batch_file is None:
            batch_file = ""
        loading_begin = time.time()
        print("Loading dataset " + dataset + "...")
        batches = df.read_parquet(os.path.join(os.path.abspath(plan_path), "batches" + batch_file + ".parquet")).transpose()             # each column is a batch of indices of dataset's row
        self.batch_num = batches.shape[1]
        self.batch_size = batches.shape[0]

        # Read dataset and process it.
        if dataset == "avazu":
            self.load_avazu_data(data_path, batches)
        else:
            exit("wrong dataset")
        loading_end = time.time()
        print("Data loading finished. (" + str(loading_end - loading_begin) + "s)")

        # Misc
        self.cached_rows = cached_rows
        self.log_path = os.path.abspath(log_path)
        self.plan_path = os.path.abspath(plan_path)
        self.warm_up_steps = min(warm_up_steps, self.batch_num)
        if search_limit is None:
            self.search_limit = self.batch_num * search_ratio
        else:
            self.search_limit = search_limit
        self.hotness_diff_threshold_update_period = hotness_diff_threshold_update_period
        if hotness_diff_threshold_init_value > 0:
            self.hotness_diff_threshold_init_value = hotness_diff_threshold_init_value
        else:
            self.hotness_diff_threshold_init_value = LARGE_NUMBER
        self.hotness_diff_threshold_base_relax_ratio = hotness_diff_threshold_base_relax_ratio
        self.hotness_diff_threshold_increment_relax_ratio = hotness_diff_threshold_increment_relax_ratio
        self.hotness_diff_threshold_relax_ratio_penalty_rate = hotness_diff_threshold_relax_ratio_penalty_rate
        self.hotness_diff_threshold_late_time_cap = hotness_diff_threshold_late_time_cap
        self.hotness_diff_threshold_startup_cap = hotness_diff_threshold_startup_cap
        if hotness_diff_threshold_recal_steps > 0:
            self.hotness_diff_threshold_recal_steps = hotness_diff_threshold_recal_steps
        else:
            self.hotness_diff_threshold_recal_steps = LARGE_NUMBER
        
    def load_avazu_data(self, data_path, batches):
        """
        Returned data:
            self.batches (list of df.Series): Categorical objects contained in each batch. (This is basically all we need from the dataset.) 
        """
        # Read cat_data
        data = np.load(os.path.join(os.path.abspath(data_path), "train_processed.npz"))
        X_cat = data["X_cat"]                                                                                               # categorical feature
        # cat_data = df.DataFrame(X_cat).astype(int)                                                                          # each column is a row of categorical features in the original dataset
        
        # Count size of features.
        count_data = np.load(os.path.join(os.path.abspath(data_path), "train_fea_count.npz"))
        self.cat_counts = list(count_data["counts"])

        # Convert id from id_in_cat to unique_id_in_dataset
        rotated_cat_data = np.swapaxes(X_cat, 0, 1)
        cat_offsets = list()
        tmp_offset = 0
        for i in range(len(self.cat_counts)):
            cat_offsets.append(tmp_offset)
            tmp_offset = tmp_offset + self.cat_counts[i]
        for i in range(len(cat_offsets)):
            rotated_cat_data[i] = rotated_cat_data[i] + cat_offsets[i]
        new_X_cat = np.swapaxes(rotated_cat_data, 0, 1)
        
        # Calculate objs in batch
        self.batches = list()
        self.freq_batches = list()
        for i in range(batches.shape[1] - 1):
            indices = batches[i].to_numpy()
            counted_batch = df.concat([df.Series(new_X_cat[indices].flatten())]).value_counts()
            indices_batch = df.Series(counted_batch.index)
            freq_batch = counted_batch.reset_index(drop=True)
            self.batches.append(indices_batch)
            self.freq_batches.append(freq_batch)
        # last batch contains -1, should remove them first.
        indices = np.array([i for i in batches[self.batch_num - 1].values.tolist() if i != -1])
        counted_batch = df.concat([df.Series(new_X_cat[indices].flatten())]).value_counts()
        indices_batch = df.Series(counted_batch.index, dtype='int32')
        freq_batch = counted_batch.reset_index(drop=True)
        self.batches.append(indices_batch)
        self.freq_batches.append(freq_batch)

    def Read_plan(self, plan_file: str = None):
        if plan_file is None:
            plan_file = ""
        full_path = os.path.join(self.plan_path, (plan_file + ".parquet"))

        print("Loading training plan from " + str(full_path))
        start_time = time.time()
        data = df.read_parquet(full_path)
        self.plan = data['plan'].to_list()
        end_time = time.time()
        print("Training plan loaded. (" + str(end_time - start_time) + "s)")

    def to_parquet(self, mark: str = None) -> None:
        if mark is None:
            mark = ""
        df.DataFrame({'plan': self.plan}).to_parquet(os.path.join(self.plan_path, ("training_plan" + mark +".parquet")))

def nlargest(data: np.ndarray, n: int, reverse: bool = False) -> np.ndarray:
    '''
    Return value: a ndarray of indices.
    Assume data >= 0.
    '''
    if reverse:
        n = data.shape[0] - n
    
    if len(data) < n:
        print("len(data) < n: len(data) = " + str(len(data)) + ", n = " + str(n))
    if len(data) == n:
        if not reverse:
            return np.arange(len(data), dtype=np.int32)
        else:
            return np.array([], dtype=np.int32)

    result = np.zeros(n, dtype=np.int32)

    # # Deal with 0 first
    # idx_nonzero = np.flatnonzero(data)
    # num_nonzero = idx_nonzero.shape[0]
    # if num_nonzero < n:
    #     idx_zero = np.flatnonzero(data == 0)
        
    #     result[:num_nonzero] = idx_nonzero
    #     result[num_nonzero:n] = idx_zero[n - num_nonzero]

    #     if not reverse:
    #         return result
    #     else:
    #         tmp = np.ones(data.shape[0], dtype=np.bool8)
    #         tmp[result] = False
    #         return np.flatnonzero(tmp)

    # if num_nonzero == n:
    #     if not reverse:
    #         return idx_nonzero
    #     else:
    #         tmp = np.ones(data.shape[0], dtype=np.bool8)
    #         tmp[idx_nonzero] = False
    #         return np.flatnonzero(tmp)

    # Search from nonzero data
    rng = np.random.default_rng()

    rest_n = n
    rest_data = data
    rest_idx = np.arange(len(data), dtype=np.int32)

    while True:
        pivot_idx = rng.integers(0, len(rest_data))
        pivot = rest_data[pivot_idx]
        idx_larger_group = np.flatnonzero(rest_data > pivot)
        idx_smaller_group = np.flatnonzero(rest_data < pivot)
        idx_equal_group = np.flatnonzero(rest_data == pivot)

        # |____smaller_____|____equal_____|_____larger_____|      Where is rest_n on this axis?
        # Case 1: rest_n falls in the larger part.
        if rest_n < len(idx_larger_group):
            rest_data = rest_data[idx_larger_group]
            rest_idx = rest_idx[idx_larger_group]
        # Case 2: rest_n falls in the equal part.
        elif rest_n < len(idx_larger_group) + len(idx_equal_group):
            result[n - rest_n:n - rest_n + len(idx_larger_group)] = rest_idx[idx_larger_group]
            rest_n -= len(idx_larger_group)

            result[n - rest_n:] = rest_idx[idx_equal_group[:rest_n]]
            rest_n = 0

            break
        # Case 3: rest_n falls in the smaller part.
        else:
            result[n - rest_n:n - rest_n + len(idx_larger_group)] = rest_idx[idx_larger_group]
            rest_n -= len(idx_larger_group)

            result[n - rest_n:n - rest_n + len(idx_equal_group)] = rest_idx[idx_equal_group]
            rest_n -= len(idx_equal_group)

            rest_data = rest_data[idx_smaller_group]
            rest_idx = rest_idx[idx_smaller_group]
    
    # Output (recover the original problem)
    if not reverse:
        return result
    else:
        tmp = np.ones(data.shape[0], dtype=np.bool8)
        tmp[result] = False
        return np.flatnonzero(tmp)

def New_Simulate_Cost(plan: list, cached_rows: int, batches_id: list, batches_freq: list, init_cache_state: np.ndarray = None, start_step: int = 0, start_cost: int = 0, process_id: int = None, should_print: bool = False):
    '''
    Calculate cost of a given route.
    '''
    num_batches = len(batches_id)

    if process_id is not None:
        process_info = "[Process " + str(process_id) + "] "
    else:
        process_info = ""

    num_steps = len(plan)
    if isinstance(init_cache_state, np.ndarray):
        gpu_cache = init_cache_state
        cost_total = start_cost
        num_available_rows = cached_rows - len(gpu_cache)
    else:
        gpu_cache = np.array([-1] * cached_rows, dtype=np.int32)
        cost_total = 0
        start_step = 0
        num_available_rows = cached_rows
    
    freq_register = np.array([-1] * cached_rows, dtype=np.int32)
    
    # the main loop
    for step in range(start_step, num_steps):
        # Print progress
        if should_print and ((step - start_step) % (int((num_steps - start_step) / 10) + 1)) == 0:
            print(process_info + str(int((step - start_step) / ((num_steps - start_step) / 10))) + "0%")
        
        # Get ids in current batch
        batch_id = batches_id[plan[step]]
        batch_freq = batches_freq[plan[step]]

        # Find ids that need to be transfered to cache
        mask_ids_to_comm = np.isin(batch_id, gpu_cache, assume_unique=True, invert=True)
        ids_to_comm = batch_id[mask_ids_to_comm]
        freq_ids_to_comm = batch_freq[mask_ids_to_comm]
        num_ids_to_comm = len(ids_to_comm)                                                                              # Cost 1: cost of moving data in
        num_rows_to_evic = num_ids_to_comm - num_available_rows
        num_rows_to_evic = num_rows_to_evic if num_rows_to_evic > 0 else 0                                              # Cost 2: cost of moving data out

        # Evic first num_rows_to_evic rows of ids, since the gpu_cache is sorted by batch flags.
        if num_rows_to_evic > 0:
            mask_evictable_row = np.isin(gpu_cache, batch_id, assume_unique=True, invert=True)
            idx_nonempty_row = np.flatnonzero(mask_evictable_row)[gpu_cache[mask_evictable_row] != -1]
            idx_idx_freq_evictable_row = nlargest(freq_register[idx_nonempty_row], num_rows_to_evic, reverse=True)
            idx_freq_evictable_row = idx_nonempty_row[idx_idx_freq_evictable_row]
            gpu_cache[idx_freq_evictable_row] = -1
            freq_register[idx_freq_evictable_row] = 0
        
        # Update gpu cache
        idx_update = np.flatnonzero(gpu_cache == -1)[:num_ids_to_comm]
        gpu_cache[idx_update] = ids_to_comm
        freq_register[idx_update] += freq_ids_to_comm
        
        num_available_rows = num_available_rows + num_rows_to_evic - num_ids_to_comm
        cost_total = cost_total + num_ids_to_comm + num_rows_to_evic

    return cost_total, gpu_cache, freq_register

def New_Heuristic_Search(
        log_path: str, 
        cached_rows: int, 
        batches_id: list, 
        batches_freq: list, 
        warm_up_steps:int = 0, 
        init_plan: list = None, 
        init_cache_state: np.ndarray = None, 
        init_cost: int = None, 
        process_id: int = None, 
        init_freq_status: np.ndarray  = None, 
        search_limit: int = 2000,
        hotness_diff_threshold_base_relax_ratio: float = 0.8,
        hotness_diff_threshold_update_window: int = 10,
        hotness_diff_threshold_startup_cap: int = 10,
        hotness_diff_threshold_increment_relax_ratio: float = 0.001,
        hotness_diff_threshold_late_time_cap: float = 1,
        hotness_diff_threshold_relax_ratio_penalty_rate: float = 0.8,
        ) -> tuple:
    num_batches = len(batches_id)
    
    if process_id is not None:
        process_info = "[Process " + str(process_id) + "] "
        log_file_suffix = "-" + str(process_id)
    else:
        process_info = ""
        log_file_suffix = ""
    log_file_path = os.path.join(log_path, ("search-log" + log_file_suffix + ".txt"))
    search_log = open(log_file_path, "w")
    # search_log_l2 = open(os.path.join(log_path, "search-log-level-2" + log_file_suffix + ".txt"), "w")

    if isinstance(init_plan, list):
        plan = init_plan
    else:
        plan = list()
    
    unused_batch = [i for i in range(num_batches) if i not in plan]
    random.shuffle(unused_batch)

    # The warm up phase (randomly fill in some steps since the first few steps don't really matter that much.)
    time_warmup_start = time.time()
    if len(plan) < warm_up_steps:
        fill_in_length = warm_up_steps - len(plan)
        plan = plan + unused_batch[ : fill_in_length]
        del unused_batch[ : fill_in_length]
    if init_cache_state is not None and init_cost is not None and init_freq_status is not None:
        cost_total = init_cost
        cache_state = init_cache_state
        freq_state = init_freq_status
    else:
        cost_total, cache_state, freq_state = New_Simulate_Cost(plan, cached_rows, batches_id, batches_freq)
    
    time_warmup_finished = time.time()

    # Set up the hotness difference threshold.
    hotness_diff_threshold = LARGE_NUMBER
    hotness_diff_history = [hotness_diff_threshold] * hotness_diff_threshold_update_window
    hotness_diff_history_idx = 0
    hotness_diff_threshold_dynamic_ratio = hotness_diff_threshold_base_relax_ratio
    hotness_diff_threshold_ratio_increment = 0

    # output_str = "[Warm up phase] cost: " + str(cost_total) + ", cache_usage: " + str(len(cache_state) / cached_rows) + " warmup time: " + str(time_warmup_finished - time_warmup_start) + "\n[Startup plan] " + str(plan) + "\n[Start searching] hotness_diff threshold = " + str(hotness_diff_threshold)
    # print(output_str)
    # search_log.write(output_str + "\n")

    # Searching
    time_last_step = time.time()
    start_step = len(plan)
    for step in range(start_step, num_batches):
        # Print progress
        if ((step - start_step) % (int((num_batches - start_step) / 100) + 1)) == 0:
            print(process_info + str(int((step - start_step) / ((num_batches - start_step) / 100))) + "%")
        
        cost_best_choice = LARGE_NUMBER
        # cache_state_best_choice = None
        best_choice = 0
        cost_worst_choice = 0
        num_available_rows = np.count_nonzero(cache_state == -1)

        # Search at most self.search_limit steps to find a choice
        for choice_idx in range(len(unused_batch)):
            # Just calculate cost, don't update cache.
            batch_id = batches_id[unused_batch[choice_idx]]
            mask_ids_to_comm = np.isin(batch_id, cache_state, assume_unique=True, invert=True)
            ids_to_comm = batch_id[mask_ids_to_comm]
            num_ids_to_comm = len(ids_to_comm)                                                                              # Cost 1: cost of moving data in
            num_rows_to_evic = num_ids_to_comm - num_available_rows
            num_rows_to_evic = num_rows_to_evic if num_rows_to_evic > 0 else 0                                              # Cost 2: cost of moving data out
            cost_tmp = num_ids_to_comm + num_rows_to_evic
            
            # Record a better/worse choice
            if cost_tmp < cost_best_choice:
                cost_best_choice = cost_tmp
                best_choice = choice_idx
            if cost_tmp > cost_worst_choice:
                cost_worst_choice = cost_tmp
            
            # Early stop conditions
            hotness_diff_tmp = cost_worst_choice - cost_best_choice
            if choice_idx > hotness_diff_threshold_startup_cap and hotness_diff_tmp > hotness_diff_threshold:
                hotness_diff_threshold_ratio_increment = hotness_diff_threshold_ratio_increment + hotness_diff_threshold_increment_relax_ratio
                hotness_diff_threshold_dynamic_ratio = hotness_diff_threshold_base_relax_ratio + hotness_diff_threshold_ratio_increment
                break

            # Late stop conditions
            if choice_idx > search_limit or choice_idx == (len(unused_batch) - 1):
                break

            # Level 2 logging
            # output_str = "[Choice " + str(choice_idx) +" in Step " + str(step) + "] choice: " + str(unused_batch[choice_idx]) + ", cost: " + str(cost_tmp) + ", best_cost: " + str(cost_best_choice) + ", worst_cost: " + str(cost_worst_choice) + ", hotness_diff: " + str(hotness_diff_tmp)
            # search_log_l2.write(output_str + "\n")
        
        # Decide current step and actually update cache state.
        choice = unused_batch.pop(best_choice)
        plan.append(choice)
        # cache_state = cache_state_best_choice
        cost_total = cost_total + cost_best_choice
        random.shuffle(unused_batch)

        # Update cache
        batch_id = batches_id[choice]
        batch_freq = batches_freq[choice]

        mask_ids_to_comm = np.isin(batch_id, cache_state, assume_unique=True, invert=True)
        ids_to_comm = batch_id[mask_ids_to_comm]
        num_ids_to_comm = len(ids_to_comm)
        if num_ids_to_comm > cached_rows:
            print("Even batch with the smallest cost exceeds the capacity of the cache.")
        freq_ids_to_comm = batch_freq[mask_ids_to_comm]
        num_rows_to_evic = num_ids_to_comm - num_available_rows

        if num_rows_to_evic > 0:
            mask_evictable_row = np.isin(cache_state, batch_id, assume_unique=True, invert=True)
            idx_nonempty_row = np.flatnonzero(mask_evictable_row)[cache_state[mask_evictable_row] != -1]
            idx_idx_freq_evictable_row = nlargest(freq_state[idx_nonempty_row], num_rows_to_evic, reverse=True)
            idx_freq_evictable_row = idx_nonempty_row[idx_idx_freq_evictable_row]
            # idx_freq_evictable_row = nlargest(freq_state[mask_evictable_row], num_rows_to_evic)                      # Very expensive!!!
            cache_state[idx_freq_evictable_row] = -1
            freq_state[idx_freq_evictable_row] = 0

        idx_update = np.flatnonzero(cache_state == -1)[:num_ids_to_comm]
        cache_state[idx_update] = ids_to_comm
        freq_state[idx_update] += freq_ids_to_comm

        # Update threshold
        hotness_diff_history[hotness_diff_history_idx] = cost_worst_choice - cost_best_choice
        hotness_diff_history_idx = (hotness_diff_history_idx + 1) % hotness_diff_threshold_update_window
        hotness_diff_mean = np.mean(hotness_diff_history)
        hotness_diff_threshold = hotness_diff_mean * hotness_diff_threshold_dynamic_ratio

        # Any step takes more than 0.3s should be considered as slightly late, thus no increment of ralax ratio
        step_time = time.time() - time_last_step
        time_last_step = time.time()
        if step_time > hotness_diff_threshold_late_time_cap:
            hotness_diff_threshold_ratio_increment = hotness_diff_threshold_ratio_increment * hotness_diff_threshold_relax_ratio_penalty_rate
            hotness_diff_threshold_dynamic_ratio = hotness_diff_threshold_base_relax_ratio + hotness_diff_threshold_ratio_increment

        # Logging
        
        output_str = "[Step " + str(step) + "] choice: " + str(choice) + ", cost: " + str(cost_best_choice) + ", hotness_diff: " + str(cost_worst_choice - cost_best_choice) + ", cache_usage(last run): " + str(1 - (num_available_rows/ cached_rows)) + ", step_time = " + str(step_time) + ", searched choices: " + str(choice_idx) + "\n             hotness_diff_history: " + str(hotness_diff_history) + "\n             mean hotness_diff: " + str(hotness_diff_mean) + ", ratio_increment: " + str(hotness_diff_threshold_ratio_increment) + ", dynamic threshold ratio: " + str(hotness_diff_threshold_dynamic_ratio) + ", new threshold: " + str(hotness_diff_threshold)
        # print(output_str)
        search_log.write(output_str + "\n")
    
    # output_str = "[cost: " + str(cost_total) + "] Training plan generated: " + str(plan)
    # print(output_str)
    # search_log.write(output_str + "\n")

    search_log.close()
    # search_log_l2.close()

    return plan, cost_total, cache_state, freq_state

def New_None_LFU_Heuristic_Search(
        log_path: str, 
        cached_rows: int, 
        batches_id: list,  
        warm_up_steps:int = 0, 
        init_plan: list = None, 
        init_cache_state: np.ndarray = None, 
        init_cost: int = None, 
        process_id: int = None, 
        search_limit: int = 10000,
        hotness_diff_threshold_base_relax_ratio: float = 0.8,
        hotness_diff_threshold_update_window: int = 10,
        hotness_diff_threshold_startup_cap: int = 10,
        hotness_diff_threshold_increment_relax_ratio: float = 0.001,
        hotness_diff_threshold_late_time_cap: float = 1,
        hotness_diff_threshold_relax_ratio_penalty_rate: float = 0.8,
        ) -> tuple:
    num_batches = len(batches_id)
    
    if process_id is not None:
        process_info = "[Process " + str(process_id) + "] "
        log_file_suffix = "-" + str(process_id)
    else:
        process_info = ""
        log_file_suffix = ""
    log_file_path = os.path.join(log_path, ("search-log" + log_file_suffix + ".txt"))
    search_log = open(log_file_path, "w")
    # search_log_l2 = open(os.path.join(log_path, "search-log-level-2" + log_file_suffix + ".txt"), "w")

    if isinstance(init_plan, list):
        plan = init_plan
    else:
        plan = list()
    
    unused_batch = [i for i in range(num_batches) if i not in plan]
    random.shuffle(unused_batch)

    # The warm up phase (randomly fill in some steps since the first few steps don't really matter that much.)
    time_warmup_start = time.time()
    if len(plan) < warm_up_steps:
        fill_in_length = warm_up_steps - len(plan)
        plan = plan + unused_batch[ : fill_in_length]
        del unused_batch[ : fill_in_length]
    if init_cache_state is not None and init_cost is not None:
        cost_total = init_cost
        cache_state = init_cache_state
    else:
        cost_total, cache_state = Non_LFU_Simulate_Cost(plan, cached_rows, batches_id)
    
    time_warmup_finished = time.time()

    # Set up the hotness difference threshold.
    hotness_diff_threshold = LARGE_NUMBER
    hotness_diff_history = [hotness_diff_threshold] * hotness_diff_threshold_update_window
    hotness_diff_history_idx = 0
    hotness_diff_threshold_dynamic_ratio = hotness_diff_threshold_base_relax_ratio
    hotness_diff_threshold_ratio_increment = 0

    # output_str = "[Warm up phase] cost: " + str(cost_total) + ", cache_usage: " + str(len(cache_state) / cached_rows) + " warmup time: " + str(time_warmup_finished - time_warmup_start) + "\n[Startup plan] " + str(plan) + "\n[Start searching] hotness_diff threshold = " + str(hotness_diff_threshold)
    # print(output_str)
    # search_log.write(output_str + "\n")

    # Searching
    time_last_step = time.time()
    start_step = len(plan)
    for step in range(start_step, num_batches):
        # Print progress
        if ((step - start_step) % (int((num_batches - start_step) / 100) + 1)) == 0:
            print(process_info + str(int((step - start_step) / ((num_batches - start_step) / 100))) + "%")
        
        cost_best_choice = LARGE_NUMBER
        # cache_state_best_choice = None
        best_choice = 0
        cost_worst_choice = 0
        num_available_rows = np.count_nonzero(cache_state == -1)

        # Search at most self.search_limit steps to find a choice
        for choice_idx in range(len(unused_batch)):
            # Just calculate cost, don't update cache.
            batch_id = batches_id[unused_batch[choice_idx]]
            mask_ids_to_comm = np.isin(batch_id, cache_state, assume_unique=True, invert=True)
            ids_to_comm = batch_id[mask_ids_to_comm]
            num_ids_to_comm = len(ids_to_comm)                                                                              # Cost 1: cost of moving data in
            num_rows_to_evic = num_ids_to_comm - num_available_rows
            num_rows_to_evic = num_rows_to_evic if num_rows_to_evic > 0 else 0                                              # Cost 2: cost of moving data out
            cost_tmp = num_ids_to_comm + num_rows_to_evic
            
            # Record a better/worse choice
            if cost_tmp < cost_best_choice:
                cost_best_choice = cost_tmp
                best_choice = choice_idx
            if cost_tmp > cost_worst_choice:
                cost_worst_choice = cost_tmp
            
            # Early stop conditions
            hotness_diff_tmp = cost_worst_choice - cost_best_choice
            if choice_idx > hotness_diff_threshold_startup_cap and hotness_diff_tmp > hotness_diff_threshold:
                hotness_diff_threshold_ratio_increment = hotness_diff_threshold_ratio_increment + hotness_diff_threshold_increment_relax_ratio
                hotness_diff_threshold_dynamic_ratio = hotness_diff_threshold_base_relax_ratio + hotness_diff_threshold_ratio_increment
                break

            # Late stop conditions
            if choice_idx > search_limit or choice_idx == (len(unused_batch) - 1):
                break

            # Level 2 logging
            # output_str = "[Choice " + str(choice_idx) +" in Step " + str(step) + "] choice: " + str(unused_batch[choice_idx]) + ", cost: " + str(cost_tmp) + ", best_cost: " + str(cost_best_choice) + ", worst_cost: " + str(cost_worst_choice) + ", hotness_diff: " + str(hotness_diff_tmp)
            # search_log_l2.write(output_str + "\n")
        
        # Decide current step and actually update cache state.
        choice = unused_batch.pop(best_choice)
        plan.append(choice)
        # cache_state = cache_state_best_choice
        cost_total = cost_total + cost_best_choice
        random.shuffle(unused_batch)

        # Update cache
        batch_id = batches_id[choice]

        mask_ids_to_comm = np.isin(batch_id, cache_state, assume_unique=True, invert=True)
        ids_to_comm = batch_id[mask_ids_to_comm]
        num_ids_to_comm = len(ids_to_comm)
        if num_ids_to_comm > cached_rows:
            print("Even batch with the smallest cost exceeds the capacity of the cache.")
        num_rows_to_evic = num_ids_to_comm - num_available_rows

        if num_rows_to_evic > 0:
            mask_evictable_row = np.isin(cache_state, batch_id, assume_unique=True, invert=True)
            idx_nonempty_row = np.flatnonzero(mask_evictable_row)[cache_state[mask_evictable_row] != -1]
            idx_evictable_row = idx_nonempty_row[:num_rows_to_evic]
            cache_state[idx_evictable_row] = -1

        idx_update = np.flatnonzero(cache_state == -1)[:num_ids_to_comm]
        cache_state[idx_update] = ids_to_comm

        # Update threshold
        hotness_diff_history[hotness_diff_history_idx] = cost_worst_choice - cost_best_choice
        hotness_diff_history_idx = (hotness_diff_history_idx + 1) % hotness_diff_threshold_update_window
        hotness_diff_mean = np.mean(hotness_diff_history)
        hotness_diff_threshold = hotness_diff_mean * hotness_diff_threshold_dynamic_ratio

        # Any step takes more than 0.3s should be considered as slightly late, thus no increment of ralax ratio
        step_time = time.time() - time_last_step
        time_last_step = time.time()
        if step_time > hotness_diff_threshold_late_time_cap:
            hotness_diff_threshold_ratio_increment = hotness_diff_threshold_ratio_increment * hotness_diff_threshold_relax_ratio_penalty_rate
            hotness_diff_threshold_dynamic_ratio = hotness_diff_threshold_base_relax_ratio + hotness_diff_threshold_ratio_increment

        # Logging
        
        output_str = "[Step " + str(step) + "] choice: " + str(choice) + ", cost: " + str(cost_best_choice) + ", hotness_diff: " + str(cost_worst_choice - cost_best_choice) + ", cache_usage(last run): " + str(1 - (num_available_rows/ cached_rows)) + ", step_time = " + str(step_time) + ", searched choices: " + str(choice_idx) + "\n             hotness_diff_history: " + str(hotness_diff_history) + "\n             mean hotness_diff: " + str(hotness_diff_mean) + ", ratio_increment: " + str(hotness_diff_threshold_ratio_increment) + ", dynamic threshold ratio: " + str(hotness_diff_threshold_dynamic_ratio) + ", new threshold: " + str(hotness_diff_threshold)
        # print(output_str)
        search_log.write(output_str + "\n")
    
    # output_str = "[cost: " + str(cost_total) + "] Training plan generated: " + str(plan)
    # print(output_str)
    # search_log.write(output_str + "\n")

    search_log.close()
    # search_log_l2.close()

    return plan, cost_total, cache_state

def Wrapper_Cost(q: Queue, process_id: int, should_print:bool, plan: list, cached_rows: int, batches_id: list, batches_freq: list):
    cost, _, _ = New_Simulate_Cost(plan, cached_rows, batches_id, batches_freq, process_id=process_id, should_print=should_print)
    q.put(cost)

def Wrapper_Search(
        q: Queue,
        log_path: str, 
        cached_rows: int, 
        batches_id: list, 
        batches_freq: list, 
        process_id: int,  
        search_limit: int,
        hotness_diff_threshold_base_relax_ratio: float,
        hotness_diff_threshold_update_window: int,
        hotness_diff_threshold_relax_ratio_penalty_rate: float,
        hotness_diff_threshold_increment_relax_ratio: float,
        hotness_diff_threshold_late_time_cap: float,
        hotness_diff_threshold_startup_cap: int,
        init_plan=None,
        init_cache_state=None,
        init_cost=None,
        init_freq_status=None,
        ):
    plan, cost, _, _ = New_Heuristic_Search(
        log_path=log_path, 
        cached_rows=cached_rows, 
        batches_id=batches_id, 
        batches_freq=batches_freq, 
        warm_up_steps=0, 
        init_plan=init_plan,
        init_cache_state=init_cache_state,
        init_cost=init_cost,
        process_id=process_id,
        init_freq_status=init_freq_status,
        search_limit=search_limit, 
        hotness_diff_threshold_base_relax_ratio=hotness_diff_threshold_base_relax_ratio, 
        hotness_diff_threshold_update_window=hotness_diff_threshold_update_window,
        hotness_diff_threshold_relax_ratio_penalty_rate=hotness_diff_threshold_relax_ratio_penalty_rate,
        hotness_diff_threshold_increment_relax_ratio=hotness_diff_threshold_increment_relax_ratio,
        hotness_diff_threshold_late_time_cap=hotness_diff_threshold_late_time_cap,
        hotness_diff_threshold_startup_cap=hotness_diff_threshold_startup_cap,
        )
    q.put(cost)
    q.put(plan)

def New_Multiprocess_Search( 
        chunk_size: int, 
        log_path: str, 
        cached_rows: int, 
        batches_id: list, 
        batches_freq: list, 
        search_limit: int = 2000,
        hotness_diff_threshold_base_relax_ratio: float = 0.8,
        hotness_diff_threshold_update_window: int = 10,
        hotness_diff_threshold_relax_ratio_penalty_rate: float = 0.8,
        hotness_diff_threshold_increment_relax_ratio: float = 0.001,
        hotness_diff_threshold_late_time_cap: float = 1,
        hotness_diff_threshold_startup_cap: int = 10,
        ):
    # Check chunk_size and make sure it is reasonable
    batch_num = len(batches_id)
    assert batch_num > chunk_size

    # Initialization. Do the first loop to handle num_warm_up_steps and init_plan.
    idx_unused_batches = list(range(batch_num))     # TODO: shuffle
    plan = list()
    processes = list()
    queues = list()
    list_batch_indices = list()
        
    chunk_count = 0

    # Segment unused batches (create new scene) and search
    while len(idx_unused_batches) > chunk_size:
        sub_batch_num = chunk_size
        sub_batches_id = list()
        sub_batches_freq = list()
        sub_batch_indices = list()

        for i in range(sub_batch_num):
            idx = idx_unused_batches.pop(0)
            sub_batch_indices.append(idx)
            sub_batches_id.append(batches_id[idx])
            sub_batches_freq.append(batches_freq[idx])
            
        queues.append(Queue())
        list_batch_indices.append(sub_batch_indices)
        processes.append(Process(target=Wrapper_Search, args=(
            queues[chunk_count], 
            log_path, 
            cached_rows, 
            sub_batches_id,
            sub_batches_freq,
            chunk_count,
            search_limit,
            hotness_diff_threshold_base_relax_ratio,
            hotness_diff_threshold_update_window,
            hotness_diff_threshold_relax_ratio_penalty_rate,
            hotness_diff_threshold_increment_relax_ratio,
            hotness_diff_threshold_late_time_cap,
            hotness_diff_threshold_startup_cap,
        )))
        processes[chunk_count].start()

        print("[Chunk " + str(chunk_count) + "] Issued.")
        chunk_count = chunk_count + 1
        
    # Last chunk: len(unused_batches) <= chunk_size
    start_time = time.time()

    sub_batch_num = len(idx_unused_batches)
    sub_batches_id = list()
    sub_batches_freq = list()
    sub_batch_indices = idx_unused_batches[:]

    for i in range(sub_batch_num):
        sub_batches_id.append(batches_id[i])
        sub_batches_freq.append(batches_freq[i])
        
    last_plan, last_cost, _, _ = New_Heuristic_Search(
        log_path, 
        cached_rows, 
        sub_batches_id, 
        sub_batches_freq, 
        warm_up_steps=0, 
        init_plan=None,
        init_cache_state=None,
        init_cost=None,
        process_id=chunk_count,
        init_freq_status=None,
        search_limit=search_limit, 
        hotness_diff_threshold_base_relax_ratio=hotness_diff_threshold_base_relax_ratio, 
        hotness_diff_threshold_update_window=hotness_diff_threshold_update_window,
        hotness_diff_threshold_relax_ratio_penalty_rate=hotness_diff_threshold_relax_ratio_penalty_rate,
        hotness_diff_threshold_increment_relax_ratio=hotness_diff_threshold_increment_relax_ratio,
        hotness_diff_threshold_late_time_cap=hotness_diff_threshold_late_time_cap,
        hotness_diff_threshold_startup_cap=hotness_diff_threshold_startup_cap,
        )
        
    end_time = time.time()
    print("[Chunk " + str(chunk_count) + " (the last chunk)] Searched. Cost = " + str(last_cost) + ". (time: " + str(end_time - start_time) + "s)")

    # Collect results
    cost = 0
    for i in range(len(processes)):
        processes[i].join()
        subcost = queues[i].get()
        subplan = queues[i].get()
        chunk_batch_indices = list_batch_indices[i]

        cost = cost + subcost
        for j in range(chunk_size):
            plan.append(chunk_batch_indices[subplan[j]])
            
        print("[Chunk " + str(i) + "] Collected. Subcost: " + str(subcost) + ", length of plan: " + str(len(subplan)) + ".")
        
    cost = cost + last_cost
    for i in range(sub_batch_num):
        plan.append(sub_batch_indices[last_plan[i]])
        
    print("[Chunk " + str(chunk_count) + "  (the last chunk)] Collected. Subcost: " + str(last_cost) + ", length of plan: " + str(len(last_plan)) + ".")

    return plan, cost

def Count_Heuristic_Search(
        log_path: str, 
        cached_rows: int, 
        batches_id: list, 
        batches_freq: list, 
        warm_up_steps:int = 0, 
        init_plan: list = None, 
        init_cache_state: np.ndarray = None, 
        init_cost: int = None, 
        process_id: int = None, 
        init_freq_status: np.ndarray  = None, 
        search_limit: int = 2000,
        hotness_diff_threshold_base_relax_ratio: float = 0.8,
        hotness_diff_threshold_update_window: int = 10,
        hotness_diff_threshold_startup_cap: int = 10,
        hotness_diff_threshold_increment_relax_ratio: float = 0.001,
        hotness_diff_threshold_late_time_cap: float = 1,
        hotness_diff_threshold_relax_ratio_penalty_rate: float = 0.8,
        ) -> tuple:
    """
    New_Heuristic_Search with different output (count search steps)
    """
    count_search_step = 0
    
    num_batches = len(batches_id)
    
    if process_id is not None:
        process_info = "[Process " + str(process_id) + "] "
        log_file_suffix = "-" + str(process_id)
    else:
        process_info = ""
        log_file_suffix = ""
    log_file_path = os.path.join(log_path, ("search-log" + log_file_suffix + ".txt"))
    search_log = open(log_file_path, "w")
    # search_log_l2 = open(os.path.join(log_path, "search-log-level-2" + log_file_suffix + ".txt"), "w")

    if isinstance(init_plan, list):
        plan = init_plan
    else:
        plan = list()
    
    unused_batch = [i for i in range(num_batches) if i not in plan]
    random.shuffle(unused_batch)

    # The warm up phase (randomly fill in some steps since the first few steps don't really matter that much.)
    time_warmup_start = time.time()
    if len(plan) < warm_up_steps:
        fill_in_length = warm_up_steps - len(plan)
        plan = plan + unused_batch[ : fill_in_length]
        del unused_batch[ : fill_in_length]
    if init_cache_state is not None and init_cost is not None and init_freq_status is not None:
        cost_total = init_cost
        cache_state = init_cache_state
        freq_state = init_freq_status
    else:
        cost_total, cache_state, freq_state = New_Simulate_Cost(plan, cached_rows, batches_id, batches_freq)
    
    time_warmup_finished = time.time()

    # Set up the hotness difference threshold.
    hotness_diff_threshold = LARGE_NUMBER
    hotness_diff_history = [hotness_diff_threshold] * hotness_diff_threshold_update_window
    hotness_diff_history_idx = 0
    hotness_diff_threshold_dynamic_ratio = hotness_diff_threshold_base_relax_ratio
    hotness_diff_threshold_ratio_increment = 0

    # output_str = "[Warm up phase] cost: " + str(cost_total) + ", cache_usage: " + str(len(cache_state) / cached_rows) + " warmup time: " + str(time_warmup_finished - time_warmup_start) + "\n[Startup plan] " + str(plan) + "\n[Start searching] hotness_diff threshold = " + str(hotness_diff_threshold)
    # print(output_str)
    # search_log.write(output_str + "\n")

    # Searching
    time_last_step = time.time()
    start_step = len(plan)
    for step in range(start_step, num_batches):
        # Print progress
        if ((step - start_step) % (int((num_batches - start_step) / 100) + 1)) == 0:
            print(process_info + str(int((step - start_step) / ((num_batches - start_step) / 100))) + "%")
        
        cost_best_choice = LARGE_NUMBER
        # cache_state_best_choice = None
        best_choice = 0
        cost_worst_choice = 0
        num_available_rows = np.count_nonzero(cache_state == -1)

        # Search at most self.search_limit steps to find a choice
        for choice_idx in range(len(unused_batch)):
            # Just calculate cost, don't update cache.
            batch_id = batches_id[unused_batch[choice_idx]]
            mask_ids_to_comm = np.isin(batch_id, cache_state, assume_unique=True, invert=True)
            ids_to_comm = batch_id[mask_ids_to_comm]
            num_ids_to_comm = len(ids_to_comm)                                                                              # Cost 1: cost of moving data in
            num_rows_to_evic = num_ids_to_comm - num_available_rows
            num_rows_to_evic = num_rows_to_evic if num_rows_to_evic > 0 else 0                                              # Cost 2: cost of moving data out
            cost_tmp = num_ids_to_comm + num_rows_to_evic
            
            # Record a better/worse choice
            if cost_tmp < cost_best_choice:
                cost_best_choice = cost_tmp
                best_choice = choice_idx
            if cost_tmp > cost_worst_choice:
                cost_worst_choice = cost_tmp
            
            # Early stop conditions
            hotness_diff_tmp = cost_worst_choice - cost_best_choice
            if choice_idx > hotness_diff_threshold_startup_cap and hotness_diff_tmp > hotness_diff_threshold:
                hotness_diff_threshold_ratio_increment = hotness_diff_threshold_ratio_increment + hotness_diff_threshold_increment_relax_ratio
                hotness_diff_threshold_dynamic_ratio = hotness_diff_threshold_base_relax_ratio + hotness_diff_threshold_ratio_increment
                break

            # Late stop conditions
            if choice_idx > search_limit or choice_idx == (len(unused_batch) - 1):
                break

            # Level 2 logging
            # output_str = "[Choice " + str(choice_idx) +" in Step " + str(step) + "] choice: " + str(unused_batch[choice_idx]) + ", cost: " + str(cost_tmp) + ", best_cost: " + str(cost_best_choice) + ", worst_cost: " + str(cost_worst_choice) + ", hotness_diff: " + str(hotness_diff_tmp)
            # search_log_l2.write(output_str + "\n")
        
        # Decide current step and actually update cache state.
        choice = unused_batch.pop(best_choice)
        plan.append(choice)
        # cache_state = cache_state_best_choice
        cost_total = cost_total + cost_best_choice
        random.shuffle(unused_batch)

        # Update cache
        batch_id = batches_id[choice]
        batch_freq = batches_freq[choice]

        mask_ids_to_comm = np.isin(batch_id, cache_state, assume_unique=True, invert=True)
        ids_to_comm = batch_id[mask_ids_to_comm]
        num_ids_to_comm = len(ids_to_comm)
        if num_ids_to_comm > cached_rows:
            print("Even batch with the smallest cost exceeds the capacity of the cache.")
        freq_ids_to_comm = batch_freq[mask_ids_to_comm]
        num_rows_to_evic = num_ids_to_comm - num_available_rows

        if num_rows_to_evic > 0:
            mask_evictable_row = np.isin(cache_state, batch_id, assume_unique=True, invert=True)
            idx_nonempty_row = np.flatnonzero(mask_evictable_row)[cache_state[mask_evictable_row] != -1]
            idx_idx_freq_evictable_row = nlargest(freq_state[idx_nonempty_row], num_rows_to_evic, reverse=True)
            idx_freq_evictable_row = idx_nonempty_row[idx_idx_freq_evictable_row]
            # idx_freq_evictable_row = nlargest(freq_state[mask_evictable_row], num_rows_to_evic)                      # Very expensive!!!
            cache_state[idx_freq_evictable_row] = -1
            freq_state[idx_freq_evictable_row] = 0

        idx_update = np.flatnonzero(cache_state == -1)[:num_ids_to_comm]
        cache_state[idx_update] = ids_to_comm
        freq_state[idx_update] += freq_ids_to_comm

        # Update threshold
        hotness_diff_history[hotness_diff_history_idx] = cost_worst_choice - cost_best_choice
        hotness_diff_history_idx = (hotness_diff_history_idx + 1) % hotness_diff_threshold_update_window
        hotness_diff_mean = np.mean(hotness_diff_history)
        hotness_diff_threshold = hotness_diff_mean * hotness_diff_threshold_dynamic_ratio

        # Any step takes more than 0.3s should be considered as slightly late, thus no increment of ralax ratio
        step_time = time.time() - time_last_step
        time_last_step = time.time()
        if step_time > hotness_diff_threshold_late_time_cap:
            hotness_diff_threshold_ratio_increment = hotness_diff_threshold_ratio_increment * hotness_diff_threshold_relax_ratio_penalty_rate
            hotness_diff_threshold_dynamic_ratio = hotness_diff_threshold_base_relax_ratio + hotness_diff_threshold_ratio_increment

        # Logging
        count_search_step += choice_idx
        output_str = "[Step " + str(step) + "] choice: " + str(choice) + ", cost: " + str(cost_best_choice) + ", hotness_diff: " + str(cost_worst_choice - cost_best_choice) + ", cache_usage(last run): " + str(1 - (num_available_rows/ cached_rows)) + ", step_time = " + str(step_time) + ", searched choices: " + str(choice_idx) + "\n             hotness_diff_history: " + str(hotness_diff_history) + "\n             mean hotness_diff: " + str(hotness_diff_mean) + ", ratio_increment: " + str(hotness_diff_threshold_ratio_increment) + ", dynamic threshold ratio: " + str(hotness_diff_threshold_dynamic_ratio) + ", new threshold: " + str(hotness_diff_threshold)
        # print(output_str)
        search_log.write(output_str + "\n")
    
    # output_str = "[cost: " + str(cost_total) + "] Training plan generated: " + str(plan)
    # print(output_str)
    # search_log.write(output_str + "\n")

    search_log.close()
    # search_log_l2.close()

    return plan, cost_total, count_search_step

def List_to_Parquet(list, output_path: str, name: str = None) -> None:
    if name is None:
        name = "list"
    df.DataFrame({name: list}).to_parquet(output_path)

def Non_LFU_Simulate_Cost(plan: list, cached_rows: int, batches_id: list, init_cache_state: np.ndarray = None, start_step: int = 0, start_cost: int = 0, process_id: int = None, should_print: bool = False):
    '''
    New_Simulated_Cost - LFU. This can't be used when there exists batch gap in prefetching scenario.
    '''
    num_batches = len(batches_id)

    if process_id is not None:
        process_info = "[Process " + str(process_id) + "] "
    else:
        process_info = ""

    num_steps = len(plan)
    if isinstance(init_cache_state, np.ndarray):
        gpu_cache = init_cache_state
        cost_total = start_cost
        num_available_rows = cached_rows - len(gpu_cache)
    else:
        gpu_cache = np.array([-1] * cached_rows, dtype=np.int32)
        cost_total = 0
        start_step = 0
        num_available_rows = cached_rows
    
    # Operations to output
    # ids_to_move_in = list()
    # slots_to_update = list()
    # slots_to_evict = list()
    # slots_to_move_in = list()

    # the main loop
    for step in range(start_step, num_steps):
        # Print progress
        if should_print and ((step - start_step) % (int((num_steps - start_step) / 10) + 1)) == 0:
            print(process_info + str(int((step - start_step) / ((num_steps - start_step) / 10))) + "0%")
        
        # Get ids in current batch
        batch_id = batches_id[plan[step]]

        # Find ids that need to be transfered to cache
        mask_ids_to_comm = np.isin(batch_id, gpu_cache, assume_unique=True, invert=True)
        ids_to_comm = batch_id[mask_ids_to_comm]
        # ids_to_move_in.append(list(ids_to_comm))
        num_ids_to_comm = len(ids_to_comm)                                                                              # Cost 1: cost of moving data in
        num_rows_to_evic = num_ids_to_comm - num_available_rows
        num_rows_to_evic = num_rows_to_evic                                              # Cost 2: cost of moving data out

        # Evic first num_rows_to_evic rows of ids, since the gpu_cache is sorted by batch flags.
        if num_rows_to_evic > 0:
            mask_evictable_row = np.isin(gpu_cache, batch_id, assume_unique=True, invert=True)
            idx_nonempty_row = np.flatnonzero(mask_evictable_row)[gpu_cache[mask_evictable_row] != -1]
            idx_evict_rows = idx_nonempty_row[:num_rows_to_evic]
            # slots_to_evict.append(list(idx_evict_rows))
            gpu_cache[idx_evict_rows] = -1
        else:
            num_rows_to_evic = 0
            # slots_to_evict.append(list())
        
        # Update gpu cache
        idx_update = np.flatnonzero(gpu_cache == -1)[:num_ids_to_comm]
        gpu_cache[idx_update] = ids_to_comm
        
        num_available_rows = num_available_rows + num_rows_to_evic - num_ids_to_comm
        cost_total = cost_total + num_ids_to_comm + num_rows_to_evic

    return cost_total, gpu_cache

def New_None_LFU_Cost_Wrapper(q: Queue, process_id: int, plan: list, cached_rows: int, batches_id: list):
    cost, _ = Non_LFU_Simulate_Cost(plan, cached_rows, batches_id, None, 0, 0, process_id, False)
    q.put(cost)

def Detailed_Heuristic_Search(
        log_path: str, 
        cached_rows: int, 
        batches_id: list, 
        batches_freq: list, 
        warm_up_steps:int = 0, 
        init_plan: list = None, 
        init_cache_state: np.ndarray = None, 
        init_cost: int = None, 
        process_id: int = None, 
        init_freq_status: np.ndarray  = None, 
        search_limit: int = 2000,
        hotness_diff_threshold_base_relax_ratio: float = 0.8,
        hotness_diff_threshold_update_window: int = 10,
        hotness_diff_threshold_startup_cap: int = 10,
        hotness_diff_threshold_increment_relax_ratio: float = 0.001,
        hotness_diff_threshold_late_time_cap: float = 1,
        hotness_diff_threshold_relax_ratio_penalty_rate: float = 0.8,
        ) -> tuple:
    """
    New_Heuristic_Search - LFU + detailed cache operations
    """
    num_batches = len(batches_id)
    
    if process_id is not None:
        process_info = "[Process " + str(process_id) + "] "
        log_file_suffix = "-" + str(process_id)
    else:
        process_info = ""
        log_file_suffix = ""
    log_file_path = os.path.join(log_path, ("search-log" + log_file_suffix + ".txt"))
    search_log = open(log_file_path, "w")
    # search_log_l2 = open(os.path.join(log_path, "search-log-level-2" + log_file_suffix + ".txt"), "w")

    if isinstance(init_plan, list):
        plan = init_plan
    else:
        plan = list()
    
    unused_batch = [i for i in range(num_batches) if i not in plan]
    random.shuffle(unused_batch)

    # The warm up phase (randomly fill in some steps since the first few steps don't really matter that much.)
    time_warmup_start = time.time()
    if len(plan) < warm_up_steps:
        fill_in_length = warm_up_steps - len(plan)
        plan = plan + unused_batch[ : fill_in_length]
        del unused_batch[ : fill_in_length]
    if init_cache_state is not None and init_cost is not None and init_freq_status is not None:
        cost_total = init_cost
        cache_state = init_cache_state
        freq_state = init_freq_status
    else:
        cost_total, cache_state, freq_state = New_Simulate_Cost(plan, cached_rows, batches_id, batches_freq)
    
    time_warmup_finished = time.time()

    # Set up the hotness difference threshold.
    hotness_diff_threshold = LARGE_NUMBER
    hotness_diff_history = [hotness_diff_threshold] * hotness_diff_threshold_update_window
    hotness_diff_history_idx = 0
    hotness_diff_threshold_dynamic_ratio = hotness_diff_threshold_base_relax_ratio
    hotness_diff_threshold_ratio_increment = 0

    # output_str = "[Warm up phase] cost: " + str(cost_total) + ", cache_usage: " + str(len(cache_state) / cached_rows) + " warmup time: " + str(time_warmup_finished - time_warmup_start) + "\n[Startup plan] " + str(plan) + "\n[Start searching] hotness_diff threshold = " + str(hotness_diff_threshold)
    # print(output_str)
    # search_log.write(output_str + "\n")

    # Searching
    time_last_step = time.time()
    start_step = len(plan)
    for step in range(start_step, num_batches):
        # Print progress
        if ((step - start_step) % (int((num_batches - start_step) / 100) + 1)) == 0:
            print(process_info + str(int((step - start_step) / ((num_batches - start_step) / 100))) + "%")
        
        cost_best_choice = LARGE_NUMBER
        # cache_state_best_choice = None
        best_choice = 0
        cost_worst_choice = 0
        num_available_rows = np.count_nonzero(cache_state == -1)

        # Search at most self.search_limit steps to find a choice
        for choice_idx in range(len(unused_batch)):
            # Just calculate cost, don't update cache.
            batch_id = batches_id[unused_batch[choice_idx]]
            mask_ids_to_comm = np.isin(batch_id, cache_state, assume_unique=True, invert=True)
            ids_to_comm = batch_id[mask_ids_to_comm]
            num_ids_to_comm = len(ids_to_comm)                                                                              # Cost 1: cost of moving data in
            num_rows_to_evic = num_ids_to_comm - num_available_rows
            num_rows_to_evic = num_rows_to_evic if num_rows_to_evic > 0 else 0                                              # Cost 2: cost of moving data out
            cost_tmp = num_ids_to_comm + num_rows_to_evic
            
            # Record a better/worse choice
            if cost_tmp < cost_best_choice:
                cost_best_choice = cost_tmp
                best_choice = choice_idx
            if cost_tmp > cost_worst_choice:
                cost_worst_choice = cost_tmp
            
            # Early stop conditions
            hotness_diff_tmp = cost_worst_choice - cost_best_choice
            if choice_idx > hotness_diff_threshold_startup_cap and hotness_diff_tmp > hotness_diff_threshold:
                hotness_diff_threshold_ratio_increment = hotness_diff_threshold_ratio_increment + hotness_diff_threshold_increment_relax_ratio
                hotness_diff_threshold_dynamic_ratio = hotness_diff_threshold_base_relax_ratio + hotness_diff_threshold_ratio_increment
                break

            # Late stop conditions
            if choice_idx > search_limit or choice_idx == (len(unused_batch) - 1):
                break

            # Level 2 logging
            # output_str = "[Choice " + str(choice_idx) +" in Step " + str(step) + "] choice: " + str(unused_batch[choice_idx]) + ", cost: " + str(cost_tmp) + ", best_cost: " + str(cost_best_choice) + ", worst_cost: " + str(cost_worst_choice) + ", hotness_diff: " + str(hotness_diff_tmp)
            # search_log_l2.write(output_str + "\n")
        
        # Decide current step and actually update cache state.
        choice = unused_batch.pop(best_choice)
        plan.append(choice)
        # cache_state = cache_state_best_choice
        cost_total = cost_total + cost_best_choice
        random.shuffle(unused_batch)

        # Update cache
        batch_id = batches_id[choice]
        batch_freq = batches_freq[choice]

        mask_ids_to_comm = np.isin(batch_id, cache_state, assume_unique=True, invert=True)
        ids_to_comm = batch_id[mask_ids_to_comm]
        num_ids_to_comm = len(ids_to_comm)
        if num_ids_to_comm > cached_rows:
            print("Even batch with the smallest cost exceeds the capacity of the cache.")
        freq_ids_to_comm = batch_freq[mask_ids_to_comm]
        num_rows_to_evic = num_ids_to_comm - num_available_rows

        if num_rows_to_evic > 0:
            mask_evictable_row = np.isin(cache_state, batch_id, assume_unique=True, invert=True)
            idx_nonempty_row = np.flatnonzero(mask_evictable_row)[cache_state[mask_evictable_row] != -1]
            idx_idx_freq_evictable_row = nlargest(freq_state[idx_nonempty_row], num_rows_to_evic, reverse=True)
            idx_freq_evictable_row = idx_nonempty_row[idx_idx_freq_evictable_row]
            # idx_freq_evictable_row = nlargest(freq_state[mask_evictable_row], num_rows_to_evic)                      # Very expensive!!!
            cache_state[idx_freq_evictable_row] = -1
            freq_state[idx_freq_evictable_row] = 0

        idx_update = np.flatnonzero(cache_state == -1)[:num_ids_to_comm]
        cache_state[idx_update] = ids_to_comm
        freq_state[idx_update] += freq_ids_to_comm

        # Update threshold
        hotness_diff_history[hotness_diff_history_idx] = cost_worst_choice - cost_best_choice
        hotness_diff_history_idx = (hotness_diff_history_idx + 1) % hotness_diff_threshold_update_window
        hotness_diff_mean = np.mean(hotness_diff_history)
        hotness_diff_threshold = hotness_diff_mean * hotness_diff_threshold_dynamic_ratio

        # Any step takes more than 0.3s should be considered as slightly late, thus no increment of ralax ratio
        step_time = time.time() - time_last_step
        time_last_step = time.time()
        if step_time > hotness_diff_threshold_late_time_cap:
            hotness_diff_threshold_ratio_increment = hotness_diff_threshold_ratio_increment * hotness_diff_threshold_relax_ratio_penalty_rate
            hotness_diff_threshold_dynamic_ratio = hotness_diff_threshold_base_relax_ratio + hotness_diff_threshold_ratio_increment

        # Logging
        
        output_str = "[Step " + str(step) + "] choice: " + str(choice) + ", cost: " + str(cost_best_choice) + ", hotness_diff: " + str(cost_worst_choice - cost_best_choice) + ", cache_usage(last run): " + str(1 - (num_available_rows/ cached_rows)) + ", step_time = " + str(step_time) + ", searched choices: " + str(choice_idx) + "\n             hotness_diff_history: " + str(hotness_diff_history) + "\n             mean hotness_diff: " + str(hotness_diff_mean) + ", ratio_increment: " + str(hotness_diff_threshold_ratio_increment) + ", dynamic threshold ratio: " + str(hotness_diff_threshold_dynamic_ratio) + ", new threshold: " + str(hotness_diff_threshold)
        # print(output_str)
        search_log.write(output_str + "\n")
    
    # output_str = "[cost: " + str(cost_total) + "] Training plan generated: " + str(plan)
    # print(output_str)
    # search_log.write(output_str + "\n")

    search_log.close()
    # search_log_l2.close()

    return plan, cost_total, cache_state, freq_state

def Training_Plan_to_ID_of_Batches(input_path: str, output_path: str, id_batches: list, freq_batches: list):
    # Read plan
    print("Loading training plan from " + str(input_path))
    start_time = time.time()
    data = df.read_parquet(input_path)
    training_plan = data['plan'].to_list()
    
    # training_plan = [i for i in range(len(id_batches))]
    # # import pdb; pdb.set_trace()
    # random.shuffle(training_plan)
    # if DATASET == "criteo":
    #     plan_path = DLRM_PLAN_PATH
    # elif DATASET == "taobao":
    #     plan_path = TBSM_PLAN_PATH
    # else:
    #     raise RuntimeError("Unrecognized  dataset")
    # df.DataFrame({'plan': training_plan}).to_parquet(os.path.join(plan_path, ("training_plan-random.parquet")))
    
    end_time = time.time()
    print("Training plan loaded. (" + str(end_time - start_time) + "s)")
    
    # Extract ids
    id_planed_batches = list()
    freq_planed_batches = list()
    for i in range(len(training_plan)):
        id_planed_batches.append(df.Series(id_batches[training_plan[i]], dtype='int32'))
        freq_planed_batches.append(df.Series(freq_batches[training_plan[i]], dtype='int32'))
    output = df.DataFrame({"id_planed_batches": id_planed_batches, "freq_planed_batches": freq_planed_batches})
    import pdb; pdb.set_trace()
    output.to_parquet(output_path)

def _search_cost(unused_batch: list, batches_id: list, cache_state: np.ndarray, num_available_rows: int, hotness_diff_threshold_startup_cap:int, hotness_diff_threshold: int, search_limit:int, result: Queue):
    '''
    Do searching of one search step in a multiprocess manner.
    return:
        1. the heightest cost
        2. the lowest cost
        3. index of choice of the lowest cost
    '''
    cost_best_choice = LARGE_NUMBER
    cost_worst_choice = 0
    best_choice = 0

    # Search at most self.search_limit steps to find a choice
    for choice_idx in range(len(unused_batch)):
        # Just calculate cost, don't update cache.
        batch_id = batches_id[unused_batch[choice_idx]]
        mask_ids_to_comm = np.isin(batch_id, cache_state, assume_unique=True, invert=True)
        ids_to_comm = batch_id[mask_ids_to_comm]
        num_ids_to_comm = len(ids_to_comm)                                                                              # Cost 1: cost of moving data in
        num_rows_to_evic = num_ids_to_comm - num_available_rows
        num_rows_to_evic = num_rows_to_evic if num_rows_to_evic > 0 else 0                                              # Cost 2: cost of moving data out
        cost_tmp = num_ids_to_comm + num_rows_to_evic
        
        # Record a better/worse choice
        if cost_tmp < cost_best_choice:
            cost_best_choice = cost_tmp
            best_choice = choice_idx
        if cost_tmp > cost_worst_choice:
            cost_worst_choice = cost_tmp
        
        # Early stop conditions
        hotness_diff_tmp = cost_worst_choice - cost_best_choice
        if choice_idx > hotness_diff_threshold_startup_cap and hotness_diff_tmp > hotness_diff_threshold:
            # put best and worst cost
            break

        # Late stop conditions
        if choice_idx > search_limit or choice_idx == (len(unused_batch) - 1):
            break

        # Level 2 logging
        # output_str = "[Choice " + str(choice_idx) +" in Step " + str(step) + "] choice: " + str(unused_batch[choice_idx]) + ", cost: " + str(cost_tmp) + ", best_cost: " + str(cost_best_choice) + ", worst_cost: " + str(cost_worst_choice) + ", hotness_diff: " + str(hotness_diff_tmp)
        # search_log_l2.write(output_str + "\n")
    
    # return info
    result.put(cost_best_choice)
    result.put(cost_worst_choice)
    result.put(best_choice)

def New_None_LFU_Multiprocess_Search(
        log_path: str, 
        cached_rows: int, 
        batches_id: list,  
        warm_up_steps:int = 0, 
        init_plan: list = None, 
        init_cache_state: np.ndarray = None, 
        init_cost: int = None, 
        search_limit: int = 2000,
        hotness_diff_threshold_base_relax_ratio: float = 0.8,
        hotness_diff_threshold_update_window: int = 10,
        hotness_diff_threshold_startup_cap: int = 10,
        hotness_diff_threshold_increment_relax_ratio: float = 0.001,
        hotness_diff_threshold_late_time_cap: float = 1,
        hotness_diff_threshold_relax_ratio_penalty_rate: float = 0.8,
        num_process: int = 40,
        ) -> tuple:
    num_batches = len(batches_id)

    log_file_path = os.path.join(log_path, ("new-none-lfu-search-log" + PLAN_FILE_NAME + ".txt"))
    search_log = open(log_file_path, "w")
    # search_log_l2 = open(os.path.join(log_path, "search-log-level-2" + log_file_suffix + ".txt"), "w")

    if isinstance(init_plan, list):
        plan = init_plan
    else:
        plan = list()
    
    unused_batch = [i for i in range(num_batches) if i not in plan]
    random.shuffle(unused_batch)

    # The warm up phase (randomly fill in some steps since the first few steps don't really matter that much.)
    time_warmup_start = time.time()
    if len(plan) < warm_up_steps:
        fill_in_length = warm_up_steps - len(plan)
        plan = plan + unused_batch[ : fill_in_length]
        del unused_batch[ : fill_in_length]
    if init_cache_state is not None and init_cost is not None:
        cost_total = init_cost
        cache_state = init_cache_state
    else:
        cost_total, cache_state = Non_LFU_Simulate_Cost(plan, cached_rows, batches_id)
    
    time_warmup_finished = time.time()

    # Set up the hotness difference threshold.
    hotness_diff_threshold = LARGE_NUMBER
    hotness_diff_history = [hotness_diff_threshold] * hotness_diff_threshold_update_window
    hotness_diff_history_idx = 0
    hotness_diff_threshold_dynamic_ratio = hotness_diff_threshold_base_relax_ratio
    hotness_diff_threshold_ratio_increment = 0

    output_str = "[Warm up phase] cost: " + str(cost_total) + ", cache_usage: " + str(len(cache_state) / cached_rows) + " warmup time: " + str(time_warmup_finished - time_warmup_start) + "\n[Start searching] hotness_diff threshold = " + str(hotness_diff_threshold)
    print(output_str)
    search_log.write(output_str + "\n")

    # Searching
    time_last_step = time.time()
    start_step = len(plan)
    for step in range(start_step, num_batches):
        # Print progress
        if (step % (int(num_batches / 100) + 1)) == 0:
            print(str(int(step / (num_batches / 100))) + "%")
            with open(os.path.join(log_path, "live_backup" + PLAN_FILE_NAME + ".json"), "w") as backup_file:
                json.dump(plan, backup_file)
        
        # cost_best_choice = LARGE_NUMBER
        # cache_state_best_choice = None
        # best_choice = 0
        # cost_worst_choice = 0
        num_available_rows = np.count_nonzero(cache_state == -1)

        # Miltiprocess search
        per_process_unused_batch = int(len(unused_batch) / num_process)
        results = list()
        processes = list()

        for i in range(num_process - 1):
            results.append(Queue())
            processes.append(Process(
                target=_search_cost, 
                args=(
                    unused_batch[i * per_process_unused_batch : (i + 1) * per_process_unused_batch], 
                    batches_id, 
                    cache_state,
                    num_available_rows,
                    hotness_diff_threshold_startup_cap,
                    hotness_diff_threshold,
                    search_limit,
                    results[i]
                    ),
                ))
            processes[i].start()
        
        results.append(Queue())
        processes.append(Process(
            target=_search_cost, 
            args=(
                unused_batch[(num_process - 1) * per_process_unused_batch:], 
                batches_id, 
                cache_state,
                num_available_rows,
                hotness_diff_threshold_startup_cap,
                hotness_diff_threshold,
                search_limit,
                results[num_process - 1]
                ),
        ))
        processes[num_process - 1].start()

        global_cost_best = LARGE_NUMBER
        global_cost_worst = 0
        global_best_choice = 0
        for i in range(num_process):
            processes[i].join()
            local_cost_best = results[i].get()
            local_cost_worst = results[i].get()
            local_best_choice = results[i].get()

            if local_cost_best < global_cost_best:
                global_cost_best = local_cost_best
                global_best_choice = i * per_process_unused_batch + local_best_choice
            
            if local_cost_worst > global_cost_worst:
                global_cost_worst = local_cost_worst
        
        # Decide current step and actually update cache state.
        choice = unused_batch.pop(global_best_choice)
        plan.append(choice)
        # cache_state = cache_state_best_choice
        cost_total = cost_total + global_best_choice
        random.shuffle(unused_batch)

        # Update hotness threshold
        if (global_cost_worst - global_cost_best) > hotness_diff_threshold:
            hotness_diff_threshold_ratio_increment = hotness_diff_threshold_ratio_increment + hotness_diff_threshold_increment_relax_ratio
            hotness_diff_threshold_dynamic_ratio = hotness_diff_threshold_base_relax_ratio + hotness_diff_threshold_ratio_increment

        # Update cache
        batch_id = batches_id[choice]
        mask_ids_to_comm = np.isin(batch_id, cache_state, assume_unique=True, invert=True)
        ids_to_comm = batch_id[mask_ids_to_comm]
        num_ids_to_comm = len(ids_to_comm)
        if num_ids_to_comm > cached_rows:
            print("Even batch with the smallest cost exceeds the capacity of the cache.")
        num_rows_to_evic = num_ids_to_comm - num_available_rows

        if num_rows_to_evic > 0:
            mask_evictable_row = np.isin(cache_state, batch_id, assume_unique=True, invert=True)
            idx_nonempty_row = np.flatnonzero(mask_evictable_row)[cache_state[mask_evictable_row] != -1]
            idx_evictable_row = idx_nonempty_row[:num_rows_to_evic]
            cache_state[idx_evictable_row] = -1

        idx_update = np.flatnonzero(cache_state == -1)[:num_ids_to_comm]
        cache_state[idx_update] = ids_to_comm

        # Update threshold
        hotness_diff_history[hotness_diff_history_idx] = global_cost_worst - global_cost_best
        hotness_diff_history_idx = (hotness_diff_history_idx + 1) % hotness_diff_threshold_update_window
        hotness_diff_mean = np.mean(hotness_diff_history)
        hotness_diff_threshold = hotness_diff_mean * hotness_diff_threshold_dynamic_ratio

        # Any step takes more than 0.3s should be considered as slightly late, thus no increment of ralax ratio
        step_time = time.time() - time_last_step
        time_last_step = time.time()
        if step_time > hotness_diff_threshold_late_time_cap:
            hotness_diff_threshold_ratio_increment = hotness_diff_threshold_ratio_increment * hotness_diff_threshold_relax_ratio_penalty_rate
            hotness_diff_threshold_dynamic_ratio = hotness_diff_threshold_base_relax_ratio + hotness_diff_threshold_ratio_increment

        # Logging
        
        output_str = "[Step " + str(step) + "] choice: " + str(choice) + ", cost: " + str(global_cost_best) + ", hotness_diff: " + str(global_cost_worst - global_cost_best) + ", cache_usage(last run): " + str(1 - (num_available_rows/ cached_rows)) + ", step_time = " + str(step_time) + "\n             hotness_diff_history: " + str(hotness_diff_history) + "\n             mean hotness_diff: " + str(hotness_diff_mean) + ", ratio_increment: " + str(hotness_diff_threshold_ratio_increment) + ", dynamic threshold ratio: " + str(hotness_diff_threshold_dynamic_ratio) + ", new threshold: " + str(hotness_diff_threshold)
        if step < 10:
            print(output_str)
        search_log.write(output_str + "\n")
    
    # output_str = "[cost: " + str(cost_total) + "] Training plan generated: " + str(plan)
    # print(output_str)
    # search_log.write(output_str + "\n")

    search_log.close()
    # search_log_l2.close()

    return plan, cost_total, cache_state

def New_LFU_Multiprocess_Search(
        log_path: str, 
        cached_rows: int, 
        batches_id: list,
        batches_freq: list,
        warm_up_steps:int = 0, 
        init_plan: list = None, 
        init_cache_state: np.ndarray = None,
        init_freq_state: np.ndarray = None,
        init_cost: int = None, 
        search_limit: int = 2000,
        hotness_diff_threshold_base_relax_ratio: float = 0.8,
        hotness_diff_threshold_update_window: int = 10,
        hotness_diff_threshold_startup_cap: int = 10,
        hotness_diff_threshold_increment_relax_ratio: float = 0.001,
        hotness_diff_threshold_late_time_cap: float = 1,
        hotness_diff_threshold_relax_ratio_penalty_rate: float = 0.8,
        num_process: int = 40,
        ) -> tuple:
    num_batches = len(batches_id)

    log_file_path = os.path.join(log_path, ("new-lfu-search-log" + PLAN_FILE_NAME + ".txt"))
    search_log = open(log_file_path, "w")
    # search_log_l2 = open(os.path.join(log_path, "search-log-level-2" + log_file_suffix + ".txt"), "w")

    if isinstance(init_plan, list):
        plan = init_plan
    else:
        plan = list()
    
    unused_batch = [i for i in range(num_batches) if i not in plan]
    random.shuffle(unused_batch)

    # The warm up phase (randomly fill in some steps since the first few steps don't really matter that much.)
    time_warmup_start = time.time()
    if len(plan) < warm_up_steps:
        fill_in_length = warm_up_steps - len(plan)
        plan = plan + unused_batch[ : fill_in_length]
        del unused_batch[ : fill_in_length]
    if init_cache_state is not None and init_cost is not None and init_freq_state is not None:
        cost_total = init_cost
        cache_state = init_cache_state
        freq_state = init_freq_state
    else:
        cost_total, cache_state, freq_state = New_Simulate_Cost(plan, cached_rows, batches_id, batches_freq, should_print=False)
    
    time_warmup_finished = time.time()

    # Set up the hotness difference threshold.
    hotness_diff_threshold = LARGE_NUMBER
    hotness_diff_history = [hotness_diff_threshold] * hotness_diff_threshold_update_window
    hotness_diff_history_idx = 0
    hotness_diff_threshold_dynamic_ratio = hotness_diff_threshold_base_relax_ratio
    hotness_diff_threshold_ratio_increment = 0

    output_str = "[Warm up phase] cost: " + str(cost_total) + ", cache_usage: " + str(len(cache_state) / cached_rows) + " warmup time: " + str(time_warmup_finished - time_warmup_start) + "\n[Start searching] hotness_diff threshold = " + str(hotness_diff_threshold)
    print(output_str)
    search_log.write(output_str + "\n")

    # Searching
    time_last_step = time.time()
    start_step = len(plan)
    for step in range(start_step, num_batches):
        # Print progress
        if (step % (int(num_batches / 100) + 1)) == 0:
            print(str(int(step / (num_batches / 100))) + "%")
            with open(os.path.join(log_path, "live_backup-lfu-" + PLAN_FILE_NAME + ".json"), "w") as backup_file:
                json.dump(plan, backup_file)
        
        # cost_best_choice = LARGE_NUMBER
        # cache_state_best_choice = None
        # best_choice = 0
        # cost_worst_choice = 0
        num_available_rows = np.count_nonzero(cache_state == -1)

        # Miltiprocess search
        per_process_unused_batch = int(len(unused_batch) / num_process)
        results = list()
        processes = list()

        for i in range(num_process - 1):
            results.append(Queue())
            processes.append(Process(
                target=_search_cost, 
                args=(
                    unused_batch[i * per_process_unused_batch : (i + 1) * per_process_unused_batch], 
                    batches_id, 
                    cache_state,
                    num_available_rows,
                    hotness_diff_threshold_startup_cap,
                    hotness_diff_threshold,
                    search_limit,
                    results[i]
                    ),
                ))
            processes[i].start()
        
        results.append(Queue())
        processes.append(Process(
            target=_search_cost, 
            args=(
                unused_batch[(num_process - 1) * per_process_unused_batch:], 
                batches_id, 
                cache_state,
                num_available_rows,
                hotness_diff_threshold_startup_cap,
                hotness_diff_threshold,
                search_limit,
                results[num_process - 1]
                ),
        ))
        processes[num_process - 1].start()

        global_cost_best = LARGE_NUMBER
        global_cost_worst = 0
        global_best_choice = 0
        for i in range(num_process):
            processes[i].join()
            local_cost_best = results[i].get()
            local_cost_worst = results[i].get()
            local_best_choice = results[i].get()

            if local_cost_best < global_cost_best:
                global_cost_best = local_cost_best
                global_best_choice = i * per_process_unused_batch + local_best_choice
            
            if local_cost_worst > global_cost_worst:
                global_cost_worst = local_cost_worst
        
        # Decide current step and actually update cache state.
        choice = unused_batch.pop(global_best_choice)
        plan.append(choice)
        # cache_state = cache_state_best_choice
        cost_total = cost_total + global_best_choice
        random.shuffle(unused_batch)

        # Update hotness threshold
        if (global_cost_worst - global_cost_best) > hotness_diff_threshold:
            hotness_diff_threshold_ratio_increment = hotness_diff_threshold_ratio_increment + hotness_diff_threshold_increment_relax_ratio
            hotness_diff_threshold_dynamic_ratio = hotness_diff_threshold_base_relax_ratio + hotness_diff_threshold_ratio_increment

        # Update cache
        batch_id = batches_id[choice]
        batch_freq = batches_freq[choice]

        mask_ids_to_comm = np.isin(batch_id, cache_state, assume_unique=True, invert=True)
        ids_to_comm = batch_id[mask_ids_to_comm]
        num_ids_to_comm = len(ids_to_comm)
        if num_ids_to_comm > cached_rows:
            print("Even batch with the smallest cost exceeds the capacity of the cache.")
        freq_ids_to_comm = batch_freq[mask_ids_to_comm]
        num_rows_to_evic = num_ids_to_comm - num_available_rows

        if num_rows_to_evic > 0:
            mask_evictable_row = np.isin(cache_state, batch_id, assume_unique=True, invert=True)
            idx_nonempty_row = np.flatnonzero(mask_evictable_row)[cache_state[mask_evictable_row] != -1]
            # idx_evictable_row = idx_nonempty_row[:num_rows_to_evic]
            # cache_state[idx_evictable_row] = -1
            idx_idx_freq_evictable_row = nlargest(freq_state[idx_nonempty_row], num_rows_to_evic, reverse=True)
            idx_freq_evictable_row = idx_nonempty_row[idx_idx_freq_evictable_row]
            cache_state[idx_freq_evictable_row] = -1
            freq_state[idx_freq_evictable_row] = 0

        idx_update = np.flatnonzero(cache_state == -1)[:num_ids_to_comm]
        cache_state[idx_update] = ids_to_comm
        freq_state[idx_update] += freq_ids_to_comm

        # Update threshold
        hotness_diff_history[hotness_diff_history_idx] = global_cost_worst - global_cost_best
        hotness_diff_history_idx = (hotness_diff_history_idx + 1) % hotness_diff_threshold_update_window
        hotness_diff_mean = np.mean(hotness_diff_history)
        hotness_diff_threshold = hotness_diff_mean * hotness_diff_threshold_dynamic_ratio

        # Any step takes more than 0.3s should be considered as slightly late, thus no increment of ralax ratio
        step_time = time.time() - time_last_step
        time_last_step = time.time()
        if step_time > hotness_diff_threshold_late_time_cap:
            hotness_diff_threshold_ratio_increment = hotness_diff_threshold_ratio_increment * hotness_diff_threshold_relax_ratio_penalty_rate
            hotness_diff_threshold_dynamic_ratio = hotness_diff_threshold_base_relax_ratio + hotness_diff_threshold_ratio_increment

        # Logging
        
        output_str = "[Step " + str(step) + "] choice: " + str(choice) + ", cost: " + str(global_cost_best) + ", hotness_diff: " + str(global_cost_worst - global_cost_best) + ", cache_usage(last run): " + str(1 - (num_available_rows/ cached_rows)) + ", step_time = " + str(step_time) + "\n             hotness_diff_history: " + str(hotness_diff_history) + "\n             mean hotness_diff: " + str(hotness_diff_mean) + ", ratio_increment: " + str(hotness_diff_threshold_ratio_increment) + ", dynamic threshold ratio: " + str(hotness_diff_threshold_dynamic_ratio) + ", new threshold: " + str(hotness_diff_threshold)
        if step < 10:
            print(output_str)
        search_log.write(output_str + "\n")
    
    # output_str = "[cost: " + str(cost_total) + "] Training plan generated: " + str(plan)
    # print(output_str)
    # search_log.write(output_str + "\n")

    search_log.close()
    # search_log_l2.close()

    return plan, cost_total, cache_state, freq_state

def Segmented_LFU_Multiprocess_Search(
        log_path: str, 
        cached_rows: int, 
        batches_id: list,
        batches_freq: list,
        warm_up_steps:int = 0, 
        init_plan: list = None, 
        init_cache_state: np.ndarray = None,
        init_freq_state: np.ndarray = None,
        init_cost: int = None, 
        search_limit: int = 200,
        hotness_diff_threshold_base_relax_ratio: float = 0.8,
        hotness_diff_threshold_update_window: int = 10,
        hotness_diff_threshold_startup_cap: int = 10,
        hotness_diff_threshold_increment_relax_ratio: float = 0.001,
        hotness_diff_threshold_late_time_cap: float = 1,
        hotness_diff_threshold_relax_ratio_penalty_rate: float = 0.8,
        num_process: int = 40,
        ) -> tuple:
    num_batches = len(batches_id)
    
    unused_batch = [i for i in range(num_batches) if i not in init_plan]
    random.shuffle(unused_batch)

    # split the remaining unused batches to sub-processes
    per_process_unused_batch = int(len(unused_batch) / num_process)
    results = list()
    processes = list()

    plan = init_plan
    rough_cost, init_cache_state, init_freq_state = New_Simulate_Cost(plan, cached_rows, batches_id, batches_freq)

    # results.append(Queue())
    # processes.append(Process(
    #     target=Wrapper_Search, 
    #     args=(
    #         results[0],
    #         log_path,
    #         cached_rows,
    #         batches_id[:per_process_unused_batch], 
    #         batches_freq[:per_process_unused_batch], 
    #         0,
    #         search_limit,
    #         hotness_diff_threshold_base_relax_ratio,
    #         hotness_diff_threshold_update_window,
    #         hotness_diff_threshold_relax_ratio_penalty_rate,
    #         hotness_diff_threshold_increment_relax_ratio,
    #         hotness_diff_threshold_late_time_cap,
    #         hotness_diff_threshold_startup_cap,
    #         None,
    #         cache_state,
    #         0,
    #         init_freq_state,
    #         )
    #     ))
    # processes[0].start()

    for i in range(num_process):
        results.append(Queue())
        processes.append(Process(
            target=Wrapper_Search, 
            args=(
                results[i],
                log_path,
                cached_rows,
                batches_id[i * per_process_unused_batch : (i + 1) * per_process_unused_batch], 
                batches_freq[i * per_process_unused_batch : (i + 1) * per_process_unused_batch], 
                i,
                search_limit,
                hotness_diff_threshold_base_relax_ratio,
                hotness_diff_threshold_update_window,
                hotness_diff_threshold_relax_ratio_penalty_rate,
                hotness_diff_threshold_increment_relax_ratio,
                hotness_diff_threshold_late_time_cap,
                hotness_diff_threshold_startup_cap,
                None,
                init_cache_state,
                0,
                init_freq_state
                ),
            ))
        processes[i].start()
    
    for i in range(num_process):
        processes[i].join()
        local_cost = results[i].get()
        local_plan = results[i].get()
        plan = plan + local_plan
        rough_cost = rough_cost + local_cost
    
    return plan, rough_cost

def Wrapper_None_LFU_Search(
        q: Queue,
        log_path: str, 
        cached_rows: int, 
        batches_id: list, 
        process_id: int,  
        search_limit: int,
        hotness_diff_threshold_base_relax_ratio: float,
        hotness_diff_threshold_update_window: int,
        hotness_diff_threshold_relax_ratio_penalty_rate: float,
        hotness_diff_threshold_increment_relax_ratio: float,
        hotness_diff_threshold_late_time_cap: float,
        hotness_diff_threshold_startup_cap: int,
        init_plan=None,
        init_cache_state=None,
        init_cost=None,
        ):
    plan, cost, _, _ = New_None_LFU_Heuristic_Search(
        log_path=log_path, 
        cached_rows=cached_rows, 
        batches_id=batches_id, 
        warm_up_steps=0, 
        init_plan=init_plan,
        init_cache_state=init_cache_state,
        init_cost=init_cost,
        process_id=process_id,
        search_limit=search_limit, 
        hotness_diff_threshold_base_relax_ratio=hotness_diff_threshold_base_relax_ratio, 
        hotness_diff_threshold_update_window=hotness_diff_threshold_update_window,
        hotness_diff_threshold_relax_ratio_penalty_rate=hotness_diff_threshold_relax_ratio_penalty_rate,
        hotness_diff_threshold_increment_relax_ratio=hotness_diff_threshold_increment_relax_ratio,
        hotness_diff_threshold_late_time_cap=hotness_diff_threshold_late_time_cap,
        hotness_diff_threshold_startup_cap=hotness_diff_threshold_startup_cap,
        )
    q.put(cost)
    q.put(plan)

def Segmented_None_LFU_Multiprocess_Search(
        log_path: str, 
        cached_rows: int, 
        batches_id: list,
        warm_up_steps:int = 0, 
        init_plan: list = None, 
        init_cache_state: np.ndarray = None,
        init_cost: int = None, 
        search_limit: int = 200,
        hotness_diff_threshold_base_relax_ratio: float = 0.8,
        hotness_diff_threshold_update_window: int = 10,
        hotness_diff_threshold_startup_cap: int = 10,
        hotness_diff_threshold_increment_relax_ratio: float = 0.001,
        hotness_diff_threshold_late_time_cap: float = 1,
        hotness_diff_threshold_relax_ratio_penalty_rate: float = 0.8,
        num_process: int = 40,
        ) -> tuple:
    num_batches = len(batches_id)
    
    unused_batch = [i for i in range(num_batches) if i not in init_plan]
    random.shuffle(unused_batch)

    # split the remaining unused batches to sub-processes
    per_process_unused_batch = int(len(unused_batch) / num_process)
    results = list()
    processes = list()

    plan = init_plan
    rough_cost, init_cache_state = Non_LFU_Simulate_Cost(plan, cached_rows, batches_id)

    for i in range(num_process):
        results.append(Queue())
        processes.append(Process(
            target=Wrapper_None_LFU_Search, 
            args=(
                results[i],
                log_path,
                cached_rows,
                batches_id[i * per_process_unused_batch : (i + 1) * per_process_unused_batch], 
                i,
                search_limit,
                hotness_diff_threshold_base_relax_ratio,
                hotness_diff_threshold_update_window,
                hotness_diff_threshold_relax_ratio_penalty_rate,
                hotness_diff_threshold_increment_relax_ratio,
                hotness_diff_threshold_late_time_cap,
                hotness_diff_threshold_startup_cap,
                None,
                init_cache_state,
                0,
                ),
            ))
        processes[i].start()
    
    for i in range(num_process):
        processes[i].join()
        local_cost = results[i].get()
        local_plan = results[i].get()
        plan = plan + local_plan
        rough_cost = rough_cost + local_cost
    
    return plan, rough_cost

if __name__ == "__main__":
    start_time = time.time()
    print("Initializing the planner...")

    '''
    Recommended hyper parameters for criteo kaggle dataset:
        warm_up_steps=150, 
        search_limit=1500, 
        hotness_diff_threshold_base_relax_ratio=0.8, 
        hotness_diff_threshold_relax_ratio_penalty_rate=0.8, 
        hotness_diff_threshold_increment_relax_ratio=0.001, 
        hotness_diff_threshold_late_time_cap=0.35,
        hotness_diff_threshold_startup_cap=10,
        hotness_diff_threshold_recal_steps=0,
    '''
    planner = Planner(
        dataset="avazu",
        plan_path=AVAZU_PLAN_PATH, 
        data_path=AVAZU_DATA_PATH, 
        log_path=LOG_PATH, 
        cached_rows=AVAZU_CACHE_SIZE, 
        warm_up_steps=0, 
        search_limit=200, 
        hotness_diff_threshold_base_relax_ratio=0.8, 
        hotness_diff_threshold_relax_ratio_penalty_rate=0.8, 
        hotness_diff_threshold_increment_relax_ratio=0.001, 
        hotness_diff_threshold_late_time_cap=1,
        hotness_diff_threshold_startup_cap=12,
        hotness_diff_threshold_recal_steps=0,
        batch_file=BATCH_FILE_SUFFIX
    )
    
    dataloading_time = time.time() - start_time
    
    print("Planner initialized. Dataset loaded. (" + str(dataloading_time) + "s)\n[batch size = " + str(planner.batch_size) + "] Start planning...") # cache ratio = " + str(CACHE_RATIO) + ", 

    # baseline route
    # random_route = list(range(len(planner.batches)))
    # accumulated_cost = 0
    # num_loop = 10
    # for i in range(num_loop):
    #     random.shuffle(random_route)
    #     cost, _, _ = planner.Simulate_Cost(random_route, None, None, None)
    #     accumulated_cost = accumulated_cost + cost
    #     print("[random route " + str(i + 1) + "] cost = " + str(cost))
    # cost = accumulated_cost / num_loop

    # Planed route
    # cost, _, _ = planner.Heuristic_Search()

    # Grouped Search
    # cost = planner.Grouped_Search(10000)

    # Chunked Search
    # cost = planner.Segmented_Search(2000)
    # cost = planner.Multiprocess_Search(9600)
    # cost, _, _ = planner.Simulate_Cost(planner.plan, None, None, None)


    # Check cost
    # planner.Read_plan("training_plan-15-1024-1")
    # print("number of batches: " + str(len(planner.plan)) + ", calculating cost......")
    # cost, _, _ = planner.Simulate_Cost(planner.plan, None, None, None)
    
    '''------------------------ new functions ----------------------------'''

    # Convert dataset data from cudf.Series to np.ndarray
    num_batches = len(planner.batches)
    batches_id = list()
    batches_freq = list()
    for i in range(num_batches):
        batches_id.append(planner.batches[i].to_numpy().astype(np.int32))
        batches_freq.append(planner.freq_batches[i].to_numpy().astype(np.int32))
    
    print("Data have been converted into ndarrays on CPU.")

    '''------------------------ New cost calculator ------------------------'''
    # planner.Read_plan("training_plan")
    # print("number of batches: " + str(len(planner.plan)) + ", calculating cost......")
    
    # cost, _, _ = New_Simulate_Cost(planner.plan, planner.cached_rows, batches_id, batches_freq)
    # cost, _ = Non_LFU_Simulate_Cost(planner.plan, planner.cached_rows, batches_id)
    
    '''------------------------ New planner ------------------------'''
    # plan, cost, _, _ = New_Heuristic_Search(
    #     planner.log_path, 
    #     planner.cached_rows, 
    #     batches_id, 
    #     batches_freq, 
    #     warm_up_steps=planner.warm_up_steps, 
    #     search_limit=planner.search_limit, 
    #     hotness_diff_threshold_base_relax_ratio=planner.hotness_diff_threshold_base_relax_ratio, 
    #     hotness_diff_threshold_relax_ratio_penalty_rate=planner.hotness_diff_threshold_relax_ratio_penalty_rate,
    #     hotness_diff_threshold_increment_relax_ratio=planner.hotness_diff_threshold_increment_relax_ratio,
    #     hotness_diff_threshold_late_time_cap=planner.hotness_diff_threshold_late_time_cap,
    #     hotness_diff_threshold_startup_cap=planner.hotness_diff_threshold_startup_cap,
    #     )

    # Comment date: 20240401
    # init_plan_path = "/root/files/coding/data_loading_planner/taobao_run_1/live_backup-lfu--LFU-15-1024-4.json"
    # with open(init_plan_path, "r") as backup_file:
    #     init_plan = json.load(backup_file)
    
    # print("Initialized plan has been loaded.")

    # plan, cost, _ = New_None_LFU_Multiprocess_Search(
    #     LOG_PATH,
    #     planner.cached_rows, 
    #     batches_id, 
    #     warm_up_steps=planner.warm_up_steps, 
    #     search_limit=planner.search_limit, 
    #     hotness_diff_threshold_base_relax_ratio=planner.hotness_diff_threshold_base_relax_ratio, 
    #     hotness_diff_threshold_relax_ratio_penalty_rate=planner.hotness_diff_threshold_relax_ratio_penalty_rate,
    #     hotness_diff_threshold_increment_relax_ratio=planner.hotness_diff_threshold_increment_relax_ratio,
    #     hotness_diff_threshold_late_time_cap=planner.hotness_diff_threshold_late_time_cap,
    #     hotness_diff_threshold_startup_cap=planner.hotness_diff_threshold_startup_cap,
    #     num_process=40,
    #     init_plan=init_plan,
    #     )

    # Comment out date: 20240401
    # plan, cost, _, _ = New_LFU_Multiprocess_Search(
    #     LOG_PATH,
    #     planner.cached_rows, 
    #     batches_id, 
    #     batches_freq,
    #     warm_up_steps=planner.warm_up_steps, 
    #     search_limit=planner.search_limit, 
    #     hotness_diff_threshold_base_relax_ratio=planner.hotness_diff_threshold_base_relax_ratio, 
    #     hotness_diff_threshold_relax_ratio_penalty_rate=planner.hotness_diff_threshold_relax_ratio_penalty_rate,
    #     hotness_diff_threshold_increment_relax_ratio=planner.hotness_diff_threshold_increment_relax_ratio,
    #     hotness_diff_threshold_late_time_cap=planner.hotness_diff_threshold_late_time_cap,
    #     hotness_diff_threshold_startup_cap=planner.hotness_diff_threshold_startup_cap,
    #     num_process=40,
    #     init_plan=None,
    # )

    # plan, cost, count_steps = Count_Heuristic_Search(
    #     planner.log_path, 
    #     planner.cached_rows, 
    #     batches_id, 
    #     batches_freq, 
    #     warm_up_steps=planner.warm_up_steps, 
    #     search_limit=planner.search_limit, 
    #     hotness_diff_threshold_base_relax_ratio=planner.hotness_diff_threshold_base_relax_ratio, 
    #     hotness_diff_threshold_relax_ratio_penalty_rate=planner.hotness_diff_threshold_relax_ratio_penalty_rate,
    #     hotness_diff_threshold_increment_relax_ratio=planner.hotness_diff_threshold_increment_relax_ratio,
    #     hotness_diff_threshold_late_time_cap=planner.hotness_diff_threshold_late_time_cap,
    #     hotness_diff_threshold_startup_cap=planner.hotness_diff_threshold_startup_cap,
    #     )

    # Date: 20240612
    # init_plan = [20211, 21917, 8860, 13351, 2429, 13228, 15374, 15383, 3366, 22269, 15075, 34546, 2511, 7210, 21537, 1841, 15042, 1094, 9449, 16492, 20229, 22680, 3526, 8885, 2029, 6902, 9543, 13137, 19403, 4582, 16051, 27885, 21015, 17234, 9959, 5462, 15878, 21422, 18512, 546, 12282, 34011, 20505, 6322, 34532, 26348, 24582, 10659, 171, 8685, 30500, 20880, 27094, 25640, 23038, 7423, 33758, 23780, 21830, 17592, 30693, 34205, 30940, 19268, 3553, 33265, 3869, 4028, 16767, 7185, 19343, 12624, 910, 30430, 20225, 19828, 18729, 23669, 2458, 5646, 20258, 17844, 10438, 28584, 32679, 24565, 24091, 27617, 1166, 7700, 12591, 8508, 19851, 13778, 3316, 10834, 17065, 3853, 32255, 5735, 32249, 20761, 19204, 25062, 33475, 12138, 30777, 21387, 31896, 5772, 33046, 19096, 27568, 5825, 26593, 25384, 25696, 15315, 30625, 27156, 8185, 26478, 23019, 5112, 28529, 17638, 24013, 32643, 26204, 17901, 23912, 8639, 4457, 23242, 18350, 7228, 27425, 31818, 27504, 14036, 26646, 4607, 18516, 1570, 7299, 24986, 27400, 6150, 4339, 2457, 11508, 8342, 23555, 10265, 12191, 23786, 5983, 10588, 19153, 10134, 30841, 32917, 18566, 32129, 28086, 24347, 4690, 10603, 2296, 10720, 25708, 27934, 9949, 8375, 12317, 316, 7094, 9403, 10443, 14753, 28287, 19756, 11295, 8730, 11858, 24006, 27850, 32524, 30775, 23395, 26433, 4881, 361, 29309, 3875, 7081, 1328, 22208, 3326, 20285, 11230, 15013, 100, 15151, 29358, 14398, 2586, 31011, 17536, 23624, 6409, 2659, 17708, 15083, 1254, 6139, 33203, 20158, 30077, 26290, 33882, 4541, 5147, 31651, 34437, 6851, 5740, 21890, 4830, 2557, 30554, 7466, 25856, 5609, 10616, 33387, 12685, 15389, 20019, 30794, 15482, 33706, 33566, 9401, 12151, 6257, 15612, 14737, 24378, 32491, 15760, 30001, 16778, 1070, 15503, 21769, 8493, 22297, 15833, 29908, 10233, 16378, 8521, 27216, 3959, 17559, 4877, 21555, 21896, 4369, 324, 32724, 29364, 7690, 34, 15180, 10283, 33496, 32920, 8220, 9508, 19417, 26964, 18443, 23103, 13652, 23237, 6449, 12655, 12468, 20219, 13138, 13108, 30100, 34105, 22567, 26385, 14174, 28630, 20439, 29939, 17278, 24328, 5197, 12686, 6859, 21906, 10461, 8103, 30458, 10923, 15052, 31256, 9112, 27401, 32486, 19869, 31660, 25276, 18804, 12956, 10123, 28917, 20999, 18835, 12342, 30702, 23588, 33047, 7045, 1069, 15814, 14522, 17787, 8913, 30574, 9919, 20738, 31445, 6218, 7407, 11686, 5173, 28724, 15992, 21327, 23737, 20241, 25024, 22254, 10605, 17564, 13734, 31163, 24076, 16633, 23744, 6114, 3675, 24141, 27516, 6899, 24668, 23541, 31889, 10082, 34116, 26617, 21828, 32932, 26206, 32165, 20575, 9832, 7393, 13906, 24814, 20848, 28784, 31868, 6691, 5940, 14152, 16245, 13840, 32322, 4588, 10742, 21125, 21751, 3094, 4460, 3060, 18905, 26215, 2914, 17582, 18391, 21551, 19663, 16537, 8111, 684, 7791, 32696, 18057, 3947, 27865, 13887, 9879, 17960, 32267, 707, 18035, 11918, 1736, 3290, 28054, 26082, 1559, 11395, 8546, 24959, 10904, 7543, 14345, 21974, 26779, 31310, 32115, 29890, 24092, 12374, 12489, 32655, 23777, 27579, 4174, 16895, 25715, 3758, 24673, 1557, 12014, 14919, 8199, 9177, 16379, 21643, 34484, 20706, 3476, 8091, 31064, 10138, 17601, 6532, 6935, 6953, 25199, 21254, 17819, 25081, 26985, 31583, 2892, 20667, 28291, 632, 27916, 7051, 31168, 12460, 12476, 32910, 16451, 19151, 23363, 6998, 6937, 32304, 23893, 10859, 26914, 258, 19633, 3625, 23134, 7841, 28490, 12498, 8183, 6896, 24250, 3646, 20447, 31550, 1428, 33875, 10352, 32585, 16592, 4887, 21000, 29460, 2545, 8547, 6631, 25397, 2736, 21803, 28058, 20709, 34284, 3268, 9312, 11915, 578, 33032, 28263, 14755, 9760, 7792, 2857, 26938, 8844, 25583, 4026, 21796, 356, 6245, 28137, 32294, 31876, 18986, 18851, 28973, 16304, 23090, 10905, 30902, 7638, 33253, 15515, 10597, 14839, 5782, 9317, 14524, 23721, 11397, 29375, 13269, 15265, 7318, 3574, 33011, 20234, 15412, 20935, 13851, 6521, 5634, 2875, 25460, 24666, 4677, 19706, 23756, 16916, 28964, 25860, 29432, 34107, 6383, 30539, 33318, 21875, 7280, 9425, 6617, 18663, 20088, 768, 24974, 28466, 8003, 30391, 8491, 33978, 34290, 8834, 14010, 14809, 34539, 17519, 27969, 4292, 11076, 33469, 16442, 19650, 12534, 4089, 13984, 32523, 4739, 30979, 9446, 27711, 25822, 1212, 3931, 596, 13687, 4595, 28275, 3517, 20860, 20781, 21166, 1046, 24053, 30411, 32379, 3932, 13302, 28183, 9065, 18348, 15475, 4706, 10926, 2851, 7907, 17763, 30887, 26698, 30640, 3149, 20309, 21575, 5022, 15607, 19850, 23614, 25787, 16591, 16364, 25740, 14039, 16574, 12831, 33299, 11679, 13691, 14268, 14905, 33140, 13047, 9800, 25029, 21652, 26132, 9622, 15830, 1168, 20287, 30755, 3256, 8190, 18692, 25761, 14140, 12607, 22159, 7858, 15285, 15031, 23335, 3558, 10532, 26661, 12016, 16483, 28260, 12612, 18696, 27195, 8133, 9578, 7679, 4533, 20083, 22916, 29204, 11568, 6724, 2286, 25214, 17266, 2882, 6097, 29010, 14554, 25444, 33997, 22155, 8854, 2499, 26389, 25057, 1121, 5002, 2838, 23757, 12953, 1017, 11026, 19069, 6464, 25415, 5350, 1113, 33700, 10657, 31081, 24586, 34326, 20938, 12853, 1885, 5476, 1126, 19968, 32268, 34233, 22362, 18461, 26227, 18027, 30990, 20885, 31598, 23, 21855, 13250, 11917, 16520, 19708, 21490, 32168, 15240, 27699, 14592, 24119, 26116, 33958, 7346, 31367, 33920, 3092, 23746, 17832, 16967, 9342, 23123, 32944, 7052, 27751, 1713, 29498, 21523, 12081, 10641, 25110, 15359, 10531, 29414, 24979, 15060, 34347, 33879, 24115, 13252, 23161, 572, 5489, 29518, 10034, 19011, 9034, 1131, 16663, 27148, 29938, 19777, 16475, 31865, 31759, 10428, 32235, 4107, 30508, 13782, 22191, 3780, 7344, 24142, 4385, 16868, 33251, 33120, 24701, 26621, 1037, 20905, 13094, 18162, 8875, 6481, 22716, 376, 7578, 5562, 32382, 15773, 29530, 17627, 26809, 263, 12092, 4373, 7400, 5720, 9951, 5107, 11300, 16927, 26743, 23189, 2948, 19456, 7659, 9322, 28315, 8613, 2530, 31490, 15, 4812, 25358, 24573, 24340, 4720, 31657, 33348, 19863, 9056, 27520, 14383, 4611, 10900, 20235, 11526, 19054, 8986, 28506, 10343, 13318, 8690, 22670, 1723, 16271, 1560, 28822, 7822, 33202, 22015, 15254, 17510, 34009, 21477, 10483, 397, 24474, 14911, 1525, 21446, 27074, 2810, 12277, 29166, 733, 14949, 32295, 2955, 33969, 19782, 13182, 3754, 21631, 33219, 25205, 18435, 15767, 27845, 13811, 28239, 28593, 13974, 31595, 22901, 25409, 16965, 8471, 22334, 16970, 13150, 2329, 31548, 5130, 3704, 16006, 29320, 32471, 30087, 32690, 28135, 14125, 4682, 24500, 3583, 31765, 11446, 22682, 27701, 18728, 24003, 6041, 20876, 3396, 28347, 22293, 22400, 18879, 3320, 9186, 32265, 14745, 11571, 24004, 1898, 24956, 11540, 31706, 29813, 3889, 16736, 31519, 31411, 27644, 30359, 24272, 9106, 29926, 33467, 8121, 10121, 6405, 1693, 32365, 7805, 24334, 11752, 16109, 24623, 12773, 13622, 18680, 21556, 14467, 27837, 22840, 16093, 21255, 29095, 20398, 4928, 30271, 26092, 13531, 27654, 26706, 25343, 32879, 2951, 14605, 5299, 184, 4316, 23084, 16836, 11154, 20568, 15476, 27876, 34076, 11302, 25527, 4530, 21795, 10587, 22327, 1228, 34269, 11447, 2808, 3031, 26172, 26581, 23233, 8714, 11575, 14088, 15367, 21926, 2689, 16361, 1055, 24041, 19668, 10170, 14212, 32972, 22345, 24048, 10986, 32327, 18970, 7746, 28255, 8608, 33388, 6814, 14532, 16877, 6967, 11136, 25865, 9734, 31205, 10200, 4329, 7037, 6716, 33206, 11202, 21321, 3801, 11156, 11833, 11748, 32161, 11264, 3880, 20191, 31413, 20573, 25174, 9096, 34528, 16949, 10938, 28708, 16804, 3438, 1824, 1447, 18736, 12802, 23688, 9344, 11152, 22488, 6262, 4207, 3209, 10021, 23524, 34145, 25026, 8648, 16680, 32836, 14064, 34381, 25185, 17934, 21564, 27027, 4650, 14290, 31472, 7304, 5734, 33842, 17918, 3908, 23006, 24884, 24731, 16562, 17982, 30376, 21229, 408, 7478, 30871, 20112, 23269, 16944, 23558, 32022, 23140, 22241, 19840, 8395, 8319, 16693, 3887, 34081, 26209, 19647, 24464, 32500, 26565, 9758, 8888, 23969, 25625, 8010, 14326, 10366, 9359, 22913, 3269, 18089, 6052, 6558, 9023, 13995, 8741, 15293, 33107, 8723, 33368, 15092, 23909, 2020, 33516, 8683, 25113, 23596, 27593, 1309, 34535, 22288, 7720, 15497, 13059, 21333, 13363, 12540, 4708, 16446, 8450, 33531, 14357, 13788, 31317, 21169, 21027, 29548, 1307, 8470, 7518, 32829, 13444, 945, 22900, 24432, 7685, 32024, 2167, 15252, 33087, 629, 14178, 19970, 13593, 34390, 8614, 13285, 7983, 5425, 17129, 7463, 5432, 19286, 2877, 17770, 14092, 33827, 1941, 8455, 1585, 21088, 13870, 23385, 24471, 28884, 14145, 7737, 6507, 23730, 9423, 20981, 27683, 15329, 15518, 2109, 18265, 28293, 1820, 9705, 20206, 385, 26211, 9586, 22060, 5267, 7328, 19476, 19812, 25649, 25833, 12579, 29045, 10803, 4924, 32368, 18037, 22420, 7739, 30976, 1858, 33702, 26942, 14387, 29619, 3379, 19012, 2974, 12872, 30689, 13672, 28032, 19875, 10126, 28883, 29666, 24789, 24856, 17649, 19258, 3444, 18163, 31412, 27538, 24790, 29151, 27576, 27820, 8895, 1457, 7609, 10351, 34175, 17240, 8230, 19460, 7911, 26619, 20171, 8120, 30516, 18433, 10396, 15174, 14523, 20798, 2198, 29190, 32252, 12270, 32090, 3857, 22952, 5525, 12696, 4803, 26553, 31725, 18877, 12888, 927, 33804, 23270, 508, 19278, 5221, 23243, 10655, 26446, 24903, 22960, 2338, 25616, 7254, 17537, 1051, 29392, 878, 32070, 24241, 34025, 9583, 23383, 34345, 1890, 17279, 26509, 17253, 30435, 16306, 14740, 12426, 25922, 6957, 14593, 24575, 9920, 4895, 8970, 17955, 233, 20475, 15904, 1923, 2408, 7539, 30155, 25792, 696, 8328, 29302, 28033, 31537, 6831, 34309, 9840, 23711, 25473, 25223, 12461, 9222, 20271, 6475, 11794, 8114, 8761, 10399, 22614, 20501, 27144, 28457, 11922, 22385, 16190, 15714, 5929, 9995, 32310, 33223, 7698, 1541, 23900, 5353, 8631, 8387, 23932, 5620, 21450, 30972, 12153, 12610, 4272, 10957, 29812, 2983, 13430, 13227, 11157, 32626, 11989, 18800, 18174, 28673, 18375, 7914, 3040, 11876, 13819, 3294, 515, 12677, 8538, 22718, 24373, 8108, 34360, 4667, 10464, 8672, 10755, 22309, 18303, 4125, 20940, 24463, 10786, 8831, 18665, 33026, 20396, 8202, 18757, 4849, 25144, 27854, 17988, 1786, 19760, 34244, 7326, 32370, 24058, 3709, 22568, 14520, 11022, 20333, 5801, 5945, 6454, 25008, 22818, 9873, 4831, 2891, 12606, 21649, 3180, 29143, 16374, 24483, 20294, 4619, 23671, 15832, 7832, 8372, 21651, 26115, 27928, 4902, 10681, 11148, 17590, 25295, 28693, 5591, 3881, 12236, 8984, 23658, 13959, 14379, 28362, 13562, 28064, 7469, 19518, 1451, 23488, 8940, 27502, 21382, 28527, 294, 26972, 22730, 31942, 33820, 1568, 16737, 9030, 32253, 7272, 24528, 3391, 29829, 7661, 32459, 12657, 32506, 824, 1554, 11520, 4987, 29133, 27511, 8506, 11528, 16102, 5282, 14492, 28984, 15728, 28132, 5550, 10569, 3141, 17529, 30274, 551, 30089, 30866, 11976, 21636, 10739, 24659, 5691, 22691, 29054, 2801, 17418, 23234, 19104, 22448, 1480, 28861, 29928, 19274, 24769, 19715, 18018, 22732, 15544, 15689, 33652, 12593, 25322, 24897, 10869, 20322, 1100, 4188, 16370, 12988, 16470, 24127, 6782, 10093, 32621, 33344, 13083, 7265, 17512, 11570, 21020, 8082, 29549, 33594, 24490, 6038, 12218, 20883, 9875, 5017, 7766, 11850, 9870, 18470, 16648, 5636, 18908, 34095, 3714, 9729, 14749, 32594, 14875, 8256, 21284, 26421, 6199, 17754, 8452, 30660, 1719, 15017, 19378, 11207, 33509, 17672, 13018, 1659, 11558, 27115, 21066, 20831, 7492, 6646, 10202, 211, 5553, 15488, 25335, 1496, 19525, 13567, 14867, 30340, 24682, 17612, 11285, 6779, 14662, 31535, 30030, 8330, 21139, 4908, 31028, 22168, 33733, 30578, 30818, 18318, 33994, 8405, 32748, 27367, 15594, 6531, 17905, 29082, 31810, 7714, 19805, 33384, 22624, 26609, 15499, 23203, 18763, 18684, 34126, 30651, 1720, 22409, 7157, 13712, 30504, 33031, 26308, 7551, 22084, 12401, 17689, 7339, 21709, 3816, 3131, 984, 15242, 11838, 18168, 22992, 16810, 18595, 16647, 3103, 24343, 9522, 27932, 24153, 22459, 797, 24632, 32708, 31899, 3972, 4063, 3115, 27273, 17038, 2908, 1334, 24300, 31693, 6591, 21776, 16157, 24386, 12325, 21405, 1507]
    # 
    # plan, cost = Segmented_LFU_Multiprocess_Search(
    #     planner.log_path, 
    #     planner.cached_rows, 
    #     batches_id,
    #     batches_freq,
    #     planner.warm_up_steps, 
    #     init_plan, 
    #     None,
    #     None,
    #     None, 
    #     search_limit=200,
    #     hotness_diff_threshold_base_relax_ratio=0.8,
    #     hotness_diff_threshold_update_window=10,
    #     hotness_diff_threshold_startup_cap=12,
    #     hotness_diff_threshold_increment_relax_ratio=0.001,
    #     hotness_diff_threshold_late_time_cap=1,
    #     hotness_diff_threshold_relax_ratio_penalty_rate=0.8,
    #     num_process=60,
    #     )

    '''------------------------ Save the generated plan ------------------------'''
    # planner.plan = plan[:]
    # if PLAN_FILE_NAME is not None:
    #     planner.to_parquet(PLAN_FILE_NAME)

    '''------------------------ New (Old) multiprocess planner ------------------------'''
    # plan, cost = New_Multiprocess_Search(
    #     1200,
    #     planner.log_path,
    #     planner.cached_rows,
    #     batches_id,
    #     batches_freq,
    # )
    # planner.plan = plan[:]
    

    '''------------------------ New baseline ------------------------'''
    # random_route = list(range(len(planner.batches)))
    # accumulated_cost = 0
    # num_loop = 10
    # costs = list()
    # simulators = list()

    # for i in range(num_loop):
    #     random.shuffle(random_route)
    #     costs.append(Queue())
    #     # simulators.append(Process(target=New_None_LFU_Cost_Wrapper, args=(costs[i], i, random_route, planner.cached_rows, batches_id)))
    #     simulators.append(Process(target=Wrapper_Cost, args=(costs[i], i, False, random_route, planner.cached_rows, batches_id, batches_freq)))
    #     simulators[i].start()

    # for i in range(num_loop):
    #     simulators[i].join()
    #     cost = costs[i].get()
    #     accumulated_cost = accumulated_cost + cost
    #     print("[random route " + str(i + 1) + "] cost = " + str(cost))
    # cost = accumulated_cost / num_loop

    '''------------------------------- Convertor ------------------------------'''
    print("Start converting training plan to batch info...")
    input_path = os.path.join(AVAZU_PLAN_PATH, "training_plan.parquet")
    output_path = os.path.join(AVAZU_PLAN_PATH, "id_to_prefetch.parquet")
    Training_Plan_to_ID_of_Batches(input_path, output_path, batches_id, batches_freq)

    '''------------------------------- End ------------------------------'''

    planning_time = time.time() - dataloading_time - start_time
    # print("Cost: " + str(cost) + ", ")
    print("dataloading_time: " + str(dataloading_time) + ", planning_time: " + str(planning_time))


    
