import cudf as df
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
LOG_PATH = "/root/files/coding/data_loading_planner/avazu_run_7"
PLAN_FILE_NAME = "-new-7"
AVAZU_CACHE_SIZE = int(40428967 * 0.05) # int(9449205 * 0.05)
AVAZU_DATA_PATH = "/root/files/coding/RecSys-Training-Planner/DLRM/input/avazu_with_id/"
AVAZU_PLAN_PATH = "/root/files/coding/RecSys-Training-Planner/DLRM/input/avazu_with_id/training_plan/"

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
        self.plan = data['plan'].to_arrow().to_pylist()
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
            print(process_info + str(int((step - start_step) / ((num_steps - start_step) / 10))) + "0%: cost = " + str(cost_total))
        
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
            with open(os.path.join(log_path, "live_backup" + PLAN_FILE_NAME + ".json"), "w") as backup_file:
                json.dump(plan, backup_file)
        
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
    # # Read plan
    # print("Loading training plan from " + str(input_path))
    # start_time = time.time()
    # data = df.read_parquet(input_path)
    # training_plan = data['plan'].to_arrow().to_pylist()
    
    # Random plan
    training_plan = [i for i in range(len(id_batches))]
    random.shuffle(training_plan)
    if DATASET == "criteo":
        plan_path = DLRM_PLAN_PATH
    elif DATASET == "taobao":
        plan_path = TBSM_PLAN_PATH
    elif DATASET == "avazu":
        plan_path = AVAZU_PLAN_PATH
    else:
        raise RuntimeError("Unrecognized  dataset")
    df.DataFrame({'plan': training_plan}).to_parquet(os.path.join(plan_path, ("training_plan-random.parquet")))
    
    end_time = time.time()
    print("Training plan loaded. (" + str(end_time - start_time) + "s)")
    
    # Extract ids
    id_planed_batches = list()
    freq_planed_batches = list()
    for i in range(len(training_plan)):
        id_planed_batches.append(df.Series(id_batches[training_plan[i]], dtype='int32'))
        freq_planed_batches.append(df.Series(freq_batches[training_plan[i]], dtype='int32'))
    output = df.DataFrame({"id_planed_batches": id_planed_batches, "freq_planed_batches": freq_planed_batches})
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
    
    if init_plan is None:
        unused_batch = [i for i in range(num_batches)]
        init_plan = list()
    else:
        unused_batch = [i for i in range(num_batches) if i not in init_plan]
    random.shuffle(unused_batch)

    # split the remaining unused batches to sub-processes
    per_process_unused_batch = int(len(unused_batch) / num_process)                 # This may drop as many as the number of processes batches
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
    # import pdb; pdb.set_trace()
    # print("number of batches: " + str(len(planner.plan)) + ", calculating cost......")
    
    # cost, _, _ = New_Simulate_Cost(planner.plan, planner.cached_rows, batches_id, batches_freq, should_print=True)
    # # # cost, _ = Non_LFU_Simulate_Cost(planner.plan, planner.cached_rows, batches_id)

    # print("Cost: " + str(cost) + ", ")
    
    '''------------------------ New planner ------------------------'''
    '''
    init_plan = [34546, 14753, 21452, 324, 24006, 11193, 6150, 5787, 8508, 27699, 10438, 1121, 29593, 26052, 29797, 32255, 11520, 6935, 26097, 26247, 16922, 20285, 29425, 27934, 3704, 22325, 12988, 23661, 7690, 6464, 15066, 8546, 2884, 28287, 11275, 741, 14061, 12220, 30043, 31901, 6896, 17691, 34126, 17620, 15987, 26417, 10904, 16364, 28291, 4671, 7032, 28984, 13058, 18970, 30651, 16754, 15060, 32605, 19343, 23304, 23042, 30614, 8120, 30926, 1665, 5130, 19395, 11724, 18497, 17256, 21739, 10906, 3336, 28255, 25170, 15473, 13328, 2752, 31543, 957, 17898, 7833, 32196, 21890, 19153, 8895, 13308, 31741, 25223, 16051, 31375, 21828, 11183, 2302, 33920, 4765, 21066, 34, 4369, 27505, 29569, 11604, 25153, 16693, 2659, 11446, 18525, 3853, 14256, 25214, 13364, 23624, 15799, 25456, 21863, 28263, 15324, 365, 31519, 34053, 5437, 17601, 8978, 23150, 22313, 10768, 3238, 15992, 23553, 24352, 11575, 1940, 1559, 21297, 17885, 15240, 10735, 20288, 15052, 18628, 14602, 22155, 25860, 16161, 28955, 15875, 14662, 3974, 20171, 9765, 5278, 27400, 6532, 24856, 32370, 4677, 11784, 20431, 33700, 16178, 10923, 22822, 6779, 20098, 19869, 12648, 23148, 3149, 8883, 20841, 11904, 8762, 7681, 7265, 609, 19756, 11679, 28630, 13782, 25057, 24598, 20706, 13107, 16024, 17233, 14462, 32322, 6705, 22182, 25062, 21985, 1505, 9578, 3227, 8337, 8183, 2494, 4243, 28260, 12802, 10343, 9610, 31535, 13939, 25322, 8761, 27490, 10004, 30768, 27432, 24048, 25126, 2429, 1725, 8035, 15170, 6431, 18753, 33181, 32664, 33945, 15136, 25289, 25625, 7524, 8957, 30246, 22309, 15639, 749, 10294, 12476, 17312, 8590, 7525, 19862, 27401, 21695, 27568, 20830, 8921, 2403, 29156, 14488, 12942, 10597, 12772, 12579, 18314, 9031, 26940, 32133, 11911, 32198, 16735, 14994, 29579, 18856, 25992, 22362, 3289, 5740, 2966, 3121, 5745, 12021, 5197, 32679, 15515, 11241, 9403, 13393, 13861, 30693, 14259, 9012, 18533, 28086, 1787, 26140, 27617, 7675, 27205, 25640, 8875, 3138, 20889, 21000, 21537, 15138, 17634, 13, 6086, 11238, 5932, 16778, 13356, 10842, 10701, 6114, 21467, 4872, 12119, 2511, 27137, 7018, 19340, 19647, 27510, 17354, 19320, 369, 15797, 33026, 14095, 11276, 17408, 12542, 18516, 22818, 8491, 4089, 18684, 23698, 5464, 6573, 15114, 29010, 14419, 21155, 8730, 10035, 2557, 7242, 10178, 770, 33978, 20654, 23669, 12399, 878, 1100, 18731, 10657, 15719, 25856, 9919, 9620, 21654, 17257, 25526, 3985, 32379, 25708, 3035, 32580, 32496, 5267, 13694, 19460, 26621, 25473, 14088, 26083, 31445, 34116, 6724, 632, 26591, 16067, 2005, 25199, 22045, 22031, 1723, 31317, 31693, 23140, 33509, 31865, 20767, 8918, 6398, 18692, 7400, 30554, 12239, 8033, 25047, 20235, 7778, 5772, 9311, 18472, 14749, 1258, 18625, 3714, 19169, 5940, 2626, 22718, 33206, 20047, 18512, 9832, 21321, 16064, 29700, 30902, 5495, 26136, 23363, 30155, 25571, 6111, 29320, 31896, 31550, 21936, 31047, 13076, 21386, 27645, 6872, 30014, 10516, 24480, 6635, 24887, 24565, 15979, 2489, 2329, 13571, 12631, 30235, 28856, 6875, 29769, 30980, 5609, 1592, 2144, 14875, 15959, 7005, 6828, 8008, 21141, 3251, 27882, 5825, 28650, 33733, 31509, 17536, 16658, 33239, 32892, 18650, 4650, 3861, 21974, 10217, 7081, 16884, 18532, 10474, 3115, 33974, 23651, 9873, 1330, 12153, 13153, 21569, 32214, 14463, 27442, 17529, 32269, 6693, 1671, 10875, 12216, 9417, 1944, 30777, 2237, 17085, 25548, 356, 7501, 16093, 18120, 18665, 8114, 18131, 23103, 34290, 1273, 7540, 13354, 8185, 3432, 4848, 31687, 7042, 23620, 28499, 22366, 14125, 14073, 15044, 30391, 13785, 15321, 11247, 1291, 13285, 16247, 28465, 30445, 25649, 14663, 30979, 17308, 692, 14509, 30002, 25081, 3729, 13137, 15150, 10834, 9344, 2327, 5510, 5173, 13680, 24250, 25415, 28991, 3326, 902, 22926, 21855, 21128, 9800, 6245, 19911, 20798, 5550, 16558, 21952, 15820, 9124, 20780, 8493, 1185, 28658, 24267, 26228, 22290, 23243, 28458, 11915, 32964, 30293, 20935, 11786, 26606, 18222, 2481, 23912, 14233, 20510, 33804, 4085, 29619, 16184, 24029, 16505, 11295, 6173, 32226, 12558, 3094, 25532, 11009, 20896, 12016, 27061, 21099, 30955, 33255, 28498, 23595, 11395, 3312, 16059, 25684, 3801, 31868, 543, 16393, 8143, 23588, 33451, 2857, 1585, 6549, 20689, 1957, 33702, 25008, 1352, 4352, 13695, 13851, 19528, 32250, 7350, 32471, 14398, 13040, 6416, 5432, 12624, 17696, 3881, 5712, 16868, 3316, 26913, 6758, 30789, 16429, 9389, 4708, 18177, 22716, 15072, 12593, 18111, 28026, 23006, 25687, 32851, 29993, 20534, 16970, 25514, 21015, 2108, 25715, 25112, 20751, 22004, 6639, 33475, 32256, 32459, 31765, 8091, 15011, 16944, 18855, 31272, 21523, 33013, 31660, 4107, 2776, 20917, 21721, 6851, 33706, 9725, 20803, 2152, 11822, 26650, 12503, 5221, 23269, 11455, 3749, 554, 23395, 3578, 20407, 27738, 33688, 27837, 21692, 33632, 5062, 15042, 4588, 3175, 2769, 7723, 15415, 22234, 26082, 28784, 28930, 2545, 12161, 30774, 31512, 20518, 11942, 22722, 15227, 11753, 8885, 977, 26478, 29784, 10998, 30226, 27690, 5398, 5111, 18673, 28293, 29498, 26240, 3583, 1096, 19778, 3290, 3576, 19054, 7659, 1229, 2118, 21002, 21649, 12282, 24313, 5474, 7052, 29594, 6332, 16690, 2426, 3907, 12217, 6471, 17201, 20906, 17713, 27695, 7700, 26206, 26186, 16570, 13294, 9951, 10506, 19567, 12628, 11029, 12735, 16858, 13777, 4533, 16804, 18168, 30981, 3513, 4454, 12336, 1591, 32249, 19223, 28973, 8532, 23961, 8073, 8400, 17627, 21652, 29249, 27865, 29548, 10659, 20219, 31943, 15174, 28457, 12655, 8387, 10443, 23693, 25113, 3526, 15212, 20936, 2586, 16451, 20577, 29513, 9449, 1882, 25757, 32932, 18908, 26619, 34323, 12151, 5002, 32765, 15365, 2565, 32327, 11433, 7638, 24884, 30371, 3268, 14166, 25385, 33940, 10945, 19268, 5147, 26659, 6019, 24775, 17988, 12324, 6433, 13228, 26021, 10754, 30775, 10352, 2456, 14495, 7437, 24175, 2909, 27146, 3078, 27433, 30660, 33250, 29626, 24790, 29137, 24582, 28785, 3089, 15265, 1804, 20019, 8270, 13596, 22911, 19633, 6511, 23722, 34503, 13612, 27906, 14919, 28058, 30887, 15692, 27763, 22735, 14624, 8968, 33614, 34437, 32486, 7859, 18763, 13771, 6139, 14276, 16388, 18162, 7720, 6459, 1254, 33169, 7466, 14208, 29799, 7578, 20999, 6887, 2696, 28239, 11764, 28833, 20848, 5167, 30121, 30858, 6841, 30976, 23768, 10248, 20395, 13483, 28450, 9223, 13776, 9522, 18184, 30107, 14178, 13995, 24409, 29117, 27334, 24817, 2955, 32242, 5171, 11372, 15607, 16663, 10713, 21017, 5688, 464, 8525, 33914, 25016, 22551, 27885, 32986, 21776, 22948, 3612, 18366, 25790, 31345, 28854, 15410, 16245, 3780, 28782, 12888, 9879, 9065, 4886, 21573, 18039, 27504, 32543, 34143, 2206, 11686, 9425, 7895, 19579, 3446, 17890, 1994, 20006, 18325, 3297, 28970, 23505, 16186, 2986, 31069, 33474, 32533, 22241, 329, 5937, 16758, 19264, 24378, 27085, 2157, 34095, 32165, 29882, 15488, 9013, 29125, 29190, 27141, 33999, 14140, 1507, 385, 11716, 11976, 2351, 676, 24666, 16106, 25595, 1489, 3320, 688, 32491, 24127, 19782, 22345, 14039, 25689, 4560, 29726, 8639, 27928, 18293, 20302, 28042, 5508, 12869, 8470, 17234, 9401, 24907, 8844, 1131, 13028, 6967, 30396, 3476, 13554, 12607, 2747, 14254, 3196, 6937, 1314, 16967, 9654, 20477, 25444, 1328, 17808, 15601, 1616, 17960, 29549, 16353, 7199, 15359, 16415, 12373, 12138, 29297, 31774, 14051, 4782, 10632, 4928, 2636, 13691, 13552, 263, 21159, 31448, 1401, 15974, 28516, 17918, 33469, 1924, 17065, 30192, 615, 1967, 2281, 18829, 6152, 13182, 3092, 4885, 7952, 28549, 9001, 30310, 24974, 791, 22015, 29056, 9528, 21391, 17787, 7979, 12545, 16198, 28576, 4648, 525, 9896, 34007, 15079, 4343, 5896, 18825, 3368, 29939, 6030, 21973, 17266, 3932, 31285, 10588, 14885, 1824, 30734, 12764, 20234, 5215, 1168, 20709, 29812, 29868, 15423, 3371, 23592, 2323, 8727, 1245, 8894, 29937, 8220, 15412, 11013, 11359, 9196, 1858, 16483, 10606, 20183, 9342, 31310, 24991, 30967, 19378, 25923, 34093, 3745, 12853, 25110, 18129, 3192, 17084, 29638, 1991, 1992, 23563, 32260, 4023, 20031, 17194, 16616, 23182, 15838, 14080, 13083, 24574, 23474, 15275, 16179, 22682, 25175, 6871, 25459, 4385, 18930, 7534, 3180, 9534, 19329, 13840, 8332, 29375, 22401, 22400, 5525, 28223, 22960, 21769, 30539, 2204, 9384, 25777, 16928, 22135, 25205, 12029, 15878, 19476, 24348, 4158, 32213, 20684, 5646, 34026, 17655, 23235, 21575, 9616, 16378, 26565, 20347, 3507, 34474, 11385, 25335, 31578, 3517, 23940, 7823, 2663, 29112, 9348, 22269, 15495, 32448, 30068, 13797, 27796, 10742, 29530, 4739, 13214, 9211, 19496, 16977, 1987, 8964, 32950, 29060, 26119, 23305, 572, 25227, 4199, 34244, 6910, 11750, 6998, 3931, 8442, 19712, 28527, 30837, 10170, 17124, 21125, 13965, 16592, 19812, 29693, 12569, 9383, 34367, 13887, 9447, 26368, 30336, 5720, 29039, 24097, 19804, 24660, 28311, 1457, 31791, 7463, 8986, 25424, 18982, 3561, 33812, 15569, 12106, 7070, 24115, 15850, 4800, 23259, 26421, 2300, 22391, 652, 13811, 13262, 5795, 28064, 16190, 9042, 1441, 8947, 4415, 4263, 27820, 21446, 7983, 8256, 192, 23908, 31725, 13514, 14900, 23435, 28165, 31029, 12804, 11076, 8527, 1166, 23766, 18550, 12132, 33318, 15518, 22440, 1890, 23220, 30718, 16304, 18880, 18027, 11721, 566, 7140, 28234, 21646, 14839, 25579, 23969, 1480, 4484, 2954, 21563, 21203, 30972, 578, 33524, 10605, 13350, 27135, 12612, 5088, 24467, 25933, 11642, 25666, 34347, 33594, 32191, 17503, 4790, 12325, 23801, 16370, 18736, 31778, 21779, 24328, 20726, 5036, 21292, 24979, 28306, 16401, 24058, 21327, 2967, 19273, 19114, 5633, 6257, 9060, 13562, 14344, 28599, 2853, 1428, 22293, 27429, 6615, 12832, 12657, 24340, 23931, 16447, 25716, 33875, 13039, 21631, 2880, 30510, 2421, 9744, 33827, 33737, 9268, 10886, 4979, 28884, 13932, 10981, 34133, 28605, 8358, 26178, 15759, 26875, 33879, 3784, 24684, 33368, 1024, 15833, 22442, 15857, 12946, 13944, 22417, 31936, 28626, 11080, 28712, 24294, 13597, 23075, 31615, 24024, 3430, 17644, 28765, 11570, 11208, 25302, 11173, 34206, 12915, 9145, 22836, 30518, 23670, 32748, 19224, 18255, 16668, 14412, 24874, 20565, 9446, 22840, 23790, 34100, 20348, 7608, 13974, 14480, 21255, 27456, 28249, 15594, 15873, 1005, 11141, 26364, 15681, 3921, 12191, 21643, 33518, 14226, 2385, 18301, 16916, 6657, 4028, 34284, 12407, 26581, 25343, 33520, 28074, 21747, 15568, 3131, 8970, 22887, 9153, 9690, 8608, 16289, 18454, 5191, 886, 3709, 12602, 20412, 31142, 20703, 6256, 13138, 412, 775, 22901, 9172, 945, 9322, 895, 22962, 30702, 10399, 7623, 8434, 23242, 12147, 15842, 10034, 16442, 27445, 16817, 29912, 5929, 3758, 13431, 25647, 20083, 10372, 27103, 20126, 29537, 27312, 7185, 10743, 17725, 5376, 21619, 15728, 15132, 6812, 4891, 9719, 843, 20910, 6606, 23102, 32894, 7393, 32070, 26391, 8190, 3262, 13287, 31490, 15937, 4063, 1568, 15065, 17139, 12128, 32478, 31548, 10549, 5060, 24711, 16700, 30294, 3369, 32022, 34360, 18886, 13094, 1637, 1130, 25243, 10199, 32639, 24871, 123, 16706, 11989, 11854, 5562, 16537, 21084, 2699, 14790, 18687, 29302, 1270, 22173, 1471, 1069, 28810, 19530, 4460, 13962, 8386, 17785, 40, 3812, 29652, 28149, 11285, 30742, 14727, 25029, 10661, 14671, 31019, 3256, 5945, 11339, 13012, 9492, 4433, 12834, 29024, 33943, 26512, 684, 10265, 24559, 11725, 33698, 17444, 1572, 34539, 27074, 7423, 31101, 14205, 8119, 8199, 2333, 19069, 4908, 16435, 9320, 1299, 30385, 87, 5358, 34030, 16233, 111, 5501, 17502, 31587, 9583, 29323, 12992, 4582, 4924, 30077, 6982, 19793, 23596, 9220, 1094, 34104, 21343, 14373, 24123, 12014, 31263, 1639, 15285, 24141, 31584, 13792, 31818, 13553, 30755, 30080, 24573, 31064, 18719, 16374, 17872, 10725, 28286, 28108, 15534, 28592, 26352, 32382, 30601, 24931, 12560, 18348, 20568, 26710, 322, 28121, 15370, 27345, 31636, 4360, 11220, 15034, 12610, 27502, 4793, 1049, 4264, 25518, 23239, 8863, 11322, 21896, 25439, 28708, 7609, 27104, 30089, 9840, 19026, 29999, 30750, 30100, 10401, 16927, 32743, 5979, 13227, 24043, 29630, 31576, 31889, 26585, 27908, 11684, 17762, 5489, 10182, 30937, 28051, 29275, 2109, 14605, 24673, 10025, 30046, 19152, 11432, 23688, 25696, 56, 14471, 17910, 25045, 27850, 26454, 20332, 27132, 9177, 3905, 15822, 30087, 7299, 14014, 20664, 7193, 19760, 32290, 17832, 29908, 27961, 6449, 11065, 5240, 17876, 9130, 17282, 28470, 17709, 11898, 18905, 31815, 14390, 8202, 6373, 9215, 8429, 23060, 14879, 6617, 5397, 20753, 22207, 2801, 22940, 34528, 7284, 11365, 1810, 12101, 27301, 21201, 10492, 3810, 34506, 15252, 15824, 21339, 25363, 445, 17111, 6951, 508, 2921, 33098, 31720, 6521, 147, 5253, 17177, 8192, 6641, 32574, 26389, 31072, 13734, 4859, 19785, 19090, 7756, 27624, 5350, 32616, 27195, 9078, 19120, 16626, 105, 20447, 19518, 11071, 1826, 8375, 30879, 2457, 1901, 8888, 1923, 12767, 24013, 11558, 34153, 12790, 8716, 3616, 15760, 30990, 18757, 7985, 33047, 16783, 1307, 32109, 3625, 9490, 4374, 12451, 23376, 14326, 30794, 32027, 25163, 22298, 5324, 29115, 30851, 4188, 31910, 27491, 5540, 27854, 16843, 33243, 17941, 13788, 16714, 28724, 15489, 8538, 7737, 6538, 28372, 16068, 18399, 3606, 10942, 14263, 18350, 15821, 22414, 20394, 23780, 27156, 14842, 32324, 11833, 9485, 714, 7139, 5179, 25397, 16583, 18769, 26143, 1955, 32055, 33066, 7732, 27273, 19968, 26433, 34348, 11880, 3129, 28442, 21477, 17284, 27639, 15140, 30113, 32314, 20402, 1046, 2232, 22222, 29561, 6357, 31651, 15360, 14174, 24810, 13876, 5693, 5112, 5674, 33779, 19628, 18391, 27560, 12953, 11100, 15372, 14384, 34371, 3857, 32731, 27914, 12211, 24003, 34312, 18851, 7249, 33969, 22199, 16246, 29890, 27305, 23447, 1051, 19349, 25551, 34105, 26688, 22385, 33602, 29926, 2641, 9949, 26996, 15587, 33205, 15446, 14121, 3064, 27846, 5225, 2606, 33299, 10159, 27876, 27190, 13672, 13193, 18013, 22665, 28356, 1903, 9521, 28839, 19441, 32419, 23394, 31876, 5715, 20112, 27642, 9508, 18974, 19081, 9023, 13420, 30558, 408, 343, 14971, 32252, 29567, 29432, 5465, 10155, 21007, 5340, 19880, 14269, 31039, 1515, 15726, 24631, 22670, 1473, 12225, 33687, 4987, 24217, 27407, 33919, 14467, 59, 17046, 25238, 32524, 10396, 30262, 30520, 6224, 4203, 7517, 238, 32142, 13437, 6704, 31173, 4619, 12837, 28019, 3774, 14520, 21101, 31880, 15725, 31456, 20575, 15503, 18704, 12136, 30085, 12568, 26508, 19219, 10731, 33134, 30867, 1287, 10398, 9959, 24774, 17607, 24690, 31078, 21146, 1557, 23, 27516, 10578, 9106, 25622, 8614, 24202, 10428, 15612, 19964, 14138, 31630, 2349, 19795, 33796, 20026, 28691, 11897, 9340, 20365, 5631, 5759, 1176, 5222, 22557, 29518, 2891, 25024, 7286, 33994, 4071, 22159, 12881, 2689, 28893, 1426, 26804, 30435, 29589, 11300, 6099, 6330, 12460, 8473, 23737, 30914, 26638, 8267, 33566, 16810, 15549, 6345, 3947, 27577, 8977, 16941, 4864, 13430, 22739, 28506, 24091, 23521, 12040, 31810, 7547, 29677, 7051, 10952, 17400, 7358, 51, 23777, 6953, 16917, 8241, 1154, 6916, 33601, 31259, 25015, 9655, 30940, 21381, 25792, 20081, 11421, 9600, 11710, 18116, 21206, 1302, 5891, 11579, 19231, 25295, 21490, 10215, 29414, 12237, 250, 19106, 8426, 7272, 1750, 5120, 10434, 24789, 21917, 7585, 12177, 26104, 24940, 20290, 11752, 23356, 5023, 32368, 27106, 27964, 27946, 10300, 24101, 10916, 19972, 15634, 17745, 797, 21712, 20020, 17816, 6213, 9687, 22426, 23671, 33008, 28093, 28362, 12426, 6814, 233, 5454, 23836, 19663, 34332, 996, 16989, 18720, 27289, 17590, 23163, 20625, 4439, 3991, 30809, 12378, 31761, 1555, 5022, 28570, 3685, 31603, 14221, 26493, 9716, 8111, 11144, 23837, 2151, 29800, 30248, 18721, 31942, 9410, 17500, 33504, 33426, 33522, 25787, 27708, 27978, 16553, 24694, 34009, 402, 12638, 13402, 9731, 26664, 2328, 20589, 6588, 21086, 3734, 24173, 478, 6022, 7842, 19246, 10729, 12880, 15341, 5325, 24521, 34306, 16327, 29557, 14991, 1928, 22864, 13101, 34326, 33758, 22257, 32129, 29421, 4226, 21254, 9063, 22981, 8874, 18163, 30375, 24962, 5735, 19706, 32836, 11895, 10859, 6646, 15705, 21108, 19445, 31920, 8521, 10408, 33648, 492, 13779, 17211, 1877, 12714, 17003, 21765, 34449, 5472, 12527, 21817, 22328, 30548, 19509, 21895, 26518, 23744, 28259, 7497, 33168, 16555, 29124, 24488, 30409, 11097, 26120, 19011, 15344, 32714, 20211, 551, 26928, 5620, 22193, 32268, 6874, 21193, 28986, 4447, 7504, 27398, 23528, 24103, 14592, 1676, 24148, 3664, 22690, 27066, 15015, 7743, 27496, 30060, 2623, 27711, 7382, 13318, 8963, 17770, 19650, 5055, 25475, 9138, 20247, 13652, 23833, 20912, 27134, 10469, 27916, 29605, 33810, 10786, 11264, 19314, 10934, 4603, 28964, 10253, 8683, 22678, 22208, 3740, 16591, 26002, 7984, 10483, 1866, 5974, 9041, 23274, 20705, 15830, 29533, 28339, 7551, 12134, 14797, 19467, 29426, 7811, 11492, 8953, 28862, 18476, 21602, 27786, 17897, 8854, 15122, 33846, 12871, 11622, 3967, 7111, 8834, 12209, 22668, 5146, 11447, 10311, 11150, 7933, 7792, 553, 1570, 22389, 28226, 23377, 22009, 2445, 20864, 15424, 14540, 3230, 1513, 32733, 14460, 14901, 15274, 12115, 30516, 28175, 897, 24849, 32294, 10306, 24258, 27799, 28118, 10913, 19999, 5381, 5636, 11656, 34079, 3668, 28752, 14010, 1156, 23902, 25522, 3730, 12957, 22412, 12886, 21949, 11050, 13166, 17423, 7907, 34081, 16271, 6591, 2178, 25477, 6806, 17383, 9002, 24456, 26210, 31672, 13875, 33328, 27929, 9763, 15913, 28137, 7679, 21518, 32690, 23612, 20791, 31213, 29994, 9727, 12668, 18955, 24805, 10499, 14186, 6865, 22495, 28564, 1048, 23709, 26280, 11751, 16988, 13736, 20778, 6647, 19850, 7752, 13543, 9491, 28518, 31641, 18729, 20241, 5357, 12270, 29873, 25190, 28373, 17126, 19715, 23133, 13927, 14402, 34011, 10624, 5107, 3908, 21847, 1905, 10603, 26754, 28183, 16636, 7478, 3971, 31300, 17324, 5500, 18582, 2847, 26237, 11397, 9531, 25366, 1661, 31589, 24623, 24270, 27637, 20640, 20627, 33661, 6217, 14152, 22356, 16530, 25202, 4325, 27632, 14379, 27747, 30689, 22317, 31496, 9173, 10594, 31838, 10250, 7255, 460, 3366, 8831, 33355, 16479, 25088, 29239, 18836, 31938, 27025, 22026, 11922, 28579, 7542, 20797, 32730, 1247, 19278, 12359, 28594, 2948, 21580, 17956, 20725, 943, 21677, 10286, 18549, 1199, 3849, 22169, 27849, 15271, 11412, 11553, 27816, 2099, 3074, 32983, 16189, 1495, 9343, 5843, 3041, 567, 32888, 10861, 14389, 25740, 11007, 27805, 25267, 33027, 1712, 5015, 12310, 28071, 26268, 33689, 10016, 7533, 8663, 24642, 25806, 28277, 4676, 18781, 33035, 12154, 109, 34022, 1087, 32323, 15951, 5382, 18035, 13620, 2864, 18061, 12236, 27724, 7947, 17013, 9854, 5065, 8685, 1793, 15069, 12293, 5587, 1541, 24586, 20314, 13892, 5917, 3381, 14867, 18595, 10026, 7320, 16913, 25682, 31988, 22254, 17453, 20600, 19306, 24092, 25160, 22746, 24708, 34381, 11800, 16047, 6885, 23244, 9214, 20531, 3002, 18943, 9718, 25026, 307, 7559, 11477, 19822, 10712, 28114, 25635, 12421, 25937, 8993, 227, 17512, 33031, 3621, 13780, 18215, 3809, 6873, 5634, 26077, 28357, 13208, 33521, 29664, 6946, 9540, 21502, 26657, 14432, 13323, 32600, 5880, 2944, 24596, 6507, 5850, 30317, 24778, 22935, 24571, 26132, 3209, 19164, 5157, 20138, 6640, 10233, 32378, 21040, 10276, 18342, 1594, 15751, 24358, 28025, 978, 1916, 28609, 27372, 26227, 13576, 33986, 15031, 22597, 18435, 20261, 24468, 24432, 24185, 19654, 29955, 31059, 13445, 22387, 8405, 10777, 15018, 30535, 2532, 1945, 16621, 28758, 23640, 9314, 16717, 23682, 23245, 19389, 10852, 21250, 5912, 23666, 27778, 22264, 3489, 26199, 26308, 31806, 20497, 2959, 16919, 26101, 7591, 29892, 12137, 26814, 27822, 18717, 21687, 1656, 19003, 8529, 31846, 1364, 4272, 16317, 24710, 1674, 22680, 34129, 12498, 20012, 21653, 31256, 23027, 30098, 17513, 20287, 5666, 6437, 32862, 25410, 18977, 8351, 14558, 4163, 28352, 27735, 8954, 999, 7658, 20593, 26914, 23739, 17178, 4830, 14776, 6979, 34361, 6623, 4840, 17028, 12146, 32917, 30142, 4630, 2784, 27675, 29372, 18355, 6796, 21764, 15257, 6038, 9287, 7780, 13246, 8613, 18631, 17198, 3691, 19721, 10616, 19096, 1341, 14431, 9776, 19942, 32900, 32171, 27835, 21548, 4157, 26839, 12312, 2081, 23786, 25212, 15832, 22842, 1771, 27333, 9432, 15773, 26729, 30419, 2423, 16464, 8774, 26662, 17407, 4001, 22001, 19599, 31999, 19824, 21594, 23123, 30685, 24500, 26669, 22137, 6690, 12515, 16163, 14183, 28037, 30543, 27565, 21622, 24471, 10387, 17671, 2680, 5393, 8010, 34119, 19209, 14717, 14747, 12782, 15445, 9026, 25608, 7431, 2467, 26970, 16781, 8343, 29725, 22900, 5410, 32972, 24410, 4834, 21180, 8040, 29490, 12518, 6260, 14734, 19714, 30700, 11407, 25853, 15384, 27047, 16403, 6422, 22606, 6944, 32416, 32021, 20377, 30956, 10261, 25147, 13767, 30969, 7632, 23090, 21400, 14247, 34456, 3343, 7600, 11174, 12990, 23746, 24646, 681, 4098, 30225, 14584, 12893, 695, 16263, 10200, 1841, 12302, 6443, 31536, 22941, 11845, 24464, 33162, 23012, 2313, 18907, 18566, 14150, 17048, 22191, 30863, 28327, 18691, 17775, 175, 15801, 2106, 27191, 280, 18036, 27086, 28867, 16476, 4541, 22542, 33658, 3477, 30567, 19199, 23335, 9021, 19668, 34336, 5124, 29622, 34056, 7346, 23280, 26017, 10395, 21902, 3344, 4469, 31488, 22139, 23930, 34073, 8749, 16556, 28664, 2038, 1986, 2301, 6024, 27900, 34395, 5283, 5630, 696, 33985, 12846, 28, 11367, 13558, 20707, 26030, 7954, 25955, 13351, 20592, 447, 22005, 7791, 21803, 1486, 5824, 15300, 18418, 15900, 3240, 17917, 15083, 25805, 11350, 8511, 13531, 27788, 21167, 22075, 26903, 24452, 8211, 30992, 29132, 29851, 22646, 3019, 31681, 34107, 32655, 30091, 21512, 24576, 14954, 31378, 30962, 31076, 24543, 20079, 12118, 18354, 24968, 24343, 3869, 22803, 12219, 33955, 27538, 31598, 7248, 32763, 16680, 8506, 3073, 10933, 596, 3807, 13755, 31847, 14523, 11991, 33310, 18259, 20728, 11775, 4269, 7267, 4498, 6254, 14807, 28359, 26235, 11713, 6113, 8444, 31030, 25392, 7518, 5832, 4771, 24919, 25858, 27353, 24397, 9710, 23536, 18574, 4335, 23838, 6415, 17948, 33660, 28749, 6551, 14521, 30788, 23233, 31905, 14434, 16912, 11184, 22919, 6691, 8342, 3156, 18968, 7148, 1554, 17904, 29460, 28343, 27498, 5914, 32412, 33337, 17819, 31393, 33690, 20139, 32619, 3567, 1577, 20667, 24632, 29898, 1374, 23044, 23966, 15457, 29389, 702, 2029, 2763, 9606, 4905, 19286, 33419, 25148, 25568, 4720, 30827, 28328, 26321, 25866, 22683, 11480, 25172, 2287, 10528, 31081, 7693, 15400, 21333, 15635, 25050, 27325, 22141, 31618, 8904, 1643, 26985, 10214, 24334, 18253, 21536, 14755, 10084, 1496, 2174, 13263, 8258, 30138, 488, 18089, 1212, 18002, 14549, 3945, 8554, 11027, 20715, 29666, 22277, 31671, 7055, 10929, 10727, 14005, 5244, 33967, 30574, 43, 23005, 32181, 32786, 7599, 26896, 33290, 4902, 31353, 15563, 3409, 4080, 22830, 20581, 34205, 3384, 4435, 29445, 29097, 15714, 12510, 7972, 6531, 34094, 22758, 12277, 22995, 23061, 17380, 18433, 33, 25934, 28414, 18783, 27007, 31083, 24981, 22353, 27096, 18947, 23657, 25046, 6811, 25558, 4159, 31273, 28146, 1418, 26108, 8757, 23753, 33077, 10295, 28574, 28045, 24137, 20439, 13810, 15020, 27907, 10461, 7962, 5205, 11578, 23949, 25736, 22392, 17726, 21788, 1597, 29341, 3379, 13812, 22409, 14057, 34497, 1339, 22614, 16794, 4791, 7288, 30214, 5028, 4806, 2530, 16006, 29504, 16813, 2140, 25747, 7260, 30691, 20213, 14392, 25023, 23488, 22267, 13212, 20000, 19430, 8208, 7421, 5281, 24714, 20488, 16123, 8588, 8090, 13715, 15451, 30779, 17994, 1683, 17657, 5856, 5546, 21410, 16037, 29702, 14191, 13459, 677, 8248, 8718, 4373, 10963, 29807, 4778, 18492, 5644, 7914, 16061, 20456, 4611, 22057, 10283, 17675, 27036, 4572, 20094, 33086, 31500, 23683, 17844, 7692, 9675, 15441, 21382, 16007, 19208, 4887, 17472, 24903, 13302, 5973, 19310, 32923, 4320, 10304, 24286, 1956, 15468, 5344, 11415, 33262, 32088, 22176, 1595, 14047, 8012, 13936, 10915, 15151, 20899, 13894, 4520, 10103, 4670, 13517, 18397, 11330, 19978, 16222, 5922, 13004, 20099, 22461, 100, 23649, 21134, 19428, 14600, 29934, 26576, 32544, 19342, 25880, 23658, 33620, 10579, 12868, 32594, 22929, 7795, 33292, 22086, 1613, 34208, 21355, 34525, 24506, 6406, 26349, 10926, 24133, 28394, 7012, 32081, 27243, 1605, 29267, 19974, 19593, 16818, 16840, 5564, 31757, 2638, 18680, 8504, 9040, 3997, 13831, 361, 11569, 32780, 17249, 17820, 10105, 21755, 18479, 30074, 19241, 4205, 29860, 15886, 28761, 25384, 1791, 17784, 24731, 28551, 12506, 3747, 19360, 34144, 7469, 5333, 21073, 31165, 22979, 6136, 26816, 17115, 30217, 30117, 10898, 986, 19321, 9032, 311, 5539, 3176, 31425, 19622, 2882, 15404, 13340, 1125, 18642, 27511, 32204, 4973, 31077, 20271, 27376, 130, 9324, 20547, 4831, 18445, 60, 25233, 33962, 7718, 33087, 31073, 2195, 6855, 3269, 19446, 11654, 14515, 20102, 26906, 2020, 1148, 27083, 19151, 34108, 31933, 7177, 32197, 33046, 15905, 10271, 29743, 11884, 1794, 2808, 1064, 2775, 29089, 3935, 28156, 3263, 33878, 11160, 1136, 20277, 32648, 27000, 2693, 26401, 9277, 3705, 25553, 3602, 225, 10442, 5479, 33715, 25945, 3486, 31861, 22190, 30201, 4090, 757, 31532, 22079, 15364, 24987, 11614, 30793, 29591, 32345, 23265, 10027, 27898, 6262, 26706, 21796, 28462, 18890, 27740, 25252, 20388, 21981, 4960, 30206, 30841, 33091, 14251, 1017, 32098, 17061, 5342, 1565, 12104, 12317, 14387, 15, 29906, 10702, 31483, 14676, 20349, 11196, 4478, 31103, 15860, 26702, 4940, 21424, 22844, 26652, 15347, 12750, 31706, 25102, 32090, 3851, 1220, 20226, 29456, 15829, 14838, 785, 9729, 2, 26180, 18618, 6109, 7665, 7636, 26646, 7220, 33015, 24278, 15242, 22021, 33219, 31420, 5782, 2512, 31595, 25277, 26639, 9162, 13587, 13829, 13061, 4858, 27425, 19738, 14930, 27852, 1562, 34307, 9640, 10234, 13674, 24453, 8019, 11312, 29767, 4706, 24089, 9677, 18319, 23910, 8519, 8937, 22917, 12838, 22420, 8071, 15818, 9760, 25358, 12218, 1885, 1578, 16165, 2355, 16492, 29367, 18557, 17954, 21079, 17938, 4675, 20086, 29268, 27611, 27976, 30124, 6838, 3948, 11631, 7261, 33596, 33034, 18121, 9596, 12224, 183, 13385, 11230, 30023, 26902, 16394, 28243, 17457, 30859, 8455, 7639, 6743, 132, 19709, 33664, 27970, 10012, 18593, 6914, 4666, 32446, 24272, 14983, 29613, 3574, 26964, 7592, 2319, 21522, 17464, 20854, 24359, 19951, 17019, 25019, 33384, 34110, 13270, 25583, 6041, 8764, 22308, 6988, 26972, 7977, 1749, 3190, 7036, 27707, 3195, 9238, 14248, 6033, 26751, 33829, 1652, 19126, 6096, 1979, 25975, 8082, 7904, 17859, 9431, 20335, 3007, 18728, 7498, 13016, 17044, 25068, 31769, 22274, 4907, 5227, 16127, 28184, 1209, 3579, 29648, 34086, 7279, 1417, 18804, 26779, 24153, 24470, 11108, 10755, 7394, 17823, 1252, 32434, 28777, 21722, 31837, 17919, 21429, 15238, 30430, 20342, 28194, 1002, 7304, 2660, 10532, 12084, 28944, 3743, 32585, 6782, 6268, 33760, 23814, 4461, 4000, 33265, 6601, 6219, 19545, 32253, 32373, 17934, 3543, 28802, 31749, 30458, 31003, 27895, 33858, 1263, 31254, 17425, 5784, 30240, 7047, 33629, 4878, 28060, 22896, 18807, 13583, 4597, 34513, 24219, 16383, 29450, 25466, 20502, 20844, 32749, 30498, 23297, 26586, 32207, 6826, 18341, 1682, 16428, 29103, 20892, 1029, 10203, 10531, 17033, 30359, 2337, 13929, 9018, 28029, 22059, 12640, 12658, 18249, 28464, 26396, 20074, 7897, 34145, 27550, 9109, 23747, 26532, 5328, 32139, 24652, 11761, 25761, 906, 3875, 18521, 14074, 3783, 27471, 13566, 7303, 29050, 4595, 6293, 11608, 26359, 33301, 31643, 20708, 29412, 22760, 4376, 24873, 17743, 14146, 22354, 18503, 32358, 8771, 25146, 12227, 23417, 5584, 30849, 13623, 16379, 8839, 18517, 19049, 19161, 1065, 24601, 19176, 6895, 18043, 10815, 3610, 663, 7527, 13879, 6820, 23828, 13084, 10088, 18696, 21777, 15708, 4544, 24787, 3031, 11513, 28136, 15266, 3152, 24858, 34501, 3335, 1412, 26211, 1919, 29429, 5801, 23142, 30449, 13466, 22299, 2736, 31413, 15454, 29128, 20883, 266, 3145, 7589, 1691, 4925, 9218, 13500, 4124, 530, 23920, 33727, 6218, 8101, 18429, 34121, 4775, 21915, 14752, 11692, 3420, 13559, 15327, 1071, 21938, 21378, 17148, 25432, 13333, 20670, 15790, 19107, 4731, 26521, 241, 248, 13063, 18050, 29081, 15541, 19666, 26534, 28560, 14167, 29240, 33593, 20501, 18486, 6480, 5948, 27220, 20124, 32483, 11456, 25964, 18219, 26610, 4066, 11645, 24732, 8066, 16343, 14351, 9604, 29501, 5637, 18233, 910, 28783, 8015, 25698, 15814, 32295, 30392, 27567, 32235, 31808, 13546, 33816, 15307, 29315, 8776, 24740, 3767, 21419, 5653, 3042, 14728, 4895, 485, 4379, 13269, 19995, 1292, 12007, 18081, 16627, 1612, 20212, 11522, 33252, 23679, 25673, 18655, 4938, 20544, 18592, 29639, 10957, 669, 24693, 27933, 24198, 16673, 23443, 13389, 9251, 10720, 23189, 21286, 25077, 33421, 21898, 22712, 16633, 8501, 26305, 17755, 18777, 22532, 12626, 856, 9132, 27264, 28428, 8147, 31352, 5729, 19198, 28033, 23485, 7721, 27171, 3057, 9819, 25554, 14906, 6276, 8087, 21365, 30334, 28836, 23793, 2546, 6052, 17156, 2584, 22543, 17270, 7841, 26593, 26761, 4869, 11418, 22732, 31759, 15470, 11961, 17793, 18090, 19879, 12489, 26188, 29846, 6475, 23350, 18881, 16688, 12669, 30369, 32961, 4951, 30487, 12584, 14355, 32487, 33725, 26170, 30720, 18428, 31411, 7607, 2965, 2908, 29213, 10999, 34160, 14153, 24990, 31131, 1497, 1797, 11847, 21965, 21514, 31887, 2236, 2146, 9056, 2270, 1309, 4991, 23016, 27845, 30871, 16779, 33348, 27654, 24251, 26829, 20905, 3544, 31275, 4857, 4307, 3103, 11277, 34024, 14008, 18879, 30713, 22780, 33496, 28038, 2877, 28602, 23228, 30064, 21878, 5592, 1631, 16976, 13178, 20206, 13522, 6999, 17473, 27652, 18243, 4245, 17773, 8659, 4709, 6366, 22578, 23862, 20398, 23371, 21003, 29219, 2015, 11592, 31042, 26259, 7296, 29486, 26388, 12210, 34025, 14642, 1560, 3112, 12327, 10954, 6985, 27512, 4916, 8031, 22733, 8672, 20077, 28488, 24386, 34427, 18794, 16579, 13440, 4517, 9360, 33839, 13983, 15752, 21780, 14064, 2193, 17333, 4702, 780, 32048, 7651, 33253, 29204, 19885, 7685, 5258, 1575, 19489, 5282, 28304, 32687, 27506, 4437, 6810, 9115, 5021, 12651, 12155, 22649, 9112, 21384, 7339, 12203, 4020, 33823, 26343, 17804, 15840, 22056, 33921, 23084, 16314, 25613, 136, 15244, 12081, 29271, 4034, 22064, 20988, 14283, 33452, 1970, 11566, 7747, 5318, 25009, 4392, 20955, 31541, 683, 23067, 203, 2246, 3, 260, 3419, 25085, 1115, 16748, 2104, 21848, 32429, 32539, 5290, 20546, 26732, 1286, 5722, 9006, 5845, 23541, 26367, 7381, 30575, 27508, 31842, 30340, 31433, 8365, 14368, 28047, 2191, 31299, 31804, 29773, 29727, 11022, 9809, 31787, 6630, 13625, 23430, 4530, 11749, 7649, 4476, 20761, 16484, 19020, 6781, 24449, 8415, 11152, 25714, 32075, 33387, 9222, 9663, 4899, 2154, 33112, 22128, 6097, 11257, 21767, 30024, 21693, 22441, 11709, 12900, 13367, 26496, 6981, 1883, 21123, 3877, 1153, 11482, 24494, 5941, 475, 30587, 7233, 25766, 22602, 12960, 7712, 6396, 31701, 21698, 30036, 19048, 28557, 15504, 1786, 21215, 32343, 24581, 8811, 16010, 23487, 462, 19875, 6239, 21538, 6076, 25994, 33140, 32354, 3620, 14669, 24682, 25516, 15786, 6031, 28340, 26092, 26962, 24615, 33595, 7157, 26635, 949, 17496, 11581, 17144, 17279, 483, 9090, 14365, 7254, 27005, 30640, 13252, 33122, 31074, 18414, 27546, 27152, 18865, 8435, 6852, 25285, 29558, 11944, 17335, 11996, 23513, 17915, 19170, 18, 28014, 16686, 26683, 484, 33567, 9556, 28347, 2566, 19704, 29191, 29086, 29349, 2701, 18166, 28693, 19319, 11146, 16470, 28896, 6199, 32470, 27851, 993, 16446, 33820, 18712, 13757, 15530, 20977, 8567, 16641, 6939, 24655, 22288, 14294, 16060, 21925, 26287, 25185, 23863, 21625, 19717, 4310, 19175, 1942, 7216, 2893, 13415, 18884, 29147, 9317, 25668, 9068, 24366, 24053, 23055, 9906, 15895, 72, 2263, 13817, 28860, 33392, 9345, 9110, 21640, 10641, 11217, 4758, 4723, 22592, 7931, 17006, 2155, 16594, 21166, 20607, 11926, 1394, 28714, 9406, 4995, 29919, 2354, 24431, 13189, 11003, 23168, 882, 24319, 24622, 10759, 29217, 8945, 30001, 33240, 3173, 27987, 30297, 16387, 13508, 17612, 8537, 32440, 15482, 23516, 7410, 22289, 653, 29886, 13288, 13678, 776, 6469, 12323, 24428, 13819, 8088, 28656, 19361, 16243, 16145, 8622, 9771, 12548, 11066, 21793, 9007, 20088, 17588, 3066, 1062, 10121, 1266, 1835, 789, 28529, 10422, 5691, 16452, 28174, 2637, 22904, 11313, 2299, 14908, 8990, 4384, 1719, 25641, 20030, 33155, 32384, 30479, 4621, 18328, 33381, 18311, 7404, 16713, 30714, 25756, 15458, 12461, 25881, 21669, 23126, 28805, 32966, 13127, 13563, 17638, 7441, 4661, 8679, 33527, 7567, 14148, 5164, 20121, 14025, 16481, 8005, 22921, 33197, 32833, 12036, 21521, 21751, 25973, 28511, 33247, 15295, 12019, 6558, 21065, 927, 29151, 6642, 22334, 26590, 8723, 30960, 11378, 22287, 32209, 14241, 5143, 11772, 14532, 11932, 11639, 12450, 15049, 10104, 34260, 7271, 16265, 29099, 5241, 23866, 17043, 14075, 6250, 17533, 17049, 12811, 5890, 19249, 2558, 26379, 15783, 6310, 1086, 17952, 26207, 24338, 29247, 11156, 20549, 14615, 19385, 27214, 991, 12328, 23054, 16511, 11521, 9186, 5399, 25707, 12284, 26348, 34269, 27037, 10228, 4685, 2418, 29059, 1186, 28013, 15499, 27748, 31802, 31586, 9422, 30576, 22448, 14268, 25865, 23358, 13891, 5139, 28662, 19051, 29979, 29573, 21251, 376, 8734, 26081, 24879, 26204, 15619, 19214, 4965, 21138, 18816, 14646, 26313, 24765, 8899, 3682, 2138, 13088, 28870, 28163, 15689, 17222, 2044, 17113, 1083, 9984, 16767, 9932, 32781, 21387, 27701, 12581, 1310, 29051, 9689, 29477, 29418, 30020, 12024, 23832, 21830, 27109, 22761, 21621, 22636, 25786, 6958, 24118, 294, 29660, 28522, 15395, 16568, 27387, 20164, 19301, 2259, 16257, 31935, 12625, 22217, 19187, 5476, 25266, 3065, 10602, 11564, 7392, 10236, 9159, 3491, 30271, 11064, 16864, 13899, 16467, 23526, 7650, 3144, 11004, 20649, 5180, 18204, 374, 11393, 5765, 16824, 6771, 3040, 15946, 8298, 10318, 8319, 21868, 3841, 25225, 629, 34512, 31738, 26172, 7187, 32683, 10868, 31168, 19415, 15475, 33532, 32671, 27435, 26093, 18316, 19083, 660, 546, 19671, 17602, 29265, 28969, 3124, 24911, 31402, 6178, 17004, 5441, 2363, 7438, 4812, 3737, 33494, 16375, 18149, 25862, 19997, 24719, 3679, 28943, 10175, 7289, 30840, 9766, 6023, 27951, 11928, 33936, 28103, 20610, 29322, 34178, 21716, 28281, 19410, 20938, 17463, 3162, 32858, 32985, 31516, 24254, 29878, 23999, 10274, 21271, 11229, 4745, 15456, 2095, 9842, 29976, 16538, 18971, 19057, 18405, 4943, 23922, 20309, 28377, 17680, 15070, 3498, 14542, 8744, 34272, 34490, 21906, 11825, 22866, 33489, 11314, 15918, 16182, 852, 7912, 26877, 25685, 4176, 4644, 23203, 7657, 17708, 32761, 34310, 16306, 28688, 13070, 24231, 3258, 33677, 6318, 29430, 5116, 17070, 17238, 14069, 31390, 24829, 19523, 3319, 228, 28637, 8677, 16015, 33446, 7686, 11008, 32169, 23063, 2451, 15113, 27783, 13266, 8995, 4557, 12374, 21306, 10797, 32394, 29802, 5696, 24114, 19925, 565, 6677, 3285, 29509, 30299, 13588, 34152, 4837, 24550, 2192, 18949, 4361, 26073, 15576, 14895, 14123, 28171, 33409, 21276, 27625, 678, 19853, 29563, 2777, 28759, 24124, 26487, 15892, 28091, 1329, 17977, 4750, 20941, 34196, 22073, 33261, 34171, 15876, 17223, 8215, 16965, 20170, 5176, 4735, 4206, 11828, 23343, 10534, 19561, 2150, 6757, 15439, 1776, 27378, 27825, 2892, 15615, 28275, 25251, 15220, 13693, 7916, 18730, 12747, 8311, 28089, 8028, 26442, 20858, 13677, 12645, 24341, 15198, 30212, 18523, 14907, 19734, 29676, 21173, 12777, 33952, 18119, 1654, 15452, 9741, 31048, 6440, 7961, 28425, 17175, 18876, 20559, 30538, 32832, 4819, 28399, 19831, 30424, 21235, 24995, 5556, 19772, 27278, 32044, 34154, 13097, 23234, 33126, 11234, 29592, 33916, 7407, 18461, 17884, 26738, 23121, 3858, 11467, 5462, 6100, 14377, 149, 14546, 131, 15297, 19403, 24986, 31974, 11646, 28354, 31971, 4581, 12290, 15747, 11006, 9457, 25693, 3882, 21527, 1774, 26403, 29543, 4599, 5531, 27984, 33639, 29009, 6859, 21734, 4956, 26634, 2167, 13828, 2703, 11235, 7137, 579, 14880, 6414, 13011, 21142, 14927, 7766, 30739, 13937, 4101, 32480, 19412, 27681, 7832, 19903, 19900, 7325, 13648, 14281, 18578, 29574, 24461, 18453, 7507, 21849, 12933, 14492, 25513, 32231, 4040, 23019, 11638, 1090, 30592, 32944, 24076, 9283, 21455, 31086, 9363, 16268, 17538, 33346, 1318, 24814, 7865, 31821, 14143, 27716, 32559, 27057, 139, 2791, 27966, 22832, 8374, 1512, 15329, 17203, 26527, 31472, 58, 24864, 11222, 27194, 34544, 28268, 16087, 17857, 5417, 15610, 24644, 19588, 7076, 11302, 11618, 16527, 13825, 3271, 10513, 19662, 1731, 19682, 3936, 29836, 2243, 30147, 4705, 9664, 33029, 13426, 14614, 18785, 26785, 17105, 24416, 25270, 425, 2059, 15496, 28905, 14514, 14957, 6577, 10737, 22147, 30528, 23380, 6542, 31659, 32640, 16850, 11535, 885, 4334, 12973, 24634, 13047, 7899, 18640, 23603, 26789, 21168, 19017, 20256, 12067, 10032, 13213, 21546, 12606, 34008, 23549, 5684, 30578, 2552, 16445, 28483, 30412, 25504, 25506, 12537, 19851, 8301, 31176, 10673, 32217, 15700, 32852, 32656, 7030, 7804, 11602, 25720, 16934, 2962, 18008, 3422, 10338, 28169, 3854, 1366, 19417, 4487, 624, 30961, 10773, 30025, 10762, 33997, 13282, 25972, 23850, 16384, 4224, 2939, 10803, 2520, 16618, 18538, 932, 14533, 16790, 14036, 8380, 28822, 16251, 4031, 20781, 23839, 16301, 30905, 14242, 7765, 22136, 25374, 11476, 25198, 1000, 20022, 463, 34534, 10037, 13305, 10717, 12363, 420, 33912, 2292, 33251, 17914, 3733, 17564, 30163, 34344, 29554, 14652, 18107, 6319, 4418, 26943, 18685, 22329, 26809, 24408, 19512, 12572, 4986, 1389, 7705, 6698, 12677, 22040, 633, 19474, 5883, 2216, 33515, 1699, 24662, 20318, 9638, 14949, 20838, 15614, 33388, 30938, 7722, 11603, 7955, 29397, 12699, 30622, 6011, 27485, 685, 31195, 11711, 15869, 9582, 13634, 20940, 33037, 21588, 9480, 11166, 4626, 34172, 33895, 32729, 31977, 27655, 316, 33054, 22367, 16250, 33508, 6289, 17687, 24499, 31006, 2646, 18229, 2846, 15293, 947, 9717, 24256, 12831, 16729, 27290, 13515, 29082, 22724, 8103, 31419, 17120, 27423, 28224, 21615, 1015, 9163, 5149, 18909, 25052, 24528, 26937, 26128, 9442, 25528, 11540, 3522, 29364, 24868, 22916, 1180, 23436, 15544, 32696, 9199, 23639, 31976, 19519, 10070, 21591, 21018, 15218, 11925, 28317, 7246, 17976, 18749, 12962, 16966, 19250, 32475, 15294, 19828, 2768, 26426, 18620, 26194, 5072, 8793, 22605, 32829, 1589, 16929, 8770, 26818, 5872, 24489, 9920, 33131, 24045, 7175, 249, 6389, 1716, 20384, 487, 32549, 33646, 31931, 898, 1102, 8459, 31276, 10736, 26115, 10739, 4653, 27303, 3435, 27607, 13249, 32707, 28395, 10138, 707, 19826, 30145, 10095, 17278, 29576, 26982, 9543, 4817, 4786, 26190, 16586, 20468, 12858, 26935, 29958, 13482, 9943, 28861, 13425, 16895, 22720, 32523, 15092, 23081, 19922, 5387, 16520, 31002, 20524, 7605, 21290, 28707, 31376, 22913, 15904, 23953, 14470, 22985, 2781, 33352, 26968, 28285, 9891, 25028, 32092, 10284, 23385, 33156, 29172, 18024, 13650, 2764, 20115, 32456, 17334, 27790, 19047, 21360, 2555, 1834, 21351, 25956, 16894, 6600, 31408, 26828, 2957, 188, 19865, 17576, 2344, 26980, 23888, 10690, 17240, 16282, 33100, 33849, 25326, 14519, 26707, 31196, 20775, 27073, 25626, 33463, 13712, 4620, 3558, 14836, 20774, 22922, 1599, 33679, 2798, 4575, 11232, 23951, 26690, 8083, 5415, 18331, 13593, 20323, 12956, 7344, 33361, 8737, 2755, 12288, 18725, 14195, 30612, 24776, 17905, 33944, 16694, 3827, 19680, 8873, 10947, 9049, 29402, 19930, 29467, 15690, 6032, 24004, 3382, 8454, 5956, 22479, 33477, 14290, 11875, 28283, 9525, 1424, 5137, 28874, 3081, 5162, 281, 26267, 16799, 17495, 30260, 20340, 3407, 26925, 5494, 16816, 11936, 32316, 17763, 8705, 33922, 29740, 2396, 11507, 19609, 22473, 18345, 9751, 3549, 12889, 11132, 7950, 1823, 15921, 29289, 19078, 234, 5458, 2308, 33203, 13112, 28408, 24556, 22327, 22518, 32626, 1315, 16888, 2444, 10847, 21352, 21438, 23155, 29100, 16490, 26260, 5235, 7886, 7539, 18870, 7999, 1638, 11478, 14855, 19093, 33102, 2518, 23512, 31093, 19196, 27592, 17412, 23196, 6595, 26560, 20620, 29644, 25986, 20876, 8919, 21670, 23309, 25357, 29620, 17710, 12728, 2859, 13120, 10223, 5483, 23730, 16557, 17153, 24143, 6025, 21229, 5, 34335, 26047, 14809, 26123, 28990, 21139, 32342, 9781, 1451, 1949, 16071, 23932, 12175, 32030, 29014, 26533, 2942, 8940, 11491, 11586, 8860, 10260, 15803, 21480, 32115, 28411, 7818, 8710, 5611, 14626, 3838, 24700, 32792, 34391, 18443, 8604, 31422, 2502, 8496, 32168, 16436, 7056, 5326, 4251, 14552, 2064, 8436, 25897, 19472, 27524, 23184, 7941, 32941, 18835, 31662, 24691, 24912, 28485, 5596, 3904, 32381, 11982, 15287, 11839, 8364, 32625, 22982, 28158, 15494, 18318, 114, 163, 18009, 12859, 4294, 14068, 26377, 11390, 4823, 10377, 4035, 8547, 17750, 28167, 28212, 30274, 4069, 20619, 799, 31925, 10385, 19082, 5250, 18577, 27935, 25891, 29981, 28878, 20694, 8094, 26236, 6753, 18802, 31772, 24947, 8169, 5343, 23975, 10441, 6062, 7574, 29373, 28813, 28032, 33754, 32474, 23738, 1659, 6929, 13882, 21818, 22405, 10638, 31487, 20040, 33842, 167, 28684, 5613, 17861, 20646, 3890, 27897, 33447, 11408, 7903, 11929, 30443, 10246, 17276, 22519, 7454, 27362, 15687, 27009, 10009, 14013, 216, 22567, 23062, 14196, 18605, 5527, 28925, 7388, 26794, 34247, 18558, 373, 4580, 5524, 28997, 15333, 3273, 24766, 11781, 455, 24607, 23327, 2668, 24031, 26412, 30076, 25534, 10685, 32266, 2914, 23918, 10374, 15882, 27409, 20771, 10872, 5860, 20881, 15521, 4523, 296, 8356, 23092, 27591, 9202, 24973, 19749, 6800, 19878, 30174, 23043, 19801, 8027, 30093, 11124, 8624, 14317, 27285, 15928, 17347, 20930, 23646, 28715, 3887, 5077, 12925, 19539, 14090, 7661, 29392, 3654, 11745, 29985, 24926, 5935, 26574, 14124, 23332, 17068, 28105, 26942, 7986, 30500, 9319, 18419, 5683, 23091, 6513, 13883, 14305, 10056, 17426, 7763, 21248, 5481, 21414, 8845, 21422, 24245, 8591, 26215, 20202, 21460, 26253, 30527, 5005, 26066, 34150, 22741, 28676, 5083, 19043, 20037, 31164, 26728, 30844, 6436, 1734, 16325, 27578, 8562, 28732, 16409, 34487, 2222, 32565, 10476, 30751, 5192, 4797, 18124, 21029, 14180, 24866, 7520, 29971, 25281, 16438, 23813, 18813, 13954, 22691, 15575, 18700, 26898, 34408, 18597, 19970, 14831, 11981, 27406, 29279, 5073, 30965, 7244, 13150, 18885, 34385, 33712, 80, 2805, 19136, 5424, 17149, 24011, 28350, 1657, 23764, 1875, 15346, 17456, 31225, 28888, 32193, 27768, 31705, 8788, 211, 9561, 15387, 25284, 21671, 25710, 7014, 5931, 27292, 9790, 2785, 24171, 10833, 6575, 33890, 13007, 24704, 27758, 16847, 171, 3341, 1606, 9312, 1165, 25738, 808, 22621, 3899, 18861, 15075, 31192, 2285, 33445, 16132, 8967, 14578, 803, 33011, 1545, 10572, 29260, 8920, 21019, 14044, 7963, 12487, 16957, 33618, 13763, 9580, 16038, 2050, 5229, 24463, 18067, 27010, 25460, 29865, 1465, 19587, 11637, 5507, 4933, 5817, 21361, 7545, 20663, 26146, 25380, 2130, 13800, 1414, 28831, 6986, 3996, 6526, 21318, 22053, 1760, 31866, 8477, 15165, 33068, 1789, 13171, 16841, 19295, 32311, 21876, 6045, 20746, 2330, 16321, 21831, 8097, 20526, 13330, 22464, 16079, 4289, 5939, 31110, 7033, 16803, 23355, 26929, 11306, 247, 32920, 12345, 865, 22525, 12064, 18393, 19266, 1941, 313, 21823, 15143, 12183, 9837, 28721, 19366, 18877, 16494, 4299, 29282, 4866, 10480, 4981, 15723, 8671, 26994, 29631, 18976, 23782, 24933, 16351, 4772, 14409, 7598, 32846, 18663, 3879, 2500, 24507, 13375, 25425, 16195, 5515, 30345, 19271, 21581, 7369, 34231, 23883, 24157, 4752, 11967, 33213, 2723, 31033, 5789, 18237, 26722, 30549, 27144, 28766, 29494, 14505, 31194, 27923, 15152, 11154, 1767, 33653, 25342, 23614, 34103, 6747, 26541, 5884, 17545, 3575, 15691, 21598, 17828, 31343, 24274, 5013, 21923, 8539, 32771, 9273, 11077, 15973, 18205, 9200, 1999, 30881, 10497, 25746, 32267, 21067, 33082, 1365, 34452, 16022, 26777, 28388, 22938, 33342, 32369, 1604, 18169, 23964, 426, 5352, 33906, 10817, 8106, 12682, 22633, 23965, 16962, 24851, 14731, 34156, 17720, 25879, 9881, 26351, 33067, 22667, 34193, 12609, 18915, 3207, 7992, 3757, 24287, 3321, 3891, 16889, 11105, 17485, 22580, 26137, 10023, 7100, 13205, 20509, 29179, 17692, 26461, 29933, 3060, 2604, 23076, 13289, 3333, 709, 16990, 30420, 23340, 31969, 4136, 2362, 17858, 15254, 29844, 17774, 9180, 19108, 19061, 16333, 4496, 25109, 1373, 6945, 16704, 7571, 31634, 23854, 17434, 10163, 22235, 4024, 3765, 24330, 23170, 14345, 18973, 19394, 10857, 32724, 3304, 30839, 10706, 19963, 20366, 1178, 872, 21572, 27925, 13838, 5853, 31202, 4043, 20880, 7045, 21768, 20561, 11508, 3675, 12309, 8206, 12696, 3044, 11954, 29075, 16293, 5987, 28112, 26826, 34488, 20258, 28438, 31105, 22376, 19138, 4927, 8741, 1396, 19408, 15964, 11567, 11324, 21526, 30202, 14451, 27543, 3021, 15941, 20218, 21869, 15812, 14513, 4623, 10721, 28645, 30119, 16313, 14893, 20401, 18726, 2027, 11165, 19099, 10867, 12672, 2389, 16516, 30109, 26852, 22148, 20614, 4591, 31471, 21489, 16543, 5711, 980, 30152, 2119, 30287, 26018, 33416, 26474, 14593, 15176, 34170, 21278, 34192, 24936, 15994, 11179, 7838, 24578, 18526, 4174, 3584, 384, 16707, 26617, 13778, 28315, 14118, 33186, 4633, 8117, 28363, 30704, 25368, 9141, 31158, 20750, 11838, 4787, 29502, 19490, 7134, 16953, 5868, 19566, 11574, 16870, 6821, 30325, 24715, 24210, 15312, 30106, 23498, 9424, 20683, 9413, 16143, 26815, 28713, 2936, 2331, 9099, 3803, 14116, 26780, 8335, 15187, 34005, 2817, 17090, 16358, 15862, 2369, 5009, 31112, 27348, 13615, 31790, 23473, 23584, 26615, 28559, 24583, 9686, 24347, 15637, 19411, 16004, 7313, 31832, 22168, 23286, 9995, 33608, 217, 33228, 16836, 14092, 13603, 22025, 4482, 16522, 10038, 19557, 20199, 11215, 5051, 33193, 12342, 27827, 9784, 17982, 31303, 6562, 27775, 28584, 1227, 17705, 27101, 18284, 17424, 29002, 16031, 2198, 29218, 19877, 27309, 15408, 18240, 10798, 33777, 22873, 20741, 13737, 28180, 26880, 7805, 23293, 11539, 30062, 31930, 595, 15084, 13911, 12999, 5992, 6966, 11917, 11858, 27437, 8018, 16501, 33174, 24433, 24701, 29588, 26529, 34060, 23194, 16171, 20573, 258, 10593, 19388, 10410, 29822, 21859, 8068, 25886, 28449, 19641, 14232, 32041, 21773, 12003, 1110, 32153, 14917, 2554, 5305, 28245, 14091, 21551, 33287, 30328, 13758, 184, 18413, 17586, 18711, 25495, 22939, 30706, 34261, 12516, 2875, 4959, 4881, 23748, 2951, 1447, 18914, 11104, 12398, 31703, 29438, 33718, 29404, 29476, 17038, 7666, 15734, 30544, 15632, 28971, 2709, 16506, 9459, 33971, 9014, 31011, 29423, 9077, 3181, 29685, 16249, 14665, 3233, 25922, 32988, 17357, 14526, 14574, 27093, 30311, 26827, 13708, 25728, 7023, 18160, 519, 29171, 16853, 17093, 42, 26936, 34443, 23323, 16635, 28050, 7318, 23302, 26065, 17799, 25162, 6900, 8801, 32643, 5466, 8628, 10126, 7734, 11927, 20217, 10437, 6767, 10321, 29514, 27751, 17757, 22903, 17297, 33531, 18357, 16747, 24762, 1721, 19996, 33353, 24246, 3221, 22006, 31886, 17250, 9622, 17390, 20351, 6508, 2485, 34518, 14967, 11207, 15343, 30095, 4045, 33960, 31156, 21473, 32304, 20562, 28330, 4044, 33235, 27797, 32667, 6555, 21039, 9924, 13450, 3049, 22969, 14011, 26370, 16286, 16430, 13745, 1681, 29104, 16200, 25465, 29694, 12166, 15261, 28214, 31405, 3036, 13049, 15420, 1055, 34246, 31153, 17080, 20225, 4417, 8333, 23758, 32410, 27859, 21151, 25697, 28543, 2969, 3201, 16, 7283, 30503, 19818, 397, 14478, 6487, 31717, 6803, 23865, 28447, 4724, 18932, 20769, 19500, 6718, 3187, 3077, 16153, 23361, 10169, 13343, 24676, 17782, 14996, 22443, 16528, 34301, 7094, 16174, 26553, 18376, 10259, 2383, 14444, 1281, 26965, 5438, 24460, 5565, 20316, 9282, 27443, 6122, 22751, 21661, 733, 31347, 27913, 22008, 850, 11506, 10774, 31235, 576, 34377, 9269, 25089, 9052, 2690, 26295, 25601, 16820, 32642, 1614, 6008, 7331, 25535, 30196, 29854, 10212, 23002, 15844, 22108, 13907, 1647, 3677, 29227, 29141, 23270, 10447, 30707, 31914, 32355, 1219, 17969, 6716, 26117, 18335, 7191, 5414, 28466, 28501, 2071, 10272, 23187, 16502, 28673, 16574, 27249, 6438, 14834, 23153, 15949, 29719, 138, 20187, 11258, 12015, 17541, 9444, 16029, 24442, 12757, 6917, 34157, 17697, 2324, 15497, 30441, 4906, 28290, 19004, 22804, 31831, 5734, 3792, 28699, 20852, 21934, 2651, 24277, 2033, 22748, 21916, 31983, 25737, 33204, 25224, 160, 25069, 6164, 31475, 29072, 11043, 3764, 25083, 17558, 10341, 32905, 5425, 29827, 31955, 17488, 26693, 24395, 12878, 20596, 4672, 21359, 21407, 27948, 24568, 32465, 16002, 23909, 14192, 17575, 14814, 9728, 28188, 13468, 12258, 8304, 3819, 24950, 24237, 8481, 4026, 20661, 8046, 13400, 18996, 3294, 7171, 6831, 8382, 3465, 21836, 34112, 33281, 10511, 23795, 12340, 16657, 26765, 17562, 34439, 1732, 12435, 22033, 563, 4649, 7080, 29369, 32245, 14262, 4012, 23701, 33624, 33780, 17305, 16862, 17727, 21185, 9880, 29143, 5755, 13059, 27589, 13256, 16361, 31205, 6481, 19253, 12431, 11660, 6540, 3898, 12150, 9772, 25971, 21240, 14424, 23715, 7514, 23057, 10093, 18265, 21088, 32037, 29680, 1138, 28798, 28949, 33828, 28092, 5122, 13237, 31579, 1494, 32816, 22608, 11766, 1146, 17491, 5361, 5156, 21312, 12566, 31700, 10238, 23051, 16413, 31065, 25419, 22380, 6778, 16454, 17643, 3657, 10075, 23316, 30486, 17693, 1898, 7050, 19465, 16140, 32542, 16041, 14979, 20307, 29829, 5892, 10573, 20073, 391, 11739, 13108, 10147, 31750, 21556, 11012, 15674, 4625, 1509, 12850, 25686, 14067, 18432, 31432, 23439, 18475, 14512, 10791, 8247, 23886, 14587, 10150, 14280, 24360, 19292, 10485, 18613, 6501, 1230, 28486, 20145, 17672, 17803, 22103, 15130, 14528, 3773, 10157, 20506, 10680, 26250, 9088, 34233, 18438, 31538, 25722, 25409, 9921, 33516, 8575, 13670, 22966, 33836, 24304, 8989, 34077, 14016, 14524, 18288, 13594, 10351, 3068, 33588, 31126, 24180, 14571, 5665, 6624, 2839, 30923, 23604, 26498, 25356, 11498, 34461, 26273, 19287, 25840, 33096, 13439, 12731, 22077, 9623, 19980, 17221, 23049, 11396, 26110, 4049, 7536, 3607, 2227, 7037, 2619, 17530, 8838, 23040, 14846, 32082, 23496, 23980, 26870, 32192, 500, 25949, 23720, 6934, 24956, 16151, 30780, 18664, 32039, 3542, 2793, 3164, 23864, 11419, 22322, 21558, 23135, 30432, 27338, 17087, 7858, 12006, 26090, 8642, 21951, 23960, 1380, 31588, 16692, 7125, 3884, 23650, 10810, 1625, 25431, 5070, 6636, 3267, 6957, 28593, 10225, 22587, 25200, 2210, 22490, 33894, 30625, 10802, 24889, 8917, 7942, 20024, 30195, 2718, 18784, 9915, 10366, 31937, 12924, 11049, 9894, 12842, 5694, 26837, 11495, 15002, 18863, 23448, 3027, 26174, 2894, 8303, 33835, 17532, 24059, 32797, 33502, 20891, 10498, 17418, 25187, 15584, 7923, 10900, 11734, 4366, 7746, 11417, 25020, 16424, 23143, 17907, 27459, 19928, 21741, 13489, 2534, 7560, 451, 23587, 19116, 6372, 3529, 764, 22090, 19985, 3603, 29665, 8410, 27647, 30058, 18200, 24959, 20043, 3815, 32234, 27503, 23806, 33071, 21177, 23756, 6070, 6644, 5079, 20947, 33275, 12600, 31219, 25729, 33814, 4614, 20344, 28717, 16512, 26773, 25449, 3658, 26867, 5044, 6468, 32879, 15565, 16531, 25317, 4607, 10301, 32507, 1983, 24706, 30744, 27760, 22730, 16662, 5575, 14794, 14507, 9308, 3128, 24544, 20067, 6842, 13295, 18899, 16857, 13738, 24661, 2097, 22283, 5322, 25960, 23141, 16877, 8594, 1488, 14808, 12066, 32955, 9469, 13959, 30013, 502, 33120, 27153, 17619, 4314, 20338, 17039, 1105, 32789, 33135, 31985, 544, 3280, 12717, 10082, 23520, 8104, 30472, 26841, 6405, 26609, 26885, 9826, 13769, 3713, 14590, 3856, 966, 6326, 19708, 21100, 26245, 24672, 25927, 4664, 12034, 19638, 11287, 27856, 22470, 664, 28571, 29084, 21170, 30496, 32439, 29321, 7221, 13416, 21829, 6883, 6142, 4223, 25422, 16405, 29379, 20623, 32145, 31332, 25166, 3370, 15679, 11344, 26328, 32264, 27209, 4014, 11541, 1619, 18367, 15901, 6735, 8058, 15912, 2918, 2453, 17414, 11081, 22837, 15071, 16737, 4877, 16731, 27070, 16280, 6460, 649, 26515, 430, 9083, 2460, 18556, 30378, 10836, 8758, 31907, 10518, 30030, 19178, 24514, 31768, 25691, 10609, 31203, 29583, 14484, 21373, 31021, 34404, 31646, 25894, 21479, 29834, 31854, 7788, 32259, 4195, 20609, 21576, 15225, 6719, 8584, 29516, 21279, 20263, 18185, 4133, 30374, 9329, 8302, 14015, 22965, 6621, 25383, 2318, 25174, 2269, 8660, 32599, 5994, 30360, 14130, 1752, 33707, 6902, 1783, 28387, 32903, 24082, 3551, 3440, 21204, 25367, 22976, 28088, 11859, 5473, 6888, 33459, 6991, 6694, 6282, 693, 13549, 9955, 103, 32804, 8065, 4763, 20552, 29344, 23495, 2372, 2166, 32247, 34426, 1820, 16683, 17599, 456, 12650, 14468, 14027, 7453, 31666, 3820, 4102, 5103, 10472, 24668, 31464, 23757, 23389, 9254, 5263, 17361, 28062, 1063, 26848, 3151, 9506, 13117, 32880, 24414, 9526, 32473, 9944, 22783, 27789, 12534, 13822, 10363, 33606, 1313, 25699, 4632, 20974, 24579, 19748, 17446, 22188, 30767, 19610, 23362, 2137, 25144, 11155, 18487, 21761, 13041, 7803, 29248, 14766, 7142, 9661, 461, 33619, 486, 3396, 18718, 11568, 5379, 4570, 16544, 8996, 19205, 10464, 6095, 28069, 22607, 2917, 24595, 1463, 33302, 17565, 12977, 25429, 33325, 19837, 31311, 30389, 22161, 1003, 33069, 25376, 15334, 29701, 446, 33857, 6556, 4065, 20533, 32930, 13292, 32130, 2945, 9024, 8446, 7130, 10789, 12386, 26095, 13653, 6007, 14357, 30978, 9650, 3183, 547, 2613, 17095, 23689, 22723, 5523, 17853, 17805, 11677, 29309, 26838, 11119, 23742, 25810, 7911, 33371, 7209, 4779, 24001, 24291, 7375, 13877, 22524, 27439, 5605, 10796, 11862, 25527, 7264, 7777, 5262, 25619, 14144, 18217, 3570, 16756, 12585, 10995, 33173, 28084, 6899, 28517, 28215, 16023, 28931, 33892, 15397, 6331, 12860, 33412, 7774, 32870, 13165, 5906, 26611, 6493, 16288, 16610, 5407, 8372, 24965, 33182, 9947, 12877, 14980, 22504, 2390, 17550, 26792, 22728, 24027, 24129, 395, 29776, 13024, 5553, 2744, 4088, 30177, 7121, 34065, 9380, 23208, 26209, 66, 11243, 17983, 22110, 8085, 9758, 20860, 20472, 7978, 31801, 33404, 20383, 21663, 16739, 29928, 19079, 22425, 6441, 34061, 2202, 5791, 17901, 18069, 23760, 15041, 7613, 27682, 148, 1833, 15978, 17481, 19710, 13854, 27076, 2787, 14481, 19072, 19833, 33730, 3141, 13048, 34203, 29540, 25617, 31389, 31361, 26219, 33865, 10459, 9876, 23546, 19723, 12504, 3741, 32265, 6878, 14385, 19958, 2135, 8213, 29842, 21289, 6087, 23917, 3548, 6226, 1177, 14170, 4348, 2885, 12485, 28607, 11380, 6845, 32499, 30866, 12465, 26315, 1754, 16472, 34532, 28124, 20160, 10415, 4514, 23556, 31480, 13291, 31430, 4179, 1047, 22643, 12444, 26776, 16185, 21163, 8077, 9198, 22017, 12271, 18220, 26223, 6483, 18047, 6012, 27959, 12678, 21822, 15311, 34155, 7682, 15017, 18634, 5001, 32161, 22764, 24575, 7169, 13448, 20759, 14856, 22888, 22872, 7374, 1162, 23544, 20158, 29393, 16113, 31572, 6696, 16766, 23625, 20696, 17014, 2149, 29449, 4939, 12745, 25533, 32538, 6409, 16422, 21401, 21129, 26157, 24492, 22372, 29748, 19954, 24068, 29813, 1989, 18102, 31506, 31755, 24474, 29264, 31747, 1506, 8984, 17807, 15548, 9226, 25231, 26601, 32500, 5236, 12447, 7092, 24169, 4296, 13331, 19919, 29471, 21685, 16695, 27003, 34297, 11265, 24888, 319, 8450, 20141, 1027, 20540, 16056, 22450, 7848, 5086, 3252, 15073, 954, 16463, 301, 26854, 7210, 7735, 11635, 1778, 14292, 22058, 15804, 26595, 31629, 34432, 34509, 15593, 4667, 503, 11831, 30630, 13474, 29407, 26502, 19132, 31771, 21798, 20467, 33331, 25282, 22742, 26661, 31436, 6192, 18434, 19568, 2553, 33144, 13148, 10927, 17197, 15646, 18353, 7116, 1839, 17174, 20029, 30909, 22745, 15574, 16030, 13296, 22476, 10787, 2290, 27871, 1095, 5706, 33911, 9046, 33662, 18158, 15272, 22311, 21260, 9518, 31197, 11, 18400, 27650, 6889, 17259, 21107, 17825, 861, 23529, 20569, 857, 20981, 12576, 25664, 25349, 24982, 4803, 8880, 29646, 23134, 16449, 16372, 22261, 24093, 14918, 13003, 3278, 25952, 20893, 6856, 12090, 17108, 19674, 8212, 30668, 14145, 24440, 15462, 7876, 31924, 963, 15729, 6668, 33535, 24300, 7200, 12322, 1243, 18154, 12629, 9621, 9039, 33458, 31262, 26326, 29973, 33223, 31997, 13118, 33548, 24852, 33713, 15320, 16231, 3373, 28316, 22421, 13794, 32407, 13622, 15432, 4182, 16819, 15184, 3929, 7021, 2162, 10816, 7334, 29374, 3943, 21584, 8217, 2997, 81, 12674, 8108, 527, 29570, 5165, 4548, 13087, 11122, 22085, 2408, 30114, 10587, 14872, 8843, 19626, 33913, 6287, 27408, 25539, 14399, 3354, 29904, 23938, 866, 31714, 5983, 1525, 20923, 17815, 32024, 13889, 14286, 27599, 27692, 30172, 24611, 12875, 29054, 400, 32662, 6395, 11664, 30394, 9965, 31560, 8149, 34180, 28495, 1172, 14525, 16421, 768, 18991, 4029, 21596, 15542, 13901, 8662, 22179, 12680, 9875, 31585, 17302, 23794, 15409, 6752, 25107, 18201, 27918, 14490, 17776, 9967, 257, 21340, 21636, 33750, 2539, 17547, 12538, 8170, 24692, 13467, 10969, 22113, 33652, 13438, 10618, 27552, 26366, 10057, 16395, 10414, 3484, 9104, 9520, 23916, 4586, 10928, 32526, 6462, 12092, 8914, 16241, 27100, 17216, 15546, 28558, 31829, 30283, 14650, 22627, 7946, 662, 4018, 734, 11487, 18839, 20082, 7716, 14244, 31485, 16414, 14147, 5981, 9416, 17299, 30563, 6418, 31776, 3189, 32188, 32506, 25541, 13601, 10313, 11483, 23524, 5959, 15764, 32068, 28581, 2346, 18018, 3880, 27420, 8004, 4761, 34238, 2368, 6151, 10477, 3800, 6361, 16562, 1126, 4531, 21597, 18444, 22374, 26675, 1372, 505, 8003, 14706, 32736, 17867, 13358, 30475, 18809, 15308, 24504, 33237, 20191, 23574, 2040, 29251, 20989, 23947, 13663, 24585, 23728, 3911, 24763, 10247, 17275, 3118, 20643, 65, 14078, 8812, 16054, 33136, 6584, 22623, 19591, 7122, 2838, 22967, 3659, 11939, 17754, 3759, 10687, 29715, 8565, 23161, 9096, 34500, 3349, 15625, 23253, 12577, 17611, 14589, 8006, 26044, 24186, 31253, 22251, 5308, 9239, 21772, 2983, 22769, 4656, 33844, 5095, 20361, 29932, 20339, 5369, 8193, 4449, 12369, 6972, 12532, 29517, 5175, 10382, 16001, 23428, 28960, 9072, 6772, 20084, 3296, 2486, 23416, 9706, 23610, 11587, 16402, 9786, 31035, 30534, 26385, 27269, 14709, 18800, 22011, 995, 6383, 11023, 178, 9602, 15848, 8833, 12035, 7342, 7411, 33838, 23462, 14873, 8196, 12812, 32077, 20630, 7781, 18446, 17356, 21132, 17772, 30258, 17865, 6103, 4317, 33039, 10183, 16759, 6638, 126, 3450, 29107, 6445, 28760, 8168, 9866, 4587, 27722, 24908, 9532, 7084, 11549, 34346, 24132, 4865, 10730, 19034, 22749, 28216, 30513, 30402, 1911, 18659, 8717, 29011, 26811, 25989, 27555, 21608, 14905, 33983, 18046, 12037, 1113, 18849, 4200, 33089, 10327, 25779, 26855, 6056, 13239, 414, 813, 30885, 31180, 14111, 28772, 11427, 12190, 1479, 21690, 12164, 5537, 24948, 21555, 24865, 17071, 27734, 5645, 14456, 19685, 32521, 9263, 15191, 32598, 11623, 18246, 12701, 19841, 9084, 27841, 27388, 30162, 13618, 7990, 29488, 24381, 23028, 2710, 27060, 6365, 16027, 15724, 10749, 32449, 2079, 25143, 11914, 292, 20270, 25819, 31028, 34032, 496, 27809, 1713, 17398, 4473, 12713, 1693, 1067, 2851, 1694, 12416, 16980, 30760, 33227, 2000, 11226, 1529, 4128, 12822, 30991, 4495, 4125, 24985, 9870, 31085, 14028, 23572, 1221, 19502, 23901, 21854, 25999, 11210, 23645, 22980, 20599, 371, 8154, 20321, 27726, 26987, 7460, 7577, 12598, 27316, 22808, 34074, 24546, 14516, 4448, 18347, 17990, 11557, 34294, 13018, 7729, 2356, 28842, 31170, 23038, 8453, 4867, 15127, 3150, 17346, 6681, 4493, 8480, 24131, 7138, 18409, 24876, 7889, 30352, 20055, 18495, 17966, 1400, 23788, 27124, 8571, 3083, 16742, 15785, 13654, 8699, 27253, 1935, 11261, 31008, 10851, 9762, 20021, 12964, 24139, 8910, 3046, 31745, 30633, 1446, 10254, 11918, 28859, 23727, 7926, 23792, 27004, 25036, 19235, 21072, 26231, 32916, 32550, 7228, 5323, 6616, 17721, 2762, 25114, 18152, 25315, 3959, 12500, 26488, 6522, 16478, 22478, 3514, 28314, 20312, 32513, 13511, 14411, 19979, 2359, 18329, 23146, 2424, 3231, 6545, 25567, 10335, 29515, 30907, 24056, 29472, 19896, 27544, 17173, 23555, 21211, 32673, 26890, 22890, 7624, 17007, 6456, 5777, 152, 31174, 29601, 9473, 20345, 5918, 5703, 13752, 6802, 3307, 23032, 25946, 12285, 12478, 30536, 7760, 33631, 16517, 32228, 18586, 10373, 625, 1237, 33148, 26599, 2052, 3386, 12349, 4985, 1197, 9022, 23518, 26150, 25816, 10490, 26196, 19909, 10198, 16410, 5839, 28748, 8648, 34295, 33118, 16050, 34410, 3154, 28348, 7057, 1011, 32385, 12591, 31248, 12055, 9414, 26434, 498, 19771, 14112, 11907, 7353, 9590, 16550, 16281, 6763, 29022, 3453, 11794, 3624, 2563, 10835, 30647, 7003, 31132, 31995, 27337, 2113, 16612, 11290, 20130, 952, 22594, 15551, 15851, 24824, 11973, 5544, 29889, 27536, 2648, 25094, 11288, 4168, 15188, 19328, 11502, 5570, 25391, 5364, 7740, 1444, 10, 273, 14007, 30925, 17045, 15156, 5905, 19450, 16891, 3114, 1319, 9817, 6965, 7098, 29378, 15713, 7145, 8907, 2794, 10986, 7184, 23540, 22230, 15591, 5512, 2102, 6137, 17303, 17228, 22663, 20687, 7827, 24373, 26106, 3414, 935, 15868, 18854, 24478, 22118, 8541, 25665, 16902, 6608, 21624, 9734, 31294, 22427, 824, 28661, 33902, 27219, 12686, 33505, 6756, 18751, 11460, 12636, 27785, 25659, 2298, 26147, 8238, 33133, 20711, 31493, 3628, 2980, 22433, 4673, 3893, 25493, 32777, 13984, 8079, 14135, 528, 8962, 11470, 27014, 11269, 892, 19122, 5384, 21465, 4410, 25176, 26278, 9827, 17733, 33145, 32737, 20615, 12901, 12139, 27447, 10365, 17362, 21236, 697, 12590, 15368, 24772, 32146, 25262, 31919, 29070, 34486, 6860, 10694, 20422, 32577, 10696, 25515, 3363, 28741, 4184, 18442, 17375, 4250, 14707, 21711, 21190, 21253, 29088, 26375, 12799, 19939, 8618, 31129, 33980, 23896, 5261, 1702, 6450, 30903, 5930, 33045, 24484, 18902, 13022, 28618, 2311, 18032, 25098, 21171, 19137, 17171, 27588, 22791, 6089, 33882, 30284, 26392, 33238, 28415, 16576, 19857, 10290, 13963, 26558, 6305, 20631, 15805, 24980, 4456, 34472, 32918, 6955, 6352, 24188, 26986, 31224, 17318, 16904, 20817, 691, 7772, 18649, 31716, 21272, 16441, 144, 15500, 19854, 24844, 9034, 14408, 12113, 17896, 10206, 12690, 331, 868, 15643, 4292, 23237, 1490, 11157, 21771, 12278, 27872, 16736, 28385, 26866, 7975, 30676, 11383, 21053, 1085, 19872, 6323, 2925, 29480, 28078, 34096, 25354, 19488, 33164, 2952, 22899, 18212, 31715, 3458, 17899, 34174, 8452, 23119, 10953, 11695, 26216, 32624, 33507, 577, 968, 10604, 4915, 14677, 31695, 25004, 23261, 29455, 24624, 33444, 12059, 8026, 23408, 13442, 20059, 10837, 15683, 24486, 4161, 6005, 11321, 15414, 32452, 4119, 20822, 8433, 18334, 767, 14376, 4892, 29034, 23477, 34076, 34484, 28085, 1456, 6721, 33717, 31318, 34151, 8285, 33794, 28850, 17845, 7496, 5511, 6082, 8352, 10591, 33362, 10190, 9656, 2034, 20541, 2101, 18423, 17118, 30, 25575, 14510, 2878, 31904, 17569, 3472, 30303, 28000, 21867, 22500, 7041, 6059, 10310, 6467, 32119, 26587, 6754, 18903, 8558, 6799, 15653, 31440, 16624, 15181, 24276, 14342, 3553, 25222, 33948, 12375, 32942, 13382, 237, 27769, 3479, 3253, 32101, 20161, 18403, 23020, 27148, 1331, 4060, 30056, 10195, 14912, 14678, 4074, 2779, 26248, 33722, 586, 1236, 21668, 19754, 12362, 8458, 27739, 15880, 11005, 9421, 31860, 30977, 21284, 30694, 24616, 11355, 30003, 14810, 21450, 11528, 5769, 22694, 17971, 3274, 24591, 1736, 31785, 10734, 11484, 27115, 6524, 30054, 318, 22149, 27936, 20322, 12902, 17891, 33032, 29762, 9591, 28433, 24086, 9537, 32328, 22905, 11450, 26886, 24783, 1680, 3136, 21050, 6933, 24718, 14185, 16995, 15206, 22928, 493, 23183, 30818, 3069, 13133, 7078, 950, 25887, 18818, 24260, 12980, 34234, 14588, 1870, 13357, 8912, 8531, 34406, 8100, 15835, 32912, 10458, 4475, 22263, 13075, 32149, 2428, 6446, 3976, 22333, 9557, 10136, 15761, 11795, 15467, 28756, 25995, 32604, 14972, 3680, 29216, 32143, 4234, 180, 2073, 30943, 7875, 34018, 5332, 2086, 14787, 19022, 1037, 25555, 23769, 32201, 29968, 1524, 6269, 17659, 16356, 11051, 27962, 15960, 25462, 6026, 19053, 21188, 1437, 15305, 10984, 31898, 15920, 31967, 33657, 25754, 913, 22449, 31575, 25229, 2341, 3840, 27218, 1264, 20555, 11242, 13199, 9393, 10666, 2883, 645, 7796, 33175, 8032, 17912, 2874, 14761, 5673, 15984, 22145, 27064, 33012, 24906, 2111, 28768, 24285, 17798, 15417, 32243, 25430, 15511, 30047, 12512, 10364, 26501, 7217, 8025, 15888, 29778, 11732, 34075, 31270, 17664, 21956, 16524, 17734, 32365, 17549, 20885, 24958, 22884, 19745, 31162, 29292, 4282, 14003, 18485, 28141, 17224, 34084, 16587, 12308, 5957, 23721, 28727, 25025, 19805, 32173, 11984, 27523, 7306, 14188, 17641, 31721, 23626, 15512, 29632, 14329, 32855, 27912, 31187, 17923, 18192, 24491, 26912, 33536, 27463, 24937, 28065, 2186, 13253, 29823, 20672, 30354, 25177, 22703, 25932, 26290, 33510, 24677, 26238, 12947, 25629, 14705, 21165, 28059, 17667, 15697, 32752, 29825, 13513, 9009, 8740, 17582, 7124, 21, 20333, 24288, 10550, 5828, 17365, 22498, 29287, 25280, 30819, 19883, 8123, 6047, 12755, 28883, 24184, 19060, 23494, 27099, 16493, 20260, 24439, 27551, 20243, 18934, 25912, 29159, 20657, 16760, 12816, 8785, 22459, 13701, 25523, 8500, 27122, 1893, 1902, 10368, 15326, 10996, 33288, 5277, 12578, 19503, 22280, 26801, 12916, 23492, 14878, 32273, 27989, 26783, 8024, 29699, 4835, 11762, 1975, 14457, 11808, 14494, 29396, 7435, 25321, 27367, 15382, 27847, 17140, 27404, 18182, 27475, 8755, 8181, 19570, 24603, 18694, 20613, 5725, 15383, 24815, 23068, 9547, 32772, 1915, 17066, 12913, 8961, 12452, 7880, 30135, 10165, 29071, 3223, 26807, 29624, 22610, 22924, 30078, 13881, 4733, 22989, 22301, 6013, 18375, 19840, 21541, 18682, 13813, 670, 6235, 18912, 3665, 30597, 16598, 4146, 570, 34212, 11134, 6393, 21729, 6676, 19484, 14531, 6770, 5362, 6499, 16427, 18832, 13545, 11361, 19769, 34382, 7535, 3530, 13569, 3485, 18546, 18295, 29210, 34209, 1228, 34039, 6857, 20153, 27254, 20181, 7929, 30629, 18019, 30330, 9742, 13254, 11148, 28755, 1667, 302, 32931, 4657, 34184, 13816, 25731, 26340, 20177, 17186, 10354, 12951, 33030, 11577, 4457, 12253, 18037, 21425, 6298, 6324, 4148, 18810, 27700, 34317, 3126, 32623, 6599, 15067, 1114, 16244, 7143, 31261, 17280, 9853, 7824, 14778, 9678, 6077, 9304, 24263, 185, 12464, 7307, 627, 30810, 18248, 3634, 23994, 18489, 11445, 16996, 14482, 8327, 929, 29252, 33005, 10128, 437, 16042, 27704, 5039, 28043, 4333, 20530, 71, 9293, 30417, 20916, 3380, 10042, 1283, 1673, 21984, 10523, 17385, 7418, 6631, 16465, 22585, 12329, 110, 32630, 8475, 11376, 10710, 18627, 20842, 3830, 6877, 22285, 29684, 17635, 31947, 26116, 4508, 14169, 4770, 30708, 23236, 28320, 3639, 6815, 6561, 20545, 7491, 2443, 23396, 32753, 5928, 25022, 12388, 19657, 23110, 30461, 28917, 32309, 32248, 13369, 4266, 708, 23313, 9451, 11896, 23207, 6165, 32843, 3990, 30253, 30272, 13720, 28123, 18276, 10160, 33103, 28225, 17618, 9635, 20952, 26746, 29176, 23166, 9479, 11515, 28685, 9979, 23558, 24105, 2296, 33977, 22106, 17577, 5794, 2479, 18470, 32004, 4751, 3417, 29391, 17328, 22492, 27414, 29545, 19384, 24698, 25204, 13719, 34286, 23803, 13122, 26704, 7295, 7237, 29859, 34299, 32949, 32390, 976, 10839, 610, 4013, 7049, 3055, 17662, 23973, 1943, 3051, 27969, 22522, 16102, 28111, 15731, 2881, 12937, 22729, 2048, 7871, 6585, 9194, 1600, 13766, 21720, 14973, 25538, 10841, 26714, 29102, 8803, 16135, 23703, 7678, 28235, 4717, 1644, 24065, 30082, 102, 33107, 28119, 6919, 9434, 7776, 21428, 32347, 3091, 25607, 20738, 9310, 4741, 2008, 29174, 29361, 5690, 366, 10132, 6840, 2713, 30050, 10925, 30717, 21947, 32310, 901, 8810, 4822, 14319, 30846, 33014, 22383, 22571, 7901, 7808, 21405, 19396, 25189, 28740, 12367, 20674, 17106, 27252, 13713, 16727, 23641, 6495, 30141, 2559, 33938, 9294, 17779, 32980, 29308, 23296, 34314, 32586, 10801, 4507, 31498, 12589, 32546, 11392, 16844, 7467, 33731, 5230, 23364, 27320, 9404, 27683, 32352, 28135, 18864, 20616, 26362, 25135, 22767, 16168, 7995, 3797, 22361, 26665, 5390, 16768, 27520, 6650, 24630, 23811, 28844, 34457, 26718, 2935, 6029, 15854, 24242, 29358, 9674, 31583, 13190, 22876, 6808, 9119, 30249, 4099, 16291, 20697, 20897, 16899, 7515, 33154, 9266, 2612, 19531, 11459, 31119, 6354, 3493, 21061, 22270, 12078, 32334, 4743, 13859, 4338, 8055, 8059, 7991, 21877, 938, 9178, 25751, 28565, 11248, 24087, 19700, 30688, 18150, 7568, 8690, 13705, 33797, 23900, 4257, 2940, 9264, 1934, 8605, 10123, 5736, 12936, 26973, 10111, 1449, 26734, 21298, 1971, 28379, 490, 21740, 19289, 14567, 15932, 2414, 1297, 3228, 19311, 14137, 17401, 19088, 17566, 2802, 2524, 17289, 31141, 30492, 12170, 5590, 26719, 18251, 19814, 20327, 871, 28003, 22126, 16764, 26134, 15787, 33242, 30853, 31508, 6679, 26674, 3329, 53, 28345, 24921, 33711, 12432, 8395, 10509, 5899, 29182, 21950, 30485, 798, 15233, 612, 9782, 22526, 30865, 17935, 20441, 32108, 3377, 3533, 10977, 14823, 11423, 32886, 31668, 1061, 17661, 15433, 21839, 34468, 33269, 21628, 19177, 5218, 19258, 6358, 22341, 6529, 27341, 20933, 18505, 26930, 27810, 33422, 4139, 16308, 28479, 24155, 20975, 18936, 31678, 12824, 8359, 17886, 30725, 9747, 17652, 9497, 1717, 8177, 13380, 6530, 24415, 11519, 24836, 5299, 19423, 3596, 26509, 21749, 24750, 5169, 22465, 15939, 6565, 23353, 10700, 164, 25692, 6670, 26446, 9952, 25091, 4947, 20023, 13347, 5386, 12125, 28982, 34082, 810, 6661, 1259, 5303, 18045, 3311, 27263, 31647, 32814, 3538, 13624, 31034, 18463, 8706, 15202, 30281, 28902, 15669, 8397, 18290, 17573, 2702, 2733, 31100, 15224, 34225, 19842, 3359, 8401, 33471, 15193, 15101, 25763, 1133, 30646, 20135, 3821, 24547, 1387, 4252, 11787, 11987, 20784, 19809, 30344, 27240, 8021, 14469, 15991, 33307, 4952, 29603, 3423, 3627, 14978, 33370, 15375, 15158, 10181, 4395, 3897, 19293, 14517, 10577, 17190, 15866, 20282, 22544, 12430, 564, 25286, 16278, 20298, 6067, 5422, 2975, 34441, 22954, 2415, 11919, 5972, 25848, 831, 3093, 4011, 7053, 9505, 15413, 33888, 33335, 20404, 27752, 925, 34520, 20430, 12826, 25963, 33581, 2241, 34146, 33055, 3993, 32257, 20849, 777, 29000, 22364, 24827, 29160, 14529, 10433, 28094, 17483, 5809, 706, 2334, 26990, 11652, 32658, 31091, 18985, 25031, 18616, 28413, 24331, 22206, 30048, 3391, 436, 23504, 23026, 20101, 20457, 27184, 11597, 24900, 19466, 33230, 30910, 29992, 424, 29657, 5226, 34163, 33280, 21730, 34050, 26820, 6727, 13573, 17540, 31394, 29400, 32718, 29462, 11026, 9050, 32167, 11663, 13803, 13919, 21799, 12127, 19725, 31567, 13286, 5496, 27636, 33460, 12240, 1918, 25762, 20328, 9863, 7863, 22915, 28274, 22227, 26229, 13979, 1799, 30915, 15846, 24530, 13255, 23523, 18084, 16540, 4638, 12334, 6189, 33408, 9539, 13772, 32741, 34401, 5068, 25365, 18134, 29578, 12424, 32637, 13858, 25467, 22171, 15981, 32450, 279, 14730, 6036, 24178, 16565, 24975, 20205, 30118, 25987, 18927, 31079, 30224, 20064, 26911, 25642, 33171, 2888, 33917, 6058, 26723, 12529, 24379, 9440, 31144, 34328, 29012, 18662, 5155, 9595, 7874, 7881, 29403, 34470, 21913, 14891, 4529, 32263, 1511, 10921, 16600, 1362, 15388, 3088, 24162, 29202, 15947, 17559, 29416, 22862, 5672, 18068, 25552, 8237, 8969, 15924, 8323, 4994, 29718, 13560, 2284, 32562, 14438, 3662, 28840, 7173, 3471, 28182, 9976, 9355, 15665, 16352, 33545, 9128, 25352, 888, 2868, 4688, 27686, 14094, 16072, 10937, 15543, 1210, 29943, 5591, 12468, 30628, 7118, 12169, 29031, 16773, 32547, 22813, 3716, 1070, 1779, 31181, 19354, 21647, 20281, 33467, 4964, 20932, 10882, 16347, 31807, 33258, 1756, 19592, 24949, 21611, 21723, 18313, 12480, 26976, 25192, 19275, 31857, 27694, 2510, 12637, 14339, 18449, 26374, 2183, 33090, 3429, 22048, 6881, 4567, 4005, 21218, 23465, 30084, 19763, 32856, 24909, 8913, 6002, 8313, 32956, 24289, 32458, 19601, 12142, 14097, 18713, 21277, 2666, 21564, 18714, 7587, 23287, 6804, 15722, 16513, 23400, 30754, 20656, 32973, 4903, 13870, 19347, 33338, 11545, 15952, 17199, 34216, 17078, 28397, 34226, 17723, 8064, 4259, 6785, 24427, 22054, 30880, 30765, 33519, 21674, 6969, 17454, 10895, 27566, 31055, 5978, 19921, 31965, 22613, 13868, 6632, 2792, 4123, 3028, 26429, 606, 14937, 31095, 7178, 21114, 21461, 10750, 25139, 2045, 7793, 77, 32595, 29839, 26165, 16861, 14758, 21205, 5274, 24119, 2521, 29785, 11938, 31477, 33283, 3157, 15371, 626, 11850, 16644, 21795, 617, 33246, 17648, 11496, 9793, 6620, 17516, 28474, 27813, 13470, 14557, 4349, 1588, 26262, 1311, 18146, 13447, 27399, 33320, 14955, 381, 26752, 13912, 4154, 23895, 5135, 7091, 5938, 33343, 15367, 34309, 7480, 12970, 14853, 1740, 29192, 2051, 12011, 23581, 15230, 20039, 30313, 30228, 9126, 16878, 26176, 15093, 17783, 13518, 8526, 11126, 2446, 19308, 17592, 29575, 23517, 34504, 2078, 22265, 21004, 12483, 25242, 27830, 19037, 12664, 7387, 28627, 5555, 17311, 26086, 23835, 21032, 3101, 22133, 30393, 29317, 30007, 23787, 32495, 5024, 34435, 23697, 17706, 18598, 20990, 6185, 20840, 33263, 16319, 22625, 10134, 22816, 30010, 5488, 16972, 7366, 4573, 2499, 20907, 25916, 15779, 24768, 30027, 30205, 33358, 22549, 17688, 21239, 17737, 9987, 31834, 2523, 27593, 5395, 32359, 8553, 25063, 29969, 26234, 15929, 11033, 5549, 8341, 8849, 24916, 32318, 22074, 29541, 29689, 32208, 2326, 34465, 7606, 3661, 7538, 18140, 11504, 30156, 27626, 25066, 28240, 21358, 23616, 6220, 11720, 12800, 16322, 23023, 11328, 16448, 16096, 10869, 26787, 5509, 119, 9815, 6512, 26361, 19377, 15707, 10115, 2657, 33976, 7902, 29470, 14684, 18745, 1499, 22534, 33151, 19416, 18715, 6707, 19427, 15429, 7117, 19012, 305, 24466, 19651, 32798, 5536, 22812, 4641, 4747, 9086, 2895, 11908, 1576, 7305, 10237, 18958, 17296, 34363, 18422, 31107, 29242, 12502, 29978, 27068, 12614, 14894, 27685, 2455, 23276, 31948, 27216, 11083, 22810, 25827, 33378, 32592, 31366, 32803, 12627, 11454, 22531, 16157, 24368, 30682, 4882, 5779, 2291, 34454, 9059, 26582, 21948, 27127, 9233, 17301, 20716, 27926, 6908, 2114, 8284, 14775, 9615, 14633, 31809, 6271, 5025, 29925, 4237, 26961, 4413, 7211, 13519, 28670, 7698, 557, 18359, 29301, 17786, 7114, 19535, 3791, 19912, 23118, 30256, 21046, 2229, 30732, 15655, 25217, 26954, 10507, 14756, 5577, 29796, 29436, 1014, 11338, 10129, 20528, 13066, 22455, 22336, 16240, 17350, 33453, 22807, 1432, 30347, 11969, 3504, 31770, 17184, 23530, 10322, 17445, 20886, 23993, 17369, 2063, 34158, 14443, 33529, 20632, 24588, 8702, 18724, 22656, 23903, 25796, 31794, 31826, 8232, 517, 28087, 22883, 23365, 16431, 17838, 18083, 21150, 28492, 24966, 4225, 25435, 31825, 33729, 16930, 25002, 7, 3328, 6123, 12030, 22055, 29293, 32982, 8098, 8102, 3177, 25049, 31157, 6680, 514, 2266, 18277, 481, 168, 16855, 24606, 22076, 27839, 14845, 11332, 6997, 20542, 30305, 16674, 14564, 15738, 330, 13111, 22963, 10567, 28569, 3299, 9376, 11600, 9463, 4922, 1961, 20466, 3695, 3496, 11185, 44, 5723, 20677, 26736, 23976, 608, 19324, 19245, 28261, 3480, 31559, 6817, 12234, 31981, 21443, 10723, 21103, 3318, 22198, 30357, 12314, 23568, 6684, 12020, 28769, 33855, 13848, 22138, 24796, 28754, 30621, 27664, 24388, 8460, 33610, 31191, 20048, 9242, 6813, 20371, 32663, 25393, 11406, 16521, 31748, 19613, 17654, 1350, 21412, 17767, 20776, 8889, 7888, 8681, 28280, 9188, 4510, 4258, 2616, 4551, 10130, 30131, 9184, 25512, 8014, 24483, 5330, 9586, 17189, 20008, 5125, 13056, 6500, 15737, 22806, 26602, 4382, 10325, 14638, 17926, 23291, 18437, 33691, 2335, 25290, 30716, 18057, 17367, 6819, 20698, 22342, 17507, 20831, 13144, 8042, 6931, 13786, 26516, 12583, 29654, 9309, 17251, 25833, 7971, 1416, 2929, 15971, 10860, 4701, 389, 29966, 25702, 13321, 8640, 8545, 30137, 7817, 11649, 24637, 814, 28096, 23147, 19261, 7541, 22340, 16458, 27454, 1143, 27999, 2978, 16981, 13184, 21027, 14735, 26481, 7622, 10494, 22553, 16109, 31562, 1865, 33762, 31537, 2968, 33886, 18722, 13673, 2795, 23421, 33420, 253, 27840, 3564, 33017, 3342, 11615, 15403, 11227, 14721, 7413, 33179, 18292, 9253, 22826, 18928, 26514, 4610, 8112, 13568, 25675, 10623, 14936, 23139, 34485, 2016, 6659, 14792, 28978, 25067, 11331, 15808, 11754, 3133, 34001, 1334, 26291, 8542, 23464, 15103, 13243, 17358, 31966, 1775, 15890, 21954, 24703, 19493, 3174, 22396, 18805, 3536, 6212, 773, 14131, 9703, 24046, 31451, 16689, 501, 7652, 18167, 9219, 17128, 22906, 28451, 19425, 21851, 1351, 30302, 17560, 15983, 34010, 27746, 29093, 31255, 25234, 31133, 25474, 25823, 21299, 12187, 5031, 12261, 15757, 17519, 9028, 33860, 838, 13520, 3286, 19464, 29964, 27041, 4997, 23387, 1780, 13994, 21955, 904, 20374, 9637, 10893, 13961, 33904, 12895, 2830, 5048, 11109, 27319, 12540, 23525, 20764, 8468, 10205, 33950, 16960, 17704, 6652, 32554, 17740, 24193, 13998, 33905, 9288, 33123, 30073, 26319, 11274, 15493, 3766, 8768, 3226, 21816, 6514, 8423, 24667, 2585, 25245, 10405, 34373, 11534, 18104, 16489, 23391, 16719, 29768, 11953, 10094, 13100, 16340, 22743, 10188, 20736, 25917, 5254, 12459, 4485, 33862, 18213, 12654, 20293, 9252, 26318, 29363, 26569, 14780, 19702, 33423, 224, 34232, 33803, 17245, 10209, 13495, 9659, 9303, 9516, 6864, 28612, 34425, 6789, 24203, 21282, 30175, 12521, 32635, 9368, 21317, 20341, 17440, 7948, 182, 9774, 16018, 1519, 15903, 26035, 32494, 15286, 19036, 8151, 9362, 30931, 29566, 10832, 15111, 13717, 9593, 12493, 24040, 9730, 1860, 12513, 10504, 8790, 24997, 26039, 32935, 11304, 5710, 7499, 18960, 30822, 25101, 26538, 12894, 816, 4316, 19479, 24717, 12729, 5667, 3778, 16982, 32296, 32976, 34027, 20594, 2219, 19156, 26061, 26645, 29296, 16081, 22734, 34147, 22369, 27094, 23180, 30590, 32701, 32977, 4424, 16366, 16691, 10539, 19140, 29608, 13933, 24902, 6453, 6400, 16412, 33130, 3404, 18748, 18860, 15283, 8242, 7444, 32053, 24566, 22651, 15604, 22944, 15109, 32131, 9768, 23225, 26782, 10515, 10061, 7256, 15053, 10361, 14616, 1128, 22115, 20580, 26580, 10640, 31431, 34175, 25638, 25056, 21244, 12047, 27576, 13029, 34188, 10197, 14641, 4121, 26917, 21095, 10113, 261, 20438, 15179, 23279, 5321, 12543, 32111, 24228, 232, 21785, 34217, 17511, 13570, 19659, 28022, 28452, 6307, 32089, 29036, 24057, 18406, 20426, 14459, 27950, 10055, 27583, 14041, 26479, 5419, 14725, 837, 22706, 19924, 13203, 3243, 25116, 23950, 7314, 13121, 1501, 29258, 32178, 19000, 4546, 21919, 17468, 16223, 29411, 3615, 6322, 2430, 13462, 8290, 29690, 22013, 26317, 19194, 17101, 23847, 34304, 1402, 18519, 13661, 11952, 14863, 4826, 29448, 31591, 5595, 14127, 3306, 32179, 25630, 15831, 25981, 30695, 7205, 3992, 12882, 26908, 11047, 4971, 31315, 8296, 31828, 26155, 6748, 24659, 3906, 14506, 23238, 22226, 1864, 21746, 22488, 18906, 6552, 16420, 10000, 4309, 7240, 4137, 30759, 30051, 1842, 12734, 1343, 17537, 19848, 6281, 769, 30268, 5576, 18352, 1641, 24854, 23583, 11633, 23272, 11748, 10644, 30425, 25407, 21430, 4291, 22360, 18919, 24102, 27812, 14126, 31250, 4565, 31335, 14198, 1038, 6066, 28541, 16406, 10633, 23275, 24084, 27956, 6370, 30703, 2537, 21781, 27457, 31501, 10784, 16475, 19755, 13384, 2100, 29427, 11889, 20949, 28567, 22701, 30353, 689, 125, 22051, 14311, 13352, 18938, 28935, 33359, 17460, 1677, 12670, 34250, 24994, 18085, 27661, 4674, 4037, 7196, 31962, 32694, 31137, 12567, 9666, 3828, 24589, 29712, 31520, 26742, 32241, 6403, 23367, 9864, 6992, 9193, 26041, 7846, 1269, 32608, 18834, 25319, 11769, 24142, 32411, 3582, 32492, 6882, 1430, 29452, 28668, 20945, 10756, 24041, 22568, 28631, 14500, 29352, 19608, 31038, 11136, 30504, 2197, 9267, 31223, 4220, 20925, 17962, 2720, 10521, 10043, 8813, 5129, 10329, 20678, 31015, 25423, 19449, 25821, 9179, 29377, 19889, 32654, 18388, 2251, 20612, 31795, 1737, 33720, 19331, 17510, 30314, 13129, 8878, 26463, 33043, 32584, 25557, 31863, 30146, 30800, 31199, 6284, 11727, 12776, 28166, 16946, 18962, 13918, 2350, 12454, 19030, 4131, 31959, 31918, 30517, 23224, 4601, 8467, 11272, 19790, 15743, 5863, 8325, 10424, 3063, 17030, 6072, 1855, 30894, 23349, 32519, 8399, 6148, 11149, 23278, 23928, 31169, 8607, 5372, 28787, 15613, 19838, 22047, 23622, 25455, 243, 5815, 15157, 28763, 31450, 12570, 3482, 14999, 4169, 439, 13055, 17987, 19113, 5448, 21186, 23997, 1878, 29615, 29972, 22297, 8246, 26338, 8330, 29290, 19316, 25597, 18303, 29463, 32121, 26612, 11993, 13481, 7371, 14804, 9871, 17059, 25909, 6172, 13551, 31409, 1200, 29861, 7522, 16789, 23954, 12172, 8282, 32383, 21516, 17937, 1124, 25803, 23880, 3509, 25489, 25868, 10980, 27954, 5012, 9351, 29205, 20134, 29273, 6006, 34124, 25609, 9366, 12188, 28355, 21370, 19627, 33347, 16712, 7482, 2307, 20224, 28934, 31561, 4094, 28957, 15668, 982, 32774, 1184, 15985, 7866, 5761, 25408, 28134, 1840, 6184, 17606, 25890, 34343, 9453, 16194, 4736, 3218, 33410, 5610, 28848, 13248, 15279, 18482, 15381, 7452, 8720, 23762, 19050, 28672, 29921, 1666, 29716, 16142, 24345, 20100, 5996, 29018, 5615, 1581, 6731, 13790, 29146, 11871, 10333, 30932, 22788, 23252, 2289, 11336, 19147, 20001, 13290, 16390, 6742, 14207, 10825, 23533, 15658, 6728, 23754, 6751, 8324, 11363, 28471, 3591, 18817, 2522, 10961, 7181, 10255, 6265, 25550, 27718, 27806, 5893, 11641, 1161, 13209, 386, 11690, 9502, 16226, 24387, 16348, 33586, 22582, 19013, 2136, 30582, 23024, 7945, 18570, 20795, 2497, 907, 10302, 2238, 32632, 11700, 15931, 16045, 5728, 32284, 15717, 28054, 33799, 25228, 1405, 32293, 3241, 22423, 10856, 20638, 14723, 11960, 24723, 9385, 32357, 5353, 31233, 13502, 15863, 11110, 29224, 33695, 33734, 8877, 26859, 13805, 27720, 19368, 4725, 14024, 235, 27354, 18707, 6176, 5545, 25472, 2977, 20968, 21341, 26014, 23271, 34400, 4488, 21131, 2013, 13716, 9529, 34345, 31911, 7064, 10172, 10110, 30736, 20399, 2411, 18001, 1573, 31412, 15214, 7409, 8670, 18732, 11088, 33930, 3965, 3108, 26084, 33628, 8162, 22880, 27613, 3178, 19542, 27673, 8955, 13878, 10993, 26551, 3710, 28083, 18778, 17263, 9166, 2879, 3590, 9585, 12748, 1379, 32018, 24862, 10699, 7361, 919, 4372, 26641, 22879, 34000, 15763, 18361, 19731, 11457, 32205, 16283, 19386, 29220, 17089, 33696, 26277, 2058, 11556, 14433, 29180, 20982, 32597, 17008, 10457, 23403, 32457, 7951, 26511, 18639, 7110, 29781, 32124, 31739, 351, 15013, 28244, 13787, 31712, 26122, 19014, 22737, 13655, 16044, 24650, 12027, 7443, 32232, 5423, 25837, 24077, 21287, 2581, 18330, 24418, 22485, 12907, 20918, 3438, 9122, 16416, 12083, 3928, 32049, 32432, 6906, 3322, 12168, 10452, 32738, 27247, 22642, 17699, 3754, 23333, 29370, 11113, 908, 17961, 11075, 21308, 28403, 25457, 9570, 26009, 33276, 20461, 19149, 27698, 19852, 19367, 26556, 21329, 5391, 14925, 27755, 19439, 17670, 28841, 15956, 16722, 28302, 29328, 16529, 31527, 23569, 12739, 4249, 21918, 1822, 33798, 24551, 5304, 32678, 20070, 974, 14497, 5800, 11060, 18862, 33635, 12321, 29453, 26644, 6434, 602, 9372, 27932, 8707, 29325, 6760, 18223, 34214, 9750, 20835, 7980, 19193, 18838, 25770, 12230, 1030, 30411, 10308, 23427, 7450, 431, 23383, 34148, 15146, 21509, 2340, 30194, 7768, 8563, 2664, 7976, 13499, 22792, 27029, 20558, 1811, 16974, 32837, 29929, 8760, 15407, 22049, 34123, 2667, 19695, 19945, 8746, 25904, 4953, 21350, 9030, 27385, 11538, 22323, 24455, 1020, 10527, 32910, 339, 14331, 3110, 22546, 2739, 12260, 14938, 18812, 18025, 5958, 12709, 30079, 20673, 32374, 8789, 12954, 834, 26252, 24113, 32796, 4297, 19028, 17419, 594, 28332, 26323, 17151, 4879, 23599, 504, 2899, 18308, 34511, 32141, 20641, 18954, 28927, 7097, 28153, 12945, 28977, 28185, 14789, 29189, 26476, 717, 33289, 15476, 28346, 30184, 24047, 21323, 23589, 32320, 24196, 6193, 1363, 28679, 20752, 31267, 14082, 9613, 11028, 5442, 30835, 15883, 24525, 4265, 7937, 24939, 13180, 26444, 17765, 23425, 5530, 8786, 3445, 16285, 6413, 7543, 24927, 7347, 16242, 34430, 15141, 34540, 29365, 11584, 23606, 14881, 5266, 5876, 8172, 17717, 33053, 5811, 13973, 7595, 12121, 32818, 28647, 21872, 31972, 16407, 14225, 5482, 8306, 27027, 24920, 18326, 2017, 10143, 20250, 23432, 32572, 9911, 11874, 4655, 26404, 22956, 32046, 23281, 23451, 27579, 16975, 12353, 17632, 2402, 31379, 27719, 19977, 14597, 17735, 4814, 5113, 13795, 15190, 24050, 26983, 28823, 27112, 31307, 5643, 22131, 10543, 1032, 3508, 32778, 27993, 14923, 7508, 7884, 15399, 33391, 20731, 4967, 22244, 10611, 28300, 9707, 24189, 15337, 17920, 18978, 20777, 30958, 7395, 14136, 29255, 6362, 8279, 1475, 12212, 3843, 9359, 11988, 12762, 23894, 16385, 20802, 24809, 12401, 4450, 4354, 18264, 1395, 2257, 3461, 33229, 24502, 4789, 26506, 4592, 31726, 10764, 21650, 32602, 32960, 34524, 26076, 12442, 15258, 19730, 24848, 11018, 26938, 34423, 10910, 24798, 14485, 6391, 13135, 26299, 15896, 6455, 14458, 32785, 25161, 12865, 1706, 29653, 27857, 9845, 20437, 30723, 14751, 21157, 17281, 12967, 32768, 24238, 10956, 25648, 28176, 20403, 31922, 8580, 16698, 17843, 13697, 8686, 27232, 19431, 889, 11730, 25742, 29918, 6488, 21456, 24963, 9367, 28625, 11738, 27158, 11032, 26997, 27644, 32080, 10732, 27045, 25888, 13067, 18702, 21731, 20229, 32555, 21503, 1288, 28145, 14612, 33941, 22936, 14586, 18323, 17852, 21216, 29764, 12228, 27030, 1627, 1724, 28431, 18931, 5676, 6180, 21875, 673, 5211, 33899, 12009, 2865, 8348, 4579, 31875, 28181, 5717, 26699, 24993, 10564, 31612, 15096, 5803, 5970, 22644, 20373, 1692, 10156, 21137, 7883, 27130, 22601, 24419, 19359, 34536, 11046, 21417, 31708, 10866, 16270, 26369, 10375, 9885, 4036, 20585, 18789, 7773, 34374, 742, 10542, 25400, 21815, 1896, 972, 19248, 7862, 15626, 30282, 22674, 21353, 4919, 5190, 2280, 31160, 30571, 6784, 30115, 18594, 13908, 29679, 6870, 11329, 12771, 14989, 21432, 30006, 25000, 30063, 25382, 8645, 5982, 20550, 14452, 31628, 30366, 4921, 12143, 195, 6699, 129, 8643, 9127, 3732, 9989, 4988, 984, 12884, 33958, 7336, 13372, 16801, 14282, 22282, 26901, 30165, 14931, 26233, 16220, 16328, 10747, 34230, 30031, 24307, 10728, 26542, 30643, 4690, 19150, 4002, 13945, 8586, 8471, 33432, 16740, 26620, 231, 17123, 15481, 18239, 19562, 23172, 4842, 29840, 34062, 9625, 7965, 25372, 27903, 21544, 10586, 1271, 19066, 3835, 28132, 30208, 13275, 5199, 23484, 4766, 4165, 13168, 32438, 30598, 5752, 6667, 30245, 34210, 4500, 28674, 33965, 25064, 18238, 5566, 17492, 25726, 7998, 20513, 33291, 828, 18797, 18887, 271, 27838, 19597, 8305, 8230, 11970, 30606, 13926, 21577, 30038, 20027, 12099, 16940, 14426, 91, 24038, 22365, 19420, 28437, 33000, 31631, 24469, 16183, 11894, 11358, 34535, 17563, 12018, 30595, 6750, 12182, 3029, 1127, 2309, 11416, 13036, 10708, 13732, 23301, 30470, 20722, 434, 17640, 22545, 29109, 5877, 24635, 25818, 30105, 22291, 30953, 848, 14423, 15485, 22211, 27600, 193, 15999, 11503, 21270, 19008, 27831, 8160, 24413, 33166, 8902, 24074, 26975, 24804, 1569, 15531, 5849, 6607, 29222, 784, 18321, 19272, 18181, 16696, 27016, 4628, 4177, 2217, 4828, 12319, 1004, 12186, 24362, 26263, 3265, 6163, 28446, 32800, 21372, 22390, 2297, 19960, 31414, 17145, 25331, 16637, 33819, 7399, 6940, 20611, 19897, 1159, 21534, 10439, 7464, 15514, 33341, 31211, 8437, 19891, 29565, 15260, 2417, 4458, 3313, 30057, 12661, 1720, 25875, 6678, 33180, 19204, 33959, 2871, 16350, 33070, 26464, 20519, 14448, 10460, 28762, 21800, 16169, 8697, 32871, 2947, 11171, 1851, 5041, 28981, 6596, 5128, 8464, 12050, 540, 28974, 22895, 24282, 8817, 23273, 34338, 16620, 17022, 28460, 2321, 30320, 2447, 34331, 28125, 14297, 325, 16564, 8124, 32001, 13580, 22304, 28392, 30929, 31088, 29572, 26597, 26949, 18698, 23566, 9848, 13099, 8891, 7451, 5662, 13728, 559, 33214, 3863, 20105, 19085, 5775, 15369, 31958, 6907, 11892, 10391, 11994, 6225, 8721, 20184, 16100, 7646, 12372, 26916, 32095, 26698, 15662, 5478, 27901, 3725, 24271, 845, 24823, 11245, 26700, 30301, 18026, 9780, 31117, 2201, 31348, 31200, 22286, 10598, 18892, 3399, 27391, 14429, 3385, 25717, 26302, 21627, 1481, 31722, 4809, 22564, 32097, 27397, 2273, 6179, 15867, 30199, 32959, 25448, 3460, 19016, 2208, 12436, 28994, 28548, 804, 31963, 13306, 2633, 31406, 19729, 8060, 12856, 7470, 22689, 9167, 17471, 4290, 16034, 863, 1300, 28987, 28586, 9333, 33286, 28405, 1925, 21595, 21448, 10269, 19300, 23029, 31685, 33953, 14561, 21191, 13909, 3822, 22800, 5217, 5998, 13276, 26992, 17801, 14279, 2686, 4781, 9476, 26335, 31730, 12906, 33530, 21264, 12158, 22750, 14050, 18658, 23755, 27527, 27671, 23631, 21092, 3750, 25914, 13198, 31130, 12371, 30550, 17622, 7580, 6340, 21928, 34235, 28068, 2094, 2380, 28687, 14850, 18681, 8153, 24976, 3444, 27, 23482, 30364, 28716, 11537, 29837, 19792, 8903, 9524, 30455, 2900, 4539, 25826, 25324, 4426, 6208, 30911, 29473, 18123, 14536, 32042, 19413, 13735, 29330, 22867, 32708, 30680, 11387, 12071, 13476, 14796, 1246, 20626, 17506, 723, 16979, 26784, 23822, 5355, 14085, 28797, 30524, 15826, 2234, 1151, 1025, 8331, 22275, 14601, 347, 11681, 17379, 16567, 14334, 9882, 27472, 27034, 13676, 5327, 13692, 30356, 30847, 2037, 22719, 25511, 14672, 20369, 9651, 1198, 20274, 10651, 13977, 4056, 13972, 27881, 27188, 11955, 30159, 5251, 33993, 4351, 23074, 20998, 28980, 28312, 10538, 16639, 5854, 27021, 26333, 26726, 22535, 20735, 27915, 23015, 8600, 6408, 12474, 14555, 3786, 19021, 29672, 21566, 12730, 21258, 8985, 10621, 8747, 9248, 12691, 31161, 2705, 25468, 20808, 32050, 33747, 21246, 13368, 19780, 24648, 10219, 24290, 637, 29597, 33403, 15936, 23398, 34179, 15125, 876, 18580, 1648, 27159, 9373, 12974, 9365, 22934, 29360, 20920, 32681, 6384, 2644, 22321, 33226, 3236, 388, 1440, 13279, 20273, 7144, 25919, 15349, 33473, 21908, 22860, 14737, 19288, 7017, 16831, 29105, 17026, 18309, 12611, 2260, 2870, 2492, 32869, 34006, 21801, 9558, 15128, 33656, 31835, 2352, 33825, 4606, 5054, 29493, 1346, 29828, 25194, 1267, 17827, 8486, 14903, 15100, 7870, 33448, 5161, 23967, 25508, 1965, 28205, 24784, 10202, 32009, 25043, 9705, 29262, 2088, 14837, 22037, 14911, 25306, 30947, 3212, 582, 2610, 881, 8253, 20268, 575, 942, 5429, 8245, 23100, 12943, 924, 22475, 20251, 17642, 29987, 33321, 22695, 4386, 28189, 11095, 22841, 17478, 15906, 31961, 23200, 22575, 17887, 14110, 18606, 7215, 10783, 11340, 29029, 33315, 12778, 19262, 15169, 19344, 3531, 29930, 4966, 10229, 10558, 14700, 11434, 5533, 30012, 7046, 6711, 12619, 32152, 26798, 19463, 20700, 9017, 20283, 20680, 4145, 33195, 22958, 23841, 9801, 27356, 4072, 5057, 30726, 5797, 5709, 19560, 23491, 20118, 28967, 26944, 31903, 10785, 10491, 25822, 25701, 21076, 25616, 16170, 19019, 9552, 32897, 15310, 7188, 15628, 20825, 20505, 34469, 13480, 18925, 15428, 29046, 13658, 4298, 3768, 1921, 15758, 27530, 13338, 27791, 1399, 20785, 8569, 32138, 28102, 14693, 4170, 28966, 16133, 5670, 26467, 8669, 6648, 19770, 6605, 25441, 19834, 19380, 29997, 21651, 3519, 25420, 30376, 22843, 13339, 18668, 17366, 3348, 29135, 19211, 21164, 23214, 95, 31523, 15995, 8002, 20014, 10495, 17057, 23875, 5312, 8139, 4975, 20659, 16332, 13314, 19661, 3463, 10080, 10681, 2514, 18774, 8469, 8756, 12596, 20823, 13504, 25669, 15553, 6688, 33044, 32453, 17831, 14667, 19934, 1532, 23129, 20289, 4955, 23532, 31952, 16129, 19455, 11905, 5093, 27815, 11135, 30321, 21466, 22339, 16339, 15211, 31544, 33704, 15040, 23341, 28992, 1206, 21196, 2810, 3528, 710, 27058, 28309, 32424, 3172, 10047, 13916, 28830, 26800, 2076, 33202, 33271, 206, 7812, 1535, 12702, 34526, 32569, 16998, 17298, 18952, 16619, 20859, 27486, 7172, 3565, 22170, 30053, 15026, 27251, 214, 20992, 12425, 32898, 31563, 18537, 22245, 20958, 11062, 9732, 24777, 20358, 25835, 20201, 2504, 9067, 3116, 10008, 10751, 2588, 11527, 3595, 48, 4631, 9795, 12890, 27965, 15449, 4759, 17796, 22258, 1408, 10779, 1275, 905, 23177, 30648, 28636, 1528, 1718, 31613, 4357, 30453, 33350, 7026, 16997, 17104, 3787, 29658, 1806, 5573, 33080, 23613, 6379, 3796, 25314, 8011, 16726, 10384, 22302, 15957, 23690, 34308, 4982, 21909, 16548, 539, 1084, 2473, 27924, 18766, 20122, 17286, 19125, 13344, 4505, 33651, 25313, 12899, 8555, 33279, 933, 23321, 32113, 19371, 27494, 21062, 23379, 1039, 22097, 30033, 28150, 6017, 28370, 10932, 33724, 22431, 29209, 2435, 25218, 11198, 23593, 19091, 6135, 7252, 17337, 23711, 3052, 20629, 2276, 28652, 12376, 3701, 18211, 6924, 7280, 30040, 19104, 4216, 28725, 22043, 13636, 3618, 2529, 9711, 4378, 22949, 20465, 377, 515, 32031, 16069, 21041, 25336, 24423, 364, 10804, 14284, 19551, 15201, 14974, 8864, 7448, 17069, 24633, 27770, 28753, 13167, 24593, 5071, 26868, 32026, 12360, 1729, 1746, 21261, 2632, 31872, 23565, 31929, 17474, 6306, 8227, 13839, 24837, 16326, 8339, 1468, 22122, 5952, 33339, 11441, 10546, 3910, 20329, 17874, 23743, 19181, 1261, 27412, 5066, 16551, 4696, 32872, 24241, 20598, 18754, 34356, 30882, 9767, 21468, 31053, 13542, 18638, 13173, 22326, 14009, 2371, 1016, 21967, 12403, 22858, 11857, 10905, 23829, 30264, 31241, 29703, 15694, 7879, 7275, 19847, 9134, 10394, 13915, 34137, 19260, 21083, 7829, 2576, 32618, 7909, 33345, 2737, 8392, 21926, 849, 8116, 19165, 7104, 4609, 15884, 6960, 4535, 31899, 887, 21225, 14447, 29478, 21345, 17666, 6629, 11236, 19167, 30684, 6903, 18847, 30820, 28774, 29095, 2618, 11605, 30383, 22848, 15748, 30021, 3794, 4067, 18315, 30382, 12262, 18688, 27618, 5614, 19280, 15648, 21499, 29169, 15389, 11651, 3917, 22346, 19067, 5287, 20748, 4518, 14159, 23853, 30908, 31324, 1617, 5889, 2035, 11958, 3034, 19124, 11805, 3972, 31050, 934, 29843, 30096, 24401, 25507, 5593, 13345, 17864, 7241, 27559, 31767, 12789, 31391, 23035, 4253, 18733, 32522, 25131, 19486, 8063, 9599, 1856, 2482, 15006, 26311, 28279, 11876, 5975, 346, 33157, 6071, 345, 33753, 9999, 16262, 1745, 6703, 17695, 27662, 6374, 33256, 32901, 22096, 12685, 18528, 27659, 173, 20276, 9756, 21483, 1241, 24233, 23342, 23978, 3781, 32788, 8485, 21808, 31900, 52, 4239, 29923, 3669, 10108, 19405, 2249, 18161, 15753, 33861, 26549, 29948, 5166, 28733, 20409, 1001, 15245, 33578, 1884, 7069, 24760, 22527, 8938, 1574, 11098, 22509, 31719, 5182, 23789, 7662, 4235, 20406, 22163, 34197, 2185, 3459, 6248, 24800, 19868, 29536, 21491, 23344, 20452, 25247, 29166, 13599, 10582, 10838, 27357, 22178, 12223, 16337, 15940, 29617, 21375, 21217, 14142, 2233, 16221, 1598, 31359, 21310, 16011, 25520, 10545, 2852, 19517, 32847, 33456, 33481, 30514, 30255, 29512, 8652, 5285, 33759, 884, 1409, 26963, 29225, 15513, 11612, 14228, 9950, 29874, 32428, 274, 10137, 17645, 7956, 23527, 32564, 28827, 32415, 24400, 30483, 4932, 19646, 15603, 7150, 14372, 13363, 5838, 4339, 12531, 25943, 29618, 19364, 9409, 7565, 8441, 17088, 33220, 7162, 31305, 14291, 12233, 23360, 12410, 23412, 29091, 3163, 14386, 6983, 4980, 23315, 26833, 5950, 17501, 29723, 8209, 7354, 33815, 3600, 8377, 13578, 12093, 31239, 14042, 11494, 10780, 22003, 30814, 14172, 14327, 3726, 29312, 28675, 30009, 30219, 18408, 20601, 10997, 34097, 32806, 11741, 5188, 13815, 30828, 11386, 20583, 15933, 34522, 21304, 30546, 29771, 27182, 13038, 9408, 26042, 30467, 27558, 9036, 12438, 14059, 12523, 22554, 30928, 1157, 17514, 3864, 30244, 33589, 16590, 17719, 759, 22624, 28962, 31492, 6428, 29656, 17229, 22797, 8625, 21008, 13091, 23820, 31163, 3421, 12700, 15055, 16730, 21342, 32333, 25748, 22236, 23251, 21245, 18972, 16012, 25674, 14785, 19631, 30748, 6102, 32433, 25271, 2317, 30505, 14395, 32038, 12617, 246, 28976, 9069, 512, 21069, 20809, 20963, 31397, 26034, 15199, 26312, 11699, 7550, 26425, 13057, 3392, 8612, 21982, 11038, 12178, 18275, 29940, 16417, 758, 33395, 11468, 23445, 18581, 16523, 26408, 12743, 20294, 22937, 31950, 29817, 27605, 27364, 25592, 21910, 2656, 8840, 12054, 7468, 396, 34194, 13690, 22491, 26327, 2057, 25191, 11547, 9978, 4853, 10135, 25030, 11591, 27542, 7996, 6720, 15834, 27759, 22666, 19993, 26946, 22466, 24628, 7022, 16647, 21736, 12051, 28353, 21280, 25434, 16092, 15137, 13904, 13885, 4815, 24429, 25704, 5805, 13272, 12287, 24356, 2212, 14843, 30852, 4537, 12828, 18404, 4532, 18656, 17972, 9536, 11168, 560, 21416, 9968, 11091, 2708, 4356, 20420, 17265, 24490, 13411, 15847, 29922, 27386, 17701, 13696, 23086, 19787, 31839, 428, 33150, 7644, 33784, 25257, 19917, 14129, 26821, 18108, 22993, 3631, 21791, 34168, 23554, 10970, 31290, 29045, 20699, 3518, 14697, 18765, 24782, 14108, 31606, 1579, 23138, 7206, 27136, 30862, 18889, 29212, 2131, 21447, 18562, 12603, 21009, 20787, 18300, 27465, 25859, 27282, 11123, 14632, 32006, 7461, 22563, 17749, 1907, 18695, 29610, 14250, 20435, 13250, 22256, 12653, 32603, 30427, 7849, 14643, 9390, 20128, 17243, 5505, 5773, 18587, 8935, 14134, 33756, 2887, 16473, 23585, 2600, 33721, 4328, 4100, 4048, 27780, 805, 30579, 31355, 20965, 22875, 14218, 23667, 8110, 2913, 12985, 26298, 10185, 24158, 33009, 4525, 19139, 22393, 31803, 26163, 26203, 20090, 26159, 11731, 21946, 15666, 19576, 20521, 14296, 30242, 13465, 21900, 26358, 21307, 3345, 4545, 30452, 31150, 2459, 22060, 27168, 31308, 17077, 14031, 1076, 11865, 9016, 32946, 18287, 5075, 30861, 25006, 21870, 33655, 13906, 24970, 32045, 32581, 7993, 13477, 31098, 13319, 16973, 12630, 3836, 28131, 13281, 12524, 10704, 9954, 24685, 8178, 946, 3707, 16903, 17835, 10982, 31677, 32963, 12358, 14363, 27974, 14107, 27725, 24080, 17011, 24802, 12773, 30917, 31338, 31097, 11526, 9143, 9140, 25596, 8216, 4617, 6928, 6589, 6177, 24872, 23122, 24769, 699, 1403, 20144, 1685, 1548, 4499, 24055, 18153, 5405, 13458, 3919, 11975, 30854, 2759, 34099, 19042, 6634, 16705, 10828, 20873, 23522, 5568, 10079, 26381, 16148, 4799, 28596, 1260, 18312, 6823, 30674, 11016, 9639, 25852, 2377, 28256, 9441, 22947, 15208, 33580, 23684, 30749, 29257, 17714, 21001, 13155, 20942, 11846, 25379, 2886, 22201, 30390, 14302, 10036, 34252, 3487, 4486, 10404, 24892, 18857, 6264, 32254, 20149, 25996, 3956, 1781, 29810, 6533, 28914, 18607, 2642, 30883, 11187, 14951, 17188, 7298, 29775, 8007, 24145, 6004, 28936, 20811, 5934, 16887, 25643, 33312, 10066, 31819, 23085, 20444, 26625, 28578, 8647, 13187, 3127, 24503, 34278, 25075, 8792, 24283, 32651, 11659, 32147, 2598, 14851, 21263, 28663, 22727, 8396, 6266, 20034, 7356, 1450, 30358, 30557, 25220, 5378, 16948, 2412, 29500, 24897, 27187, 18473, 30559, 27081, 687, 15899, 15435, 10028, 8132, 5097, 8269, 5296, 30243, 23726, 13432, 25099, 16950, 7672, 30170, 14527, 15317, 3464, 6146, 4429, 1953, 17475, 32561, 19052, 32225, 15200, 28008, 27855, 15720, 8870, 538, 30607, 429, 15203, 10529, 32795, 22515, 10551, 9107, 5560, 34098, 25499, 18752, 6995, 22165, 30015, 21714, 1536, 1073, 26744, 6210, 27461, 12299, 12642, 30719, 3362, 15219, 5256, 23107, 19863, 30618, 19866, 457, 29131, 32341, 1354, 17846, 8289, 26040, 9183, 5431, 29027, 24012, 28012, 24749, 9280, 28198, 419, 15874, 5293, 27383, 24108, 1377, 3443, 6827, 28680, 4293, 15406, 16834, 21613, 29037, 581, 24422, 31222, 10083, 21774, 3437, 19643, 12923, 10820, 5335, 12105, 2509, 12013, 15237, 22496, 8876, 14105, 16269, 19819, 5953, 6768, 9352, 16215, 16216, 24249, 32849, 1491, 33461, 17009, 1445, 20052, 3276, 26445, 22071, 32223, 16359, 31330, 29990, 20159, 5964, 7230, 12805, 11394, 18911, 19864, 21130, 13749, 15699, 7484, 10850, 33939, 32016, 30605, 2531, 32575, 33085, 28817, 33528, 657, 27922, 21510, 30488, 20702, 4862, 29197, 15080, 8916, 9792, 15508, 26863, 18563, 14072, 8982, 10467, 22032, 29482, 2716, 23716, 3410, 28989, 22035, 15706, 14522, 29977, 18016, 14826, 8641, 1007, 26265, 34047, 13864, 794, 33360, 6214, 16994, 34358, 17862, 28523, 30429, 3711, 25130, 22752, 20103, 12304, 9665, 10653, 16945, 31877, 28179, 23575, 30110, 17572, 27499, 14019, 29442, 14769, 6175, 31312, 30652, 22480, 277, 28589, 18462, 34149, 20163, 24618, 26072, 26213, 10583, 21413, 22453, 6780, 7222, 1036, 6695, 5168, 28814, 17351, 8463, 34428, 33048, 28993, 17669, 19690, 9541, 1516, 18096, 11374, 20816, 220, 9074, 25219, 218, 24572, 14415, 4913, 16164, 16710, 21058, 18194, 20851, 12338, 27331, 4540, 27296, 16119, 33745, 14635, 7779, 30795, 31468, 14627, 449, 31134, 1320, 12597, 7082, 8268, 30637, 12395, 27729, 13871, 24164, 20976, 29850, 10622, 32479, 12840, 17037, 9029, 33802, 10446, 975, 15806, 13512, 26844, 1483, 10141, 5707, 33465, 33968, 18187, 10013, 14598, 25784, 6697, 26244, 5886, 7924, 27842, 15975, 23011, 4277, 34052, 6682, 16386, 34038, 31154, 13030, 13836, 16336, 26217, 5212, 9829, 8161, 2090, 2906, 25084, 22583, 20660, 20961, 1244, 9208, 30363, 21665, 1954, 20512, 526, 6722, 4261, 14833, 23206, 31415, 26258, 9607, 28055, 10669, 23093, 3001, 22536, 22847, 14915, 21912, 20969, 1551, 34003, 3981, 24793, 12825, 20730, 1294, 31111, 27883, 25874, 22456, 32801, 4571, 19111, 6692, 15259, 31661, 6850, 15571, 11070, 29495, 14719, 7332, 8915, 25141, 20078, 23099, 28808, 3871, 3405, 1231, 24200, 13801, 16827, 8972, 26376, 22315, 27038, 16063, 32406, 30874, 5778, 15736, 1171, 23198, 735, 27766, 6244, 20221, 31621, 31648, 25, 24311, 26181, 11834, 31927, 21224, 7161, 10955, 29779, 31209, 5616, 26691, 34537, 15539, 27531, 7556, 22358, 4207, 32688, 2611, 17237, 11517, 9812, 113, 12332, 20970, 7465, 24022, 30823, 12193, 19166, 7442, 31231, 18176, 27737, 26166, 2943, 13895, 19447, 20419, 8431, 12434, 15180, 5490, 28023, 12467, 4920, 25416, 20608, 5705, 27947, 22771, 18378, 22099, 17172, 8140, 30856, 6544, 14366, 11676, 13533, 17698, 23191, 24190, 8418, 34240, 19574, 18310, 10871, 8038, 3674, 17262, 10353, 2790, 22634, 4978, 13020, 10675, 11052, 11702, 26192, 22082, 29942, 20757, 4160, 4322, 28024, 1484, 14824, 23370, 12001, 24458, 29848, 31297, 6833, 21728, 23713, 33714, 7476, 28232, 3248, 314, 13110, 17310, 20960, 7352, 29346, 31689, 14852, 7815, 28540, 951, 33723, 12939, 24964, 19443, 3986, 7015, 19442, 33889, 14489, 10328, 32606, 1279, 11689, 22250, 186, 17438, 18088, 7038, 12497, 24951, 22271, 1707, 477, 28031, 25730, 12488, 4344, 3793, 4984, 15682, 26998, 8479, 22424, 14397, 1701, 11791, 3134, 14472, 6744, 18370, 20433, 12887, 13667, 30954, 17394, 13452, 33427, 19358, 14668, 21607, 13008, 25338, 1553, 23160, 7277, 19207, 19765, 16797, 30661, 25900, 4820, 5659, 14572, 18344, 12921, 12417, 25055, 15673, 6546, 20591, 17742, 17639, 12666, 5999, 15009, 30176, 29113, 12466, 9221, 21214, 21930, 11073, 22868, 4583, 25196, 29927, 14861, 2159, 20856, 34380, 7928, 31367, 27596, 12955, 10809, 19830, 21208, 27535, 3696, 20662, 29625, 11055, 2010, 14887, 23580, 11204, 28170, 23692, 2536, 1534, 13001, 24885, 26622, 11228, 13170, 31331, 22932, 30560, 32877, 28412, 8224, 14349, 11267, 12496, 1304, 30069, 11348, 26520, 6880, 15527, 24459, 31051, 12507, 17993, 7219, 3930, 12344, 2754, 25300, 18236, 13910, 23710, 17169, 16381, 22379, 19007, 7604, 27869, 30331, 7809, 23098, 28520, 8505, 17191, 10654, 14401, 9702, 20637, 24479, 6132, 7783, 34533, 33211, 1118, 20238, 8802, 25253, 10569, 9966, 34455, 17568, 33490, 5743, 11965, 5187, 32503, 3678, 1413, 5894, 24894, 9914, 15572, 19956, 12049, 15172, 12109, 13188, 11701, 33669, 5351, 18559, 16273, 26297, 34292, 16173, 199, 1044, 9189, 26952, 6710, 22477, 6473, 26680, 21364, 7170, 20137, 30552, 5895, 24797, 23097, 12775, 1349, 23453, 5443, 32822, 1382, 18346, 7872, 25920, 29047, 5237, 33449, 1757, 14606, 13824, 21162, 24337, 16835, 4008, 15355, 7739, 25706, 14560, 26627, 29534, 30697, 1485, 24818, 29357, 21380, 628, 7186, 24295, 4811, 7154, 14359, 22972, 6382, 17313, 10184, 5730, 2564, 29055, 12351, 31439, 25215, 27114, 3229, 9993, 29468, 2205, 3123, 22647, 31124, 24067, 19456, 15598, 29756, 29078, 27363, 30662, 12896, 7019, 31534, 10634, 19256, 11404, 24527, 28072, 28010, 31845, 30407, 18708, 24767, 11422, 20515, 26125, 22061, 5622, 31830, 12530, 17821, 18710, 31473, 17689, 27413, 24726, 28070, 2533, 17214, 20972, 14993, 15567, 655, 23120, 31507, 6947, 1686, 22444, 25758, 20766, 2466, 31992, 4016, 10436, 2991, 10620, 17411, 18339, 16648, 25776, 13833, 17112, 32117, 6299, 779, 326, 9858, 33788, 24645, 31624, 29017, 31718, 14, 4156, 30143, 1504, 22359, 12979, 2678, 1831, 7234, 12274, 28843, 21005, 20476, 31135, 11347, 16866, 32813, 8133, 25884, 30970, 26354, 12008, 28015, 33710, 30653, 8837, 16588, 29359, 32504, 16549, 5294, 15609, 34276, 28416, 32881, 24078, 31189, 32033, 2314, 28657, 31652, 5949, 33910, 17476, 7483, 21049, 13747, 33076, 28002, 21648, 20804, 24150, 5668, 33293, 31989, 25510, 21411, 7898, 20931, 8782, 14619, 26543, 15284, 1295, 33800, 11525, 10555, 25303, 4479, 11087, 11599, 20204, 1596, 13422, 27715, 20140, 27174, 19357, 18138, 33672, 18196, 29521, 17097, 31020, 12592, 12229, 12659, 30446, 32258, 26519, 24609, 21560, 13964, 1282, 10455, 23514, 17997, 32455, 6292, 17162, 12330, 21200, 20624, 11673, 33300, 14618, 3199, 9339, 19279, 1033, 23636, 22692, 9946, 15442, 18614, 9435, 19200, 5411, 24657, 2498, 24023, 32467, 26472, 7561, 28764, 8577, 26535, 812, 34059, 16134, 10849, 8736, 823, 20833, 15448, 12573, 12377, 12294, 28321, 21841, 22030, 9430, 25627, 13401, 34390, 18674, 23633, 33523, 3666, 26977, 22893, 33218, 2765, 12647, 30338, 16882, 17929, 25546, 15108, 12898, 27140, 33863, 16264, 28305, 18011, 6766, 25798, 8987, 16969, 23759, 12839, 1728, 5517, 32875, 16987, 28950, 25861, 21972, 24567, 18468, 7667, 6101, 17010, 26904, 31177, 33306, 26679, 23740, 5370, 15907, 3431, 22181, 25598, 30004, 12289, 23017, 32686, 18896, 25406, 10379, 17903, 23211, 5004, 3510, 31657, 997, 31382, 19937, 29076, 12387, 15005, 10478, 27080, 4871, 63, 4552, 28928, 12272, 31489, 15207, 25611, 26649, 24071, 7385, 9298, 9544, 5681, 26027, 23022, 19257, 26536, 24389, 19485, 17005, 27323, 11869, 22891, 28615, 26594, 832, 6296, 18336, 21074, 12704, 28979, 9942, 3015, 15168, 6835, 19044, 23884, 643, 16939, 17002, 32587, 3308, 20056, 15910, 9791, 2946, 20866, 14685, 5284, 14745, 10892, 27493, 6474, 12633, 18796, 33975, 20275, 19497, 2495, 30918, 24734, 6316, 30884, 27927, 10359, 4380, 4768, 24109, 22259, 7724, 18913, 7190, 13336, 5878, 14133, 2542, 30474, 16376, 21658, 16218, 20469, 17548, 4538, 30438, 33493, 16539, 25646, 31864, 24451, 14611, 22407, 1531, 19881, 30257, 11453, 2931, 13847, 32203, 8378, 22093, 6425, 8704, 9489, 15864, 13242, 26432, 27121, 10463, 28100, 34440, 9992, 25418, 646, 27040, 28227, 32298, 33915, 17409, 24835, 4934, 21192, 33645, 2442, 2231, 30342, 16649, 2484, 7730, 4004, 9844, 20693, 27982, 12671, 23654, 11169, 822, 10148, 34237, 14819, 13444, 13949, 27943, 8929, 3389, 34087, 16084, 25272, 27545, 11590, 13689, 34422, 20475, 3109, 7282, 11978, 32868, 24021, 22907, 34298, 1502, 22711, 9597, 9475, 24785, 6927, 12389, 17927, 21853, 14985, 12588, 6863, 32170, 34289, 34495, 14866, 10684, 33782, 10125, 30442, 17551, 4402, 19776, 18573, 4682, 15126, 32291, 27691, 25137, 16678, 24257, 27119, 9969, 4818, 25896, 9940, 30304, 34337, 25263, 23826, 15380, 20846, 3186, 25171, 4210, 27380, 12784, 16775, 14273, 34228, 11019, 8983, 18515, 14004, 3925, 4346, 26989, 5594, 3811, 476, 30613, 5600, 20486, 27955, 497, 7825, 21363, 25311, 25244, 27787, 29138, 14929, 21929, 29614, 30878, 29552, 12333, 5969, 9325, 34351, 26629, 27911, 20957, 8072, 5459, 10014, 14966, 26484, 3100, 12983, 22838, 21059, 6547, 4162, 1053, 28729, 7822, 28829, 14744, 30698, 19312, 8644, 24117, 23112, 15153, 15693, 5799, 20508, 14071, 13971, 7073, 24741, 29777, 33132, 19240, 30731, 28196, 3535, 4662, 7329, 13572, 8617, 921, 25256, 212, 26279, 26060, 28811, 31691, 33273, 5650, 17716, 10758, 17432, 26745, 16578, 23096, 15464, 4855, 27095, 1316, 26067, 32064, 21697, 6663, 10053, 7090, 28097, 4894, 33512, 7597, 14548, 26339, 17595, 11610, 1715, 482, 12599, 34324, 20665, 7896, 30654, 4329, 13109, 11223, 15402, 29559, 33083, 22710, 5314, 25003, 2919, 32907, 18203, 15672, 19302, 25080, 11034, 18012, 25276, 22987, 9739, 2042, 28750, 11601, 13924, 27226, 2826, 13114, 22294, 31358, 33268, 12232, 12715, 31455, 13506, 6337, 12347, 32685, 6333, 25240, 9927, 26883, 18066, 25305, 29750, 21504, 15134, 15671, 26756, 11250, 8881, 12392, 2729, 32066, 29553, 23700, 23877, 8338, 32699, 1900, 31692, 7717, 31524, 31649, 13638, 4808, 9108, 12475, 33598, 9831, 13761, 17403, 13207, 18675, 21496, 16040, 7569, 22696, 9499, 4459, 4929, 22382, 11305, 34291, 3515, 31682, 18142, 7855, 32185, 16365, 9228, 31026, 22352, 10041, 21827, 5337, 4273, 1406, 24639, 32278, 13940, 21759, 23078, 215, 19927, 28512, 17319, 34165, 18776, 13706, 32915, 32144, 22687, 15727, 11559, 1651, 18174, 13684, 19483, 19337, 29246, 29170, 28524, 3492, 11788, 21905, 4180, 31763, 23765, 27640, 29152, 31339, 22152, 22869, 31878, 27648, 5047, 16312, 7189, 18151, 15597, 30188, 9296, 34044, 33542, 2080, 32576, 31068, 6534, 12520, 22541, 27049, 11167, 18531, 12440, 3408, 33303, 45, 21423, 22296, 29920, 15686, 3673, 15021, 25749, 3946, 34494, 34300, 3356, 14681, 33075, 22622, 5307, 413, 21368, 10019, 5586, 26671, 601, 31853, 18425, 27868, 23230, 18604, 22180, 916, 31783, 20192, 32810, 29687, 13304, 4687, 24088, 31533, 23809, 659, 6085, 14771, 8598, 29284, 13762, 2740, 25771, 9685, 12914, 19027, 28178, 1695, 141, 14953, 32396, 1170, 33050, 9301, 27880, 19764, 14295, 17143, 3495, 9185, 5433, 27478, 26356, 22142, 8078, 10693, 4381, 4754, 2507, 19673, 19102, 18534, 2772, 24151, 809, 32216, 33954, 4996, 7357, 31257, 16640, 32137, 28200, 12996, 11980, 9574, 10045, 17591, 14813, 13540, 14673, 23171, 32948, 10176, 26845, 7526, 21979, 12346, 15427, 19077, 32091, 31099, 32364, 433, 13264, 13986, 6790, 8043, 6093, 31400, 7510, 3569, 6759, 4974, 5875, 13969, 3056, 29879, 368, 8404, 5552, 6918, 25711, 19635, 30936, 16237, 26145, 6502, 2972, 19867, 15145, 22272, 30355, 30845, 17626, 651, 1972, 2974, 1766, 28270, 33525, 278, 3720, 26743, 22089, 30471, 817, 17268, 32525, 17073, 6259, 28677, 1829, 19212, 21374, 19173, 21825, 10282, 2394, 842, 766, 4714, 14573, 8656, 17504, 30247, 20588, 8371, 8225, 21559, 22688, 33332, 29193, 19672, 1274, 33051, 2994, 16650, 28011, 30339, 155, 9120, 6429, 5096, 3742, 16052, 20853, 7311, 27832, 12562, 32601, 22631, 4564, 14754, 8667, 32464, 7198, 3539, 10631, 4522, 10740, 28113, 20572, 9275, 7910, 12010, 28767, 21349, 11465, 25249, 13835, 21517, 22192, 31145, 12038, 26678, 27015, 14227, 10074, 571, 2934, 10450, 27311, 24533, 30178, 20532, 2322, 11729, 18382, 32703, 31892, 2093, 15540, 17523, 7964, 24161, 22445, 7728, 13224, 10305, 32160, 10667, 13479, 25246, 15622, 11176, 9477, 26151, 20590, 2860, 14539, 15280, 23772, 24327, 18648, 29148, 12872, 13051, 33097, 17676, 1739, 20325, 24795, 3337, 29819, 17902, 11593, 24781, 16235, 28739, 552, 17020, 23924, 11548, 30525, 3293, 24412, 23502, 18798, 3505, 22242, 31417, 23819, 13194, 12276, 8596, 5160, 6325, 6572, 22664, 7851, 27767, 23944, 10465, 23442, 19835, 27395, 16655, 10698, 8818, 2046, 32497, 30426, 13140, 12222, 28325, 9810, 33221, 13378, 9377, 22411, 6891, 13914, 7340, 6328, 3770, 33584, 20262, 3608, 25844, 16859, 14703, 2141, 1952, 11114, 27075, 18761, 20470, 13115, 6708, 11913, 16354, 6777, 20827, 19936, 16646, 13157, 31495, 10894, 1174, 9789, 24163, 27663, 27098, 23192, 10101, 14886, 13721, 1306, 31, 20286, 22999, 9246, 13921, 21688, 26537, 18044, 25653, 32588, 1583, 33811, 19148, 27501, 15004, 10619, 18198, 32962, 6580, 15953, 18190, 6314, 1958, 12688, 21850, 21148, 14609, 29984, 3171, 8370, 5461, 8980, 10264, 10431, 13251, 13863, 20334, 5213, 3257, 33623, 12026, 14909, 24531, 13310, 11315, 10336, 13096, 14299, 3866, 26266, 15750, 5046, 4275, 16496, 13931, 8848, 13362, 8857, 7327, 14608, 10049, 29746, 25800, 31856, 24308, 17631, 15701, 3090, 21742, 23548, 16462, 20443, 34014, 28547, 7010, 3179, 288, 10500, 29711, 28815, 22796, 4774, 5109, 15815, 11820, 25411, 29788, 6074, 34043, 16746, 5640, 18647, 26871, 24273, 30367, 13371, 11197, 24946, 13509, 6582, 15249, 15624, 23573, 13529, 2789, 15014, 24735, 30016, 3250, 15023, 34442, 16197, 13247, 20909, 893, 13337, 26958, 8566, 9137, 7165, 17629, 471, 8484, 18387, 2603, 12622, 25869, 32366, 21826, 12482, 13373, 34420, 24942, 11829, 20537, 7492, 21178, 27757, 21725, 2727, 917, 17158, 8825, 5858, 22002, 28812, 10665, 11388, 1013, 26924, 25519, 25794, 30701, 26050, 800, 31597, 10748, 23992, 12446, 2028, 22593, 1849, 9195, 6134, 8911, 15519, 15865, 16073, 12885, 30290, 27428, 15363, 31454, 3573, 30636, 786, 34236, 26448, 8568, 18750, 17792, 13890, 19399, 24370, 31127, 31392, 367, 25446, 25688, 11461, 26766, 26941, 194, 4305, 8164, 3626, 32063, 2923, 9307, 21501, 30581, 12273, 14998, 11518, 20009, 17880, 13687, 21036, 24230, 12641, 21550, 6516, 4863, 11089, 3874, 5647, 18059, 10874, 22279, 17064, 8510, 24663, 16910, 33883, 25525, 31515, 20675, 1059, 4442, 19160, 20228, 9908, 5718, 29782, 30323, 327, 29049, 30061, 13525, 69, 9464, 14899, 33224, 24355, 6709, 7628, 10540, 16294, 2024, 1933, 33016, 28572, 26909, 32853, 30183, 25334, 20352, 21976, 27569, 17336, 29739, 20602, 9828, 29745, 23010, 1727, 27418, 16498, 27761, 21995, 19684, 27024, 5074, 26016, 25330, 27200, 4769, 7239, 20427, 22673, 6088, 23979, 26281, 33513, 6402, 833, 32536, 32529, 4794, 20449, 18821, 12938, 29253, 13682, 1969, 2464, 6327, 21578, 7072, 19619, 29447, 2115, 24136, 34305, 7397, 21207, 22042, 32007, 29805, 15970, 28885, 22331, 28278, 31016, 11182, 3852, 12646, 12754, 6387, 4196, 1773, 19387, 16873, 473, 31913, 5196, 34521, 11945, 5198, 18049, 13741, 7040, 26763, 8029, 11334, 21962, 32823, 755, 12959, 32380, 23159, 17812, 33222, 32047, 23438, 31229, 11935, 29306, 3527, 17589, 31340, 19244, 4171, 23699, 27043, 4684, 5708, 12140, 2615, 27018, 6144, 18737, 10173, 5685, 10887, 8490, 16104, 2458, 32573, 4086, 522, 207, 8067, 8297, 7009, 13852, 9769, 30333, 27992, 6053, 27819, 18087, 14012, 10081, 17422, 3105, 14720, 21879, 16154, 12656, 10840, 642, 16703, 1193, 31510, 26842, 31932, 24361, 3776, 6614, 29492, 26525, 25236, 14613, 15318, 5792, 5656, 33992, 29136, 18060, 10658, 17132, 20387, 16666, 22991, 15897, 4658, 2842, 34405, 28296, 2463, 26981, 10530, 34012, 3966, 32036, 7107, 17459, 11909, 34125, 14640, 30039, 18269, 10475, 8118, 7936, 20331, 21078, 3868, 19024, 19507, 29952, 31220, 10429, 20132, 27138, 27534, 2503, 16095, 19233, 28139, 23378, 32578, 32422, 6762, 30456, 18332, 32634, 2582, 21520, 11746, 15709, 5660, 32938, 4103, 17928, 10063, 5214, 29629, 16724, 32417, 29897, 2434, 26068, 6510, 25497, 10186, 1772, 16593, 27581, 32610, 6568, 8080, 31212, 5276, 4680, 18608, 25959, 4097, 127, 23911, 18588, 30464, 4135, 29529, 32405, 5394, 11616, 613, 28611, 9726, 10568, 8221, 27931, 27165, 24435, 10164, 16755, 11728, 27750, 8175, 27963, 6463, 2381, 3481, 9928, 28919, 9538, 31062, 5847, 10231, 1217, 16577, 5232, 3200, 7684, 14289, 25123, 28539, 3762, 33245, 22091, 8271, 33770, 29804, 33241, 10753, 6411, 25011, 8992, 27020, 2691, 25461, 32224, 19940, 3832, 9387, 16217, 9748, 20742, 34199, 33537, 21921, 27173, 14724, 31616, 33506, 16772, 23434, 26230, 4155, 6302, 11827, 17925, 30512, 19983, 30873, 25828, 6203, 4070, 20991, 17244, 27411, 3097, 14358, 8743, 4784, 15392, 14030, 24847, 1433, 9334, 14675, 25845, 12268, 31249, 8030, 33654, 9291, 20016, 15927, 13798, 33209, 12847, 2267, 84, 26394, 26884, 15086, 30968, 33264, 29526, 18511, 33761, 8207, 12615, 4566, 14913, 13050, 33540, 32717, 18280, 11765, 337, 27878, 8494, 14087, 9076, 24636, 3511, 2722, 7857, 7742, 14581, 27608, 21840, 23945, 21030, 31286, 25809, 13220, 15291, 6938, 2751, 28504, 23176, 22635, 33981, 23655, 4608, 33558, 29568, 13032, 8391, 1764, 10089, 14204, 27861, 15982, 34120, 3125, 31106, 21145, 3067, 32114, 6893, 31216, 11722, 11864, 600, 26214, 30599, 24335, 14260, 12404, 33929, 24788, 9093, 747, 29505, 16581, 3604, 29607, 31442, 8344, 34444, 4181, 16021, 25842, 12074, 21023, 19325, 4542, 16770, 27221, 30459, 12815, 5679, 31178, 31357, 2743, 27424, 13872, 33297, 15596, 33636, 18623, 4192, 73, 5172, 3024, 19437, 27771, 20479, 9095, 22143, 24501, 1079, 1201, 875, 31941, 27668, 21112, 31463, 19191, 33552, 19998, 8665, 4053, 16720, 11118, 28908, 15422, 16149, 1990, 24819, 24626, 13709, 30724, 630, 29733, 16035, 532, 2515, 20148, 11206, 31888, 1452, 25118, 9097, 18917, 8731, 32148, 3823, 7836, 33841, 18965, 25371, 8086, 21113, 5770, 19582, 16905, 27497, 1950, 33129, 18214, 13635, 8050, 7673, 32346, 24833, 28868, 18474, 23881, 13710, 9230, 24212, 22249, 25158, 24722, 3926, 262, 4874, 13922, 829, 5655, 10663, 32762, 11409, 13743, 10337, 20875, 27330, 15501, 32682, 16130, 21064, 29326, 24509, 8794, 5960, 19406, 11246, 6741, 6258, 9111, 2760, 33719, 13359, 12291, 31736, 32071, 21160, 28946, 9205, 24052, 28820, 11002, 26025, 1762, 33935, 20114, 23644, 12057, 12993, 5159, 8578, 8752, 9797, 27595, 1190, 985, 5671, 923, 18193, 13200, 31427, 7113, 15396, 14830, 34529, 18007, 33694, 28575, 2209, 31883, 14304, 17584, 25652, 3798, 17099, 15611, 3731, 17769, 5986, 33771, 3291, 33533, 18543, 33119, 28445, 28497, 27921, 14862, 7635, 24269, 13475, 11770, 12555, 5943, 21169, 32292, 12944, 17417, 29101, 15038, 11178, 3988, 16000, 31486, 6363, 19063, 18076, 30406, 20249, 6297, 25864, 29706, 17405, 27071, 26701, 17944, 32238, 17860, 2239, 12385, 25549, 18953, 14653, 17850, 29133, 25820, 32270, 23229, 918, 7531, 18803, 32107, 10397, 32469, 26739, 23859, 13930, 6439, 30273, 29555, 13265, 1867, 28193, 32553, 2837, 22892, 16121, 19276, 18770, 7906, 27283, 21199, 18551, 11920, 10174, 9210, 2213, 19553, 1966, 27944, 19332, 10127, 21174, 28671, 15268, 16908, 8636, 12969, 31928, 33093, 21545, 31295, 20300, 3834, 21792, 6746, 10116, 12462, 32825, 6044, 15966, 30952, 15374, 6787, 32308, 3637, 33830, 25287, 7944, 19274, 31071, 27949, 33840, 14367, 17242, 1466, 32612, 33786, 1801, 14884, 27091, 2597, 2749, 6223, 21814, 26378, 12088, 4059, 29254, 19888, 31457, 17623, 17624, 34388, 12207, 5963, 7506, 24498, 30896, 17027, 24010, 17074, 30615, 9785, 2022, 27360, 24008, 5491, 4365, 32010, 738, 30448, 12927, 12841, 7079, 21267, 9957, 17761, 8925, 33932, 20669, 15186, 25299, 21718, 2999, 11111, 27228, 19575, 29595, 9265, 4931, 33988, 4556, 11337, 15843, 19117, 752, 22435, 29064, 11755, 3723, 26087, 14404, 10482, 12292, 28937, 24182, 18143, 20806, 5082, 26834, 24170, 24799, 24702, 16078, 8951, 4268, 20186, 14610, 7232, 22952, 8725, 4860, 30811, 25915, 4930, 25807, 25347, 23509, 7970, 3587, 3772, 21682, 4483, 21189, 12275, 4400, 18402, 9834, 27997, 9601, 8346, 20142, 18020, 12247, 27958, 33401, 11345, 3305, 28378, 25378, 4629, 17955, 26127, 3562, 16782, 33177, 6385, 22238, 10281, 31227, 26013, 358, 33038, 19002, 32436, 3788, 29048, 8381, 6061, 9399, 2250, 13042, 1698, 33270, 8001, 11147, 5947, 3512, 25928, 14212, 9575, 13582, 25683, 20453, 22335, 32395, 20392, 25570, 16272, 18979, 2325, 28409, 5873, 26471, 32552, 34421, 28148, 16335, 6765, 5061, 26324, 9658, 7837, 17531, 14312, 30300, 16931, 12908, 14352, 14141, 10403, 3728, 25241, 23454, 34021, 8142, 31605, 18003, 21606, 14944, 11923, 170, 29016, 19015, 27450, 33612, 12017, 8573, 23552, 14274, 30761, 13627, 30066, 22707, 32387, 19186, 27892, 5117, 17017, 634, 19768, 26177, 19572, 1675, 11824, 33767, 421, 27621, 26424, 6, 19800, 32233, 31470, 28535, 21457, 24886, 26920, 26160, 15250, 21846, 34419, 33061, 16085, 23798, 14383, 28018, 19123, 1817, 14058, 33476, 30565, 26934, 16330, 19524, 23662, 27660, 32040, 15358, 25132, 16017, 9423, 11177, 11186, 13130, 2400, 4444, 618, 27468, 13997, 23845, 11253, 7129, 9418, 2179, 9433, 465, 17781, 17875, 10453, 2110, 23665, 5551, 14391, 15057, 5580, 1058, 29760, 12550, 6847, 17594, 14400, 22704, 2796, 14156, 11093, 3338, 10210, 10312, 14435, 16787, 13557, 844, 7670, 30308, 20397, 27702, 854, 4562, 25333, 13124, 15870, 24441, 28463, 15675, 32751, 5870, 11327, 24743, 15453, 32408, 32200, 1390, 10890, 26830, 4462, 17152, 10920, 13724, 25965, 190, 20763, 15033, 14420, 7981, 30758, 21332, 6567, 5727, 20523, 25301, 6858, 9064, 29469, 33809, 33479, 11414, 382, 26019, 25951, 1704, 13134, 146, 10939, 21354, 22229, 6867, 19699, 10029, 13960, 19179, 8942, 11671, 28063, 12053, 14002, 9118, 10805, 30816, 27697, 29582, 16077, 20116, 26440, 16371, 3327, 19457, 27874, 32940, 2700, 2399, 31184, 27649, 24969, 26272, 22619, 21273, 27528, 6190, 2007, 27052, 11665, 29584, 8865, 24211, 32628, 1768, 4068, 7420, 16342, 4549, 19540, 21843, 6063, 23982, 34333, 18071, 13342, 8121, 25471, 9465, 32680, 9527, 33957, 10790, 11057, 30730, 29479, 14048, 14374, 21978, 7751, 26850, 24392, 3640, 25700, 33311, 22022, 19270, 22562, 19074, 9861, 28591, 33081, 28420, 3763, 13632, 17724, 32330, 28490, 31013, 12069, 2572, 26202, 27147, 18299, 973, 29642, 5206, 11278, 32742, 26910, 21294, 6228, 24820, 338, 30329, 21616, 6869, 21571, 33708, 34138, 25929, 2516, 28734, 13806, 20621, 22882, 16828, 2256, 10770, 24845, 2161, 3235, 16661, 30381, 30099, 26011, 14631, 15025, 28130, 9051, 11956, 14702, 31113, 15556, 7112, 33429, 24910, 11717, 4467, 11202, 21303, 12441, 7440, 19444, 1175, 24680, 32460, 4622, 22, 26455, 11758, 23635, 5123, 8361, 5203, 5599, 2422, 11674, 8061, 3193, 3194, 25892, 12420, 5920, 3937, 11708, 2491, 30149, 27862, 7258, 22975, 13103, 1533, 31387, 33668, 24252, 5911, 13429, 8535, 18324, 4577, 11771, 13842, 16886, 21960, 14890, 801, 33649, 2593, 30715, 5818, 30259, 994, 19952, 8184, 24437, 10519, 1388, 2950, 10903, 30770, 12100, 746, 29256, 18645, 19948, 16138, 16961, 3360, 23706, 31179, 10581, 23414, 25559, 2180, 30966, 22556, 22314, 6876, 867, 19905, 34072, 3022, 11025, 15104, 5245, 8606, 1455, 25164, 34113, 8621, 23339, 6905, 18969, 8414, 19363, 4082, 27257, 16334, 13141, 27628, 13773, 12176, 4198, 25128, 31848, 7519, 16377, 23449, 32636, 27279, 12366, 27270, 33693, 27736, 28221, 96, 2096, 17032, 28639, 3876, 18372, 22773, 26098, 9617, 27728, 4313, 21274, 32414, 3113, 23094, 34396, 24482, 31277, 12854, 11145, 20595, 31619, 28909, 28781, 4804, 21034, 18771, 22653, 15798, 9683, 21656, 16848, 12185, 4744, 19299, 8135, 16208, 9865, 10357, 13492, 27979, 32713, 20894, 20372, 3075, 19698, 28983, 10508, 7801, 7830, 2181, 6846, 9962, 3215, 33480, 8474, 11878, 2587, 21120, 3952, 22671, 13669, 8266, 26141, 15968, 31057, 18660, 19434, 24786, 3488, 24209, 30667, 20267, 1074, 22172, 16514, 15182, 32289, 26768, 350, 32557, 22787, 21433, 3202, 13225, 17396, 16842, 29883, 22065, 10571, 16685, 3006, 16418, 26152, 16652, 4278, 20807, 12402, 21238, 2478, 23322, 12528, 24989, 3005, 15967, 2828, 23476, 30161, 24225, 33683, 20493, 3873, 19238, 3213, 4326, 8632, 9998, 11120, 8023, 32873, 26795, 24222, 20106, 870, 14479, 9234, 3825, 29571, 23984, 4193, 33424, 15390, 20233, 29015, 23284, 7900, 7183, 33926, 1758, 12384, 29324, 8898, 6825, 31871, 18221, 9455, 32728, 10109, 840, 5768, 32727, 30173, 24857, 13592, 26596, 27013, 23072, 9916, 3889, 23590, 13679, 9209, 30644, 21232, 10167, 4616, 1926, 22160, 11647, 22946, 17036, 14726, 32989, 6525, 15742, 25027, 26171, 20839, 24454, 24834, 28301, 4554, 11999, 32609, 28703, 25767, 14585, 30065, 518, 32746, 6628, 32110, 30380, 12594, 23205, 28696, 26999, 4681, 21376, 11381, 7835, 29755, 15353, 5606, 23962, 9860, 29818, 16734, 21935, 25814, 28628, 5503, 28683, 17025, 26667, 11697, 11792, 19581, 30939, 4912, 6683, 5435, 15466, 19229, 11691, 31435, 18956, 30466, 4832, 11054, 19759, 12163, 5732, 11451, 11509, 24205, 8593, 718, 23149, 16016, 10002, 28689, 31204, 8499, 20644, 3594, 11552, 11819, 2286, 23988, 24126, 28879, 10488, 17219, 33036, 28806, 20744, 20810, 27275, 22930, 270, 17936, 4280, 4142, 15247, 25721, 22726, 9581, 3719, 26663, 8733, 19317, 14275, 18173, 18709, 23678, 11092, 17924, 24649, 432, 30791, 6098, 23632, 30092, 19269, 25849, 18202, 25670, 16193, 31208, 13054, 19794, 7627, 32067, 7590, 6729, 750, 25216, 21530, 12491, 1549, 6651, 10112, 6277, 14428, 12806, 4399, 19492, 3184, 17058, 19751, 9154, 16613, 17965, 11310, 27873, 26600, 22216, 11464, 24510, 4301, 19670, 11424, 16774, 32887, 28027, 3457, 32590, 33671, 27430, 11200, 34281, 11307, 3288, 7789, 32672, 18773, 8053, 21820, 1899, 19923, 6843, 20586, 14865, 26138, 33744, 24605, 19648, 8516, 7820, 12866, 28875, 32399, 17012, 29130, 9094, 19252, 3003, 9559, 28080, 8107, 28681, 17650, 17345, 30941, 4096, 33442, 17744, 21833, 20772, 2824, 29435, 7943, 14461, 4695, 22220, 14945, 23215, 5700, 6160, 9857, 27975, 14325, 15110, 13417, 4506, 27170, 14086, 19913, 16697, 2254, 22505, 33605, 3239, 10965, 15314, 24496, 3020, 1832, 31188, 7989, 33692, 9696, 30573, 7814, 23116, 19791, 8036, 29780, 24737, 18502, 10864, 25661, 16146, 11059, 27246, 29381, 27817, 15490, 11127, 14436, 1848, 16871, 6003, 15022, 14767, 623, 21344, 25590, 11368, 7586, 9657, 14229, 27515, 8922, 14120, 34415, 9350, 16793, 13985, 22539, 1564, 18431, 9925, 10222, 29398, 28616, 4605, 6255, 3506, 16172, 27410, 27458, 18706, 32106, 27416, 3920, 31824, 11096, 10085, 20432, 19585, 23618, 7713, 18986, 33439, 12803, 14310, 9982, 28421, 9045, 33577, 28207, 12997, 26410, 24002, 25450, 1741, 11777, 20904, 2228, 23571, 1257, 30891, 27048, 18671, 27062, 9020, 22910, 27065, 23874, 23733, 189, 21964, 29069, 6586, 20011, 23849, 9370, 4827, 13781, 20820, 17360, 32183, 30620, 18529, 6300, 28632, 17136, 4287, 6404, 1081, 24913, 28697, 22555, 28743, 32926, 30519, 6861, 26473, 30216, 177, 4712, 9461, 28537, 26492, 36, 15908, 32758, 34183, 11298, 20871, 12025, 23115, 13600, 15444, 30104, 9165, 32889, 8684, 29005, 10289, 23848, 24243, 33500, 30850, 33575, 18296, 29461, 10230, 13123, 12005, 32527, 11489, 27658, 19719, 97, 4466, 620, 26325, 16287, 12368, 23824, 30276, 8579, 24934, 32627, 20136, 15510, 29524, 12716, 2339, 32876, 17736, 1477, 24120, 34070, 29241, 17953, 10592, 22798, 20418, 26100, 2741, 12928, 6848, 27092, 21969, 940, 24446, 31874, 27680, 32943, 16733, 13820, 29830, 29598, 12494, 3686, 18926, 17142, 15197, 944, 13507, 3272, 10406, 7868, 11756, 19953, 27244, 16224, 4740, 27107, 8589, 11474, 4246, 23663, 23990, 9243, 25788, 13260, 32502, 22765, 19726, 33128, 33556, 26988, 12981, 16808, 10677, 30508, 11744, 14200, 10315, 4942, 23933, 26888, 24019, 25140, 30610, 3633, 31753, 31230, 33682, 25503, 1335, 21118, 727, 26499, 32339, 21488, 26957, 6783, 23470, 16867, 16469, 2025, 3182, 9255, 24613, 13174, 17427, 4077, 29720, 11301, 24302, 14568, 24522, 4578, 14847, 3390, 23256, 30645, 33877, 25984, 32072, 19580, 9306, 9850, 19263, 2011, 7028, 29532, 4512, 19607, 5038, 12484, 3978, 7119, 16284, 11804, 26184, 7602, 26790, 22829, 6923, 15998, 4584, 19955, 7059, 417, 10788, 13390, 33900, 12181, 31478, 19092, 19369, 23406, 27023, 20107, 17018, 19228, 25491, 24236, 24477, 19957, 8466, 13152, 25328, 15000, 8095, 15667, 26139, 1385, 30773, 29343, 161, 25578, 7558, 9481, 16584, 20002, 21882, 10249, 28869, 19543, 12536, 32824, 9286, 2117, 31322, 1338, 32060, 28846, 5561, 29351, 18858, 32815, 17585, 31023, 18379, 31644, 18669, 34418, 24832, 30171, 1869, 31293, 732, 9392, 33201, 4271, 11793, 6570, 6091, 12904, 7141, 34273, 1632, 30182, 27211, 28173, 33383, 31571, 21147, 6278, 5834, 27795, 11431, 28961, 25309, 21744, 1892, 4218, 22078, 7326, 24881, 7457, 32164, 31780, 9204, 21737, 9323, 28899, 14898, 13675, 30291, 1798, 20986, 28747, 14052, 20446, 20566, 4367, 24322, 17870, 15509, 24354, 14607, 3050, 5404, 13853, 2277, 14446, 28213, 26500, 24570, 26334, 21020, 32166, 17309, 30666, 31066, 28503, 4370, 30469, 1397, 12833, 27179, 4990, 22111, 33023, 23109, 2971, 11977, 20928, 24037, 1948, 9735, 31010, 28389, 18250, 29090, 509, 8524, 20834, 14952, 27972, 13893, 19578, 13446, 18377, 2200, 14921, 23749, 3080, 14821, 2937, 24728, 32716, 12419, 7058, 8148, 15231, 26637, 18742, 14656, 17534, 3916, 33872, 10218, 22243, 29157, 10778, 1345, 15629, 34489, 28889, 24465, 32885, 10030, 9512, 29201, 16508, 16560, 32773, 29832, 26469, 20306, 16495, 13005, 34303, 32782, 14098, 30603, 29721, 14658, 4845, 30125, 31784, 32558, 28790, 1034, 16575, 7027, 3918, 20888, 1733, 31109, 5602, 20445, 14498, 13867, 23431, 6844, 18946, 13640, 9953, 26222, 23311, 10655, 17996, 23410, 19691, 28801, 24312, 32313, 13727, 7719, 34424, 26144, 28258, 28374, 31569, 32510, 2725, 32508, 16833, 15631, 1845, 25610, 16400, 29023, 30289, 29831, 16667, 30741, 33818, 12201, 24945, 2903, 19899, 29385, 18901, 19525, 28568, 18775, 26099, 2084, 4999, 27612, 23729, 29353, 34195, 29013, 20691, 31334, 21275, 18396, 10660, 23393, 17738, 24487, 33778, 12660, 20671, 16014, 15858, 31890, 19174, 31386, 33400, 6240, 29394, 17863, 16191, 8559, 21048, 14732, 31031, 26802, 3552, 851, 5154, 30261, 18458, 5535, 6311, 10974, 12949, 30727, 19855, 17528, 34403, 2056, 31686, 1802, 19018, 8165, 9682, 6273, 12157, 26960, 10514, 33200, 28855, 18493, 25799, 22246, 25255, 1322, 30949, 3205, 26331, 22187, 23687, 2224, 21970, 24183, 1290, 10166, 3072, 5993, 17023, 26130, 18963, 9783, 3361, 16924, 28838, 27390, 13221, 32279, 14511, 22484, 6338, 5014, 14940, 14815, 24831, 10930, 13548, 2278, 2031, 15765, 26757, 1854, 19487, 34329, 31729, 27321, 24485, 17771, 29814, 20405, 31149, 17605, 7309, 34243, 24840, 19617, 27973, 19906, 15045, 27476, 19526, 23998, 31502, 24738, 33901, 9224, 10931, 25488, 11902, 25785, 29737, 6776, 16232, 26540, 15194, 28382, 13098, 2905, 25623, 5449, 1476, 33972, 10829, 21971, 34393, 5857, 2268, 24438, 6242, 10978, 29671, 2169, 28271, 28152, 1253, 12063, 21462, 6671, 11170, 28555, 31484, 9806, 25889, 24000, 14932, 9470, 31879, 1743, 13043, 4330, 20884, 14369, 1819, 23597, 1827, 27440, 9798, 26713, 22616, 12267, 6769, 30729, 16880, 9618, 4284, 29223, 32093, 6610, 15535, 21524, 26300, 16302, 11891, 32356, 11031, 10639, 24462, 21592, 24030, 13629, 8931, 21562, 14777, 15825, 13267, 12706, 5042, 30963, 24688, 22801, 17247, 26618, 11814, 22186, 19920, 34414, 14860, 20326, 12409, 19504, 12873, 9438, 30735, 23065, 24770, 6120, 18833, 14648, 7201, 22483, 1922, 21362, 22330, 16535, 5264, 24214, 21784, 26001, 30139, 6313, 22827, 14231, 29887, 26015, 15281, 19546, 10287, 19005, 18940, 11625, 10152, 30804, 32078, 33351, 29650, 21266, 9415, 17381, 4494, 4420, 26330, 32669, 29231, 33178, 19870, 21026, 12400, 29142, 34502, 11843, 25882, 16651, 26810, 6592, 5657, 26079, 12073, 17732, 5926, 12703, 8427, 11629, 2049, 61, 6115, 24495, 25134, 4613, 21313, 28531, 5629, 18786, 23413, 3961, 4436, 14203, 23070, 28454, 15056, 23124, 15336, 9281, 4046, 2774, 6131, 11644, 5067, 701, 10862, 31611, 28771, 16541, 8044, 14249, 5697, 6380, 5421, 17231, 4318, 27142, 4303, 29759, 20005, 541, 19669, 33996, 8796, 357, 19693, 26984, 21319, 4568, 3012, 21104, 24929, 26878, 24904, 31511, 18689, 22378, 15063, 7338, 28746, 13656, 20085, 7582, 13320, 27631, 4635, 2165, 6094, 12795, 29126, 15226, 16702, 30996, 34399, 15871, 1763, 10213, 28337, 7330, 2687, 14553, 13784, 10813, 6739, 22854, 26805, 27492, 1690, 19975, 13774, 29410, 32589, 13946, 18285, 20058, 29704, 18897, 8394, 27514, 28177, 30813, 1337, 2245, 4283, 3352, 26149, 28837, 32750, 15776, 27945, 3951, 19086, 6665, 33740, 5020, 19815, 5882, 11466, 3980, 7810, 24399, 30857, 18848, 18612, 3378, 7108, 6987, 12740, 22506, 19718, 23136, 20485, 18666, 24695, 27995, 5997, 16901, 1052, 20364, 6182, 2993, 13935, 33834, 14056, 5216, 12370, 16684, 18139, 700, 13941, 32221, 21281, 29669, 21052, 25129, 22521, 24658, 28484, 34190, 26817, 22933, 5132, 10795, 32425, 24924, 10347, 510, 16805, 13982, 33773, 29585, 1709, 14300, 21862, 10899, 17253, 30526, 20979, 4833, 18676, 8856, 19548, 23478, 24215, 29, 32442, 17147, 11779, 12427, 15463, 7163, 31638, 10097, 16765, 6207, 8754, 24344, 22909, 11082, 17386, 12533, 12129, 16653, 1608, 27998, 18500, 393, 3963, 27256, 8076, 34181, 23089, 31292, 27580, 16629, 4274, 12766, 8729, 30821, 12809, 33041, 24519, 15525, 8244, 25712, 21763, 30401, 20227, 5484, 12235, 27377, 28877, 16569, 12676, 11219, 6343, 32059, 19436, 25825, 21356, 14149, 32409, 31540, 2724, 6252, 25913, 11369, 29741, 3998, 135, 18520, 34002, 32556, 12721, 15506, 24206, 32025, 11533, 30254, 30609, 15898, 4276, 1755, 22672, 32530, 1748, 27234, 20712, 26109, 12836, 27011, 15304, 24390, 25727, 33848, 7502, 26406, 25531, 3979, 17790, 16126, 20690, 21852, 378, 16971, 22237, 21476, 7337, 2706, 20500, 29065, 21681, 18988, 17452, 30280, 7236, 11151, 23810, 2579, 6360, 8805, 4112, 25645, 17317, 27324, 16991, 5581, 5280, 1108, 174, 3700, 7671, 18033, 31036, 33806, 15775, 27479, 20237, 26724, 21406, 13325, 15777, 9961, 11514, 27224, 32315, 23776, 26526, 2543, 12618, 3234, 4821, 12724, 12687, 14961, 34327, 3814, 16892, 9980, 31388, 7759, 20063, 6627, 8709, 30796, 12733, 12931, 18793, 19025, 2312, 17086, 27304, 4698, 20176, 33627, 16411, 27762, 11212, 34176, 29168, 4416, 33970, 107, 14670, 20484, 32462, 15394, 5373, 7645, 3400, 15600, 31234, 27382, 8597, 2293, 5664, 32509, 3061, 12517, 31529, 33918, 30555, 7544, 28419, 4838, 25901, 24072, 5627, 19604, 24199, 26419, 5897, 25337, 4340, 28406, 14556, 3982, 23188, 30860, 7351, 19236, 14176, 15379, 2370, 10626, 10256, 30888, 23797, 21528, 19421, 26029, 18225, 34042, 11821, 33282, 26951, 29368, 32896, 5033, 24707, 7259, 20, 12117, 1602, 8049, 30551, 30889, 14742, 29232, 9721, 26414, 24971, 21268, 21745, 29864, 14308, 5714, 21612, 22343, 8408, 18058, 21675, 16761, 1547, 9498, 14375, 4785, 33934, 6027, 31521, 26959, 13793, 21638, 23277, 18686, 6994, 31504, 16324, 2679, 16815, 5766, 25373, 14122, 13045, 9676, 18305, 398, 10918, 13626, 17681, 31027, 28709, 4423, 22262, 435, 3902, 2226, 7044, 21259, 6263, 17283, 900, 18830, 15325, 6420, 34384, 31354, 27823, 12184, 3398, 29635, 20208, 6081, 18080, 20295, 11436, 24513, 32079, 10149, 26332, 9357, 22579, 17621, 13090, 29162, 16399, 1023, 11594, 31108, 26510, 8798, 14081, 29855, 17332, 17730, 17647, 23995, 28660, 21552, 20538, 18740, 19705, 16028, 31271, 6503, 14370, 10822, 16070, 23991, 9630, 12348, 24135, 7426, 17261, 4122, 26832, 32537, 26399, 9148, 28218, 32058, 32596, 24756, 21458, 33147, 23113, 16791, 10277, 17218, 9048, 17609, 20041, 8693, 4755, 29196, 24594, 6048, 22537, 16914, 5311, 11037, 8478, 18591, 6706, 23048, 32019, 16440, 25719, 17712, 409, 26607, 33795, 32840, 12571, 29792, 4888, 1796, 23489, 9073, 13307, 472, 16609, 10565, 5744, 22332, 27178, 26452, 3724, 22499, 23249, 54, 11014, 3688, 18317, 27365, 3168, 18466, 10853, 15299, 20921, 5841, 23904, 28824, 13830, 24224, 6793, 34227, 6519, 5885, 14773, 13391, 297, 1620, 17760, 8765, 79, 25602, 13376, 1107, 6964, 22116, 33092, 27616, 17841, 1985, 25209, 31089, 25316, 17869, 2634, 16911, 16959, 24450, 7932, 5259, 17998, 1022, 1550, 8742, 10846, 17205, 9487, 13796, 13608, 21639, 30757, 7159, 19201, 5638, 21219, 10243, 7612, 34092, 19510, 9892, 21939, 5018, 9460, 27346, 21903, 6984, 33850, 142, 17677, 17848, 15684, 20157, 21085, 11021, 20076, 9573, 23083, 18683, 32709, 21953, 275, 2628, 4868, 11767, 28001, 7322, 26948, 27087, 19144, 21293, 8724, 25838, 32641, 3593, 19217, 13807, 32563, 23014, 32175, 14213, 16743, 1668, 23069, 28365, 20695, 14258, 16546, 27705, 12820, 14686, 2361, 10711, 22699, 27753, 29243, 10794, 31453, 4452, 16628, 2074, 19724, 32568, 20686, 5467, 1376, 26224, 1493, 1472, 9155, 32698, 7343, 9643, 14877, 5513, 17160, 665, 27651, 19678, 10054, 32702, 24801, 12587, 28203, 15780, 18584, 17449, 25436, 25718, 7087, 22547, 21927, 26969, 19203, 5094, 27466, 10421, 17833, 16482, 27834, 15222, 26711, 24385, 830, 1563, 14799, 15154, 4143, 25614, 9515, 33294, 32842, 10192, 29606, 20220, 15255, 23131, 28404, 15029, 9560, 9471, 15774, 30450, 27031, 21453, 12141, 7225, 23292, 22957, 33774, 17083, 15745, 30710, 21892, 16563, 7905, 25817, 2188, 32176, 15123, 17499, 6021, 5504, 34224, 20934, 8767, 10807, 22351, 25359, 26274, 3560, 16329, 6590, 28202, 28210, 15362, 10201, 31951, 21844, 17119, 15315, 6801, 26747, 24699, 8157, 26797, 17746, 33062, 34372, 18112, 3969, 12046, 20915, 21427, 28643, 6818, 21051, 13033, 20359, 5528, 7894, 33808, 20195, 24580, 7603, 24584, 31580, 13280, 15010, 16255, 1610, 6950, 13175, 24803, 6926, 9271, 33564, 14139, 25565, 27889, 3295, 26004, 10445, 24324, 10320, 32999, 24197, 941, 25854, 29547, 19720, 28435, 34296, 32128, 18159, 28398, 19438, 21508, 12751, 23158, 19284, 26107, 25801, 2811, 3611, 879, 12418, 11164, 5632, 2221, 6060, 13233, 34266, 26681, 24220, 6454, 529, 27615, 27201, 4683, 23858, 7089, 15590, 22157, 5675, 8806, 8832, 19426, 21715, 2998, 29329, 29058, 29266, 5806, 14439, 14499, 6169, 1710, 12689, 22617, 34350, 3833, 20155, 21195, 14562, 20179, 20576, 5704, 12350, 21529, 26548, 2003, 4909, 30239, 28794, 16608, 33470, 17374, 28513, 27614, 33115, 19747, 8438, 15076, 11263, 12249, 13707, 983, 21390, 4304, 9672, 20068, 17951, 33394, 33052, 5966, 8171, 22889, 5239, 22194, 12852, 31812, 15768, 11101, 23559, 19926, 10670, 6633, 17561, 17079, 21256, 28942, 12961, 731, 25235, 25433, 28603, 14253, 13424, 28248, 25536, 27603, 30829, 8488, 29907, 1679, 22497, 26480, 18904, 22007, 4445, 21754, 9396, 28907, 379, 15112, 31443, 22789, 16434, 2091, 5916, 26126, 7425, 9260, 23934, 33951, 1938, 14599, 16260, 28368, 32286, 30930, 2861, 28369, 18272, 25573, 17649, 9483, 5944, 19135, 30921, 26882, 31852, 1234, 12479, 18933, 20223, 26329, 19283, 31632, 10118, 25930, 668, 20956, 20786, 23299, 5228, 2215, 13082, 25556, 22205, 32726, 32929, 26380, 12707, 9486, 38, 26892, 537, 27886, 4748, 19469, 25312, 9884, 31614, 17758, 12215, 12148, 30790, 33340, 18178, 30052, 4730, 8543, 17082, 17518, 22150, 14450, 33382, 5541, 34473, 19100, 13131, 14026, 32174, 25369, 3139, 31665, 22821, 24323, 4397, 8291, 8293, 7885, 17888, 28197, 33897, 33751, 22599, 26843, 18110, 31792, 1785, 7601, 30631, 1566, 33457, 21554, 7744, 20733, 24177, 19534, 2614, 6355, 30501, 13245, 32908, 29616, 18480, 18065, 30545, 24195, 19336, 23893, 22857, 33647, 22203, 18145, 27128, 32275, 24651, 29815, 29355, 31897, 33550, 11561, 21230, 22094, 3970, 21300, 16136, 6283, 7563, 16900, 10917, 23791, 9754, 230, 21945, 18491, 14315, 30193, 14795, 29600, 7335, 7235, 9755, 20459, 10076, 8300, 19767, 18871, 33323, 25327, 19397, 15582, 17505, 5485, 10612, 5846, 3416, 25545, 8646, 21626, 14335, 15232, 8218, 30634, 3395, 33344, 5887, 1348, 20280, 30671, 34189, 9996, 2820, 12439, 7156, 5574, 32808, 26716, 30672, 25341, 24590, 13526, 12761, 7850, 12248, 2410, 19743, 16572, 15350, 23372, 23971, 23704, 10775, 28789, 18298, 18571, 4561, 16822, 4511, 20793, 2437, 10938, 1623, 8159, 14132, 27051, 10050, 4404, 6737, 239, 17137, 7360, 34498, 4191, 19062, 33365, 4201, 27582, 5569, 14565, 5345, 29670, 23885, 9888, 9061, 18819, 12408, 811, 13866, 12737, 18535, 1852, 10648, 1054, 27821, 15588, 14715, 21680, 3586, 30120, 1672, 31698, 9157, 23551, 26450, 28818, 1976, 8203, 3004, 16532, 20336, 18950, 1795, 5607, 11903, 20166, 27328, 32363, 24791, 25079, 29525, 24915, 26616, 10599, 14859, 16207, 26156, 31949, 18097, 30657, 11606, 9986, 30218, 4388, 10994, 4659, 12783, 9545, 16396, 24922, 15330, 28929, 4963, 14440, 27994, 12978, 32947, 25044, 21999, 24960, 10676, 26185, 7194, 26851, 17553, 31017, 29466, 25353, 10635, 10220, 12457, 12256, 7935, 23104, 19101, 4010, 11735, 21886, 10761, 26360, 2304, 17613, 15239, 5542, 21705, 32560, 3601, 27952, 28468, 8976, 11636, 8753, 6963, 16589, 405, 28436, 2605, 14554, 15003, 21309, 24296, 10133, 20605, 22705, 15498, 1097, 31811, 30864, 14354, 26643, 2070, 22533, 23510, 25150, 32136, 14784, 24696, 17806, 7067, 20305, 13902, 28383, 19184, 788, 17226, 8692, 3383, 166, 31383, 23179, 13609, 30365, 2427, 13703, 21838, 3867, 28273, 18029, 25165, 3900, 31360, 1980, 30103, 6732, 15277, 306, 3415, 28704, 20389, 4242, 10268, 5463, 33874, 14414, 6126, 4052, 16099, 13846, 29863, 30090, 22997, 19947, 34516, 9799, 26226, 24310, 18136, 6517, 25501, 25910, 19115, 29986, 8808, 16875, 21604, 8522, 33898, 33739, 15161, 11809, 672, 28864, 30627, 8249, 18294, 14364, 17277, 14128, 22189, 15054, 14106, 27417, 7300, 21874, 21012, 32084, 22223, 19808, 16423, 22927, 13433, 28954, 5445, 197, 4111, 12123, 22576, 1552, 29280, 10018, 12165, 34316, 30309, 6700, 22754, 10015, 10625, 4737, 18895, 23805, 17021, 25935, 23602, 12124, 31214, 9256, 33187, 20363, 14603, 70, 14683, 19059, 15192, 4147, 11161, 20240, 23591, 3939, 6553, 2920, 7511, 18263, 26054, 21487, 7653, 23890, 989, 10187, 26733, 7349, 21221, 1580, 19874, 19400, 12525, 1103, 31756, 28030, 17521, 15316, 26836, 33386, 23227, 10924, 24855, 30699, 33964, 18077, 5019, 33152, 9289, 29443, 15663, 8360, 9225, 4231, 22510, 33961, 8973, 32155, 22775, 32099, 30097, 2018, 33078, 19583, 29496, 3829, 20257, 22550, 5436, 3436, 21804, 7348, 25480, 9262, 9407, 33111, 12941, 11472, 5316, 12077, 14425, 30153, 987, 26755, 28556, 25795, 11040, 23201, 15523, 17729, 21699, 31152, 16468, 26208, 31697, 11890, 19234, 4151, 1747, 4341, 1837, 19001, 11797, 33010, 17477, 31822, 10091, 14623, 6587, 34283, 17847, 10501, 29596, 28036, 4875, 4401, 31316, 29519, 10224, 9405, 27116, 10672, 32723, 19189, 11430, 28778, 3454, 17959, 25768, 24870, 24325, 1621, 26456, 17883, 31754, 3350, 29729, 11743, 5298, 15965, 15730, 970, 7074, 10140, 28623, 31581, 5064, 29025, 13026, 10845, 7625, 4425, 7920, 187, 22559, 33434, 3648, 2551, 7224, 28940, 22955, 15545, 26939, 3261, 31782, 12295, 16118, 14765, 21121, 11349, 16362, 17637, 24540, 24239, 19845, 24771, 134, 2477, 18070, 24407, 8417, 21733, 542, 18844, 33630, 29884, 25032, 20222, 12390, 23329, 24353, 3076, 10522, 15178, 7489, 31546, 6515, 10471, 8413, 13975, 30127, 15088, 29110, 21283, 3525, 29440, 14202, 28526, 33749, 33843, 8696, 25656, 2897, 1464, 23212, 26771, 9116, 6822, 21213, 18756, 30490, 6125, 28246, 22865, 26490, 16510, 16611, 14534, 17479, 30166, 24248, 21710, 2440, 21330, 29489, 778, 6417, 7562, 24806, 21766, 26513, 17741, 18677, 19829, 31000, 10381, 17399, 23025, 18340, 17062, 591, 34029, 12396, 9918, 8357, 30341, 2607, 9229, 1891, 30210, 13473, 7427, 10884, 31987, 19226, 11511, 10897, 13069, 17947, 300, 4516, 6497, 20740, 20156, 18548, 4363, 9085, 15702, 32740, 9474, 11303, 19532, 1225, 1910, 10607, 8280, 9347, 17395, 29299, 8934, 32276, 33622, 6350, 4504, 12356, 9080, 18530, 4342, 4244, 31326, 11543, 21809, 1873, 34313, 24168, 2248, 23919, 5962, 17824, 20710, 7212, 6578, 22084, 21057, 20049, 8797, 17554, 18266, 7243, 29809, 3613, 26864, 3397, 23330, 9757, 5990, 25672, 13201, 10221, 34279, 14504, 7310, 32032, 5186, 33450, 10208, 30346, 1012, 2508, 6977, 9300, 14968, 24671, 11883, 4841, 2617, 14394, 8869, 3555, 10580, 5538, 9922, 19251, 3376, 15438, 34066, 1510, 7481, 5624, 23039, 33322, 16932, 23047, 26020, 18536, 10386, 29692, 3418, 3456, 30454, 27986, 30511, 936, 3214, 6535, 30596, 32282, 14155, 756, 1155, 22463, 1571, 23781, 13501, 22768, 22069, 754, 25745, 34101, 10855, 32389, 16025, 25103, 13900, 28110, 4300, 28519, 7328, 13116, 19309, 30586, 30895, 21071, 17722, 4856, 10972, 18916, 21209, 9679, 34257, 16444, 14261, 2342, 23050, 20133, 34514, 8927, 4805, 13886, 15794, 12031, 33947, 27266, 24721, 1140, 27904, 6367, 17342, 7412, 32340, 27394, 29026, 27126, 18619, 26760, 13843, 13195, 26303, 30250, 7439, 28619, 29119, 3048, 10901, 12995, 10226, 23629, 4917, 2148, 1056, 30669, 14323, 11899, 10683, 15019, 29786, 28525, 24748, 11342, 18601, 28851, 914, 32998, 32933, 14018, 9467, 12197, 30904, 16665, 13818, 12080, 24757, 9197, 10324, 16725, 23306, 2255, 4669, 10752, 30922, 2398, 15097, 18739, 33824, 32848, 20713, 22833, 29457, 27777, 2932, 24842, 30447, 6604, 5402, 4371, 24268, 4173, 25464, 23705, 11974, 31424, 2938, 6054, 27684, 26696, 2818, 23970, 14801, 8286, 33991, 3958, 19404, 28651, 25167, 26630, 21998, 14995, 15487, 18322, 20245, 16274, 12382, 9503, 5388, 20279, 28429, 12041, 14580, 21439, 14621, 4105, 24751, 5804, 20460, 27598, 26037, 4788, 33108, 31226, 14902, 28161, 33393, 9114, 9270, 12337, 17389, 32096, 29917, 2419, 10096, 16234, 12091, 19265, 28172, 3082, 24280, 5234, 25599, 10358, 3298, 23963, 7370, 28514, 19649, 33406, 6166, 15976, 6593, 4543, 25791, 3023, 10303, 9926, 17329, 961, 25750, 11203, 31159, 13226, 27800, 8950, 11562, 33065, 29962, 31798, 28366, 6809, 11429, 10426, 16561, 26567, 9704, 30111, 5788, 27744, 32922, 16116, 467, 25505, 32532, 3597, 20543, 10714, 29835, 25878, 27920, 2320, 20042, 16943, 22218, 15660, 3058, 13731, 21486, 18135, 20097, 21655, 34135, 14806, 2721, 2912, 8498, 21762, 8411, 5767, 20190, 156, 4054, 2778, 16964, 4585, 8807, 3850, 11280, 15887, 18022, 6816, 8715, 891, 6755, 13607, 2911, 17975, 32073, 5942, 27597, 4792, 17210, 34221, 10854, 29599, 29909, 5699, 18564, 30624, 7227, 28770, 20075, 531, 29342, 21641, 12855, 14230, 28530, 26758, 26545, 3521, 8345, 10967, 10682, 31183, 1844, 2673, 20756, 32817, 25883, 5980, 10332, 24640, 17995, 13301, 7695, 26923, 26554, 14278, 15782, 4116, 8355, 16614, 33726, 18547, 8958, 7088, 15636, 8219, 19618, 31139, 1158, 30945, 13527, 30275, 32644, 9217, 27082, 11688, 31140, 26720, 32338, 15001, 29924, 24138, 122, 27811, 23456, 10878, 28338, 2190, 18230, 10098, 10958, 13046, 32512, 2055, 7386, 2645, 15909, 18156, 19946, 16978, 33212, 33257, 7917, 23507, 14503, 4852, 30762, 9723, 4760, 17451, 10562, 21519, 13421, 23621, 10416, 34541, 20714, 9603, 4643, 20320, 12473, 25752, 29194, 28710, 12072, 28963, 33216, 2806, 30026, 26088, 5430, 2125, 27310, 5194, 34031, 31654, 6625, 25789, 17135, 2134, 16595, 10114, 4331, 1846, 9191, 399, 33079, 25017, 18048, 7250, 21031, 33405, 34167, 18257, 27355, 14252, 14022, 25292, 25059, 6786, 14840, 15342, 15171, 12698, 26730, 32441, 10988, 6315, 5356, 21328, 5011, 31251, 2990, 4113, 6849, 7549, 6378, 11488, 3357, 24955, 24727, 1356, 10293, 28654, 5084, 25210, 7292, 28826, 24326, 16890, 22538, 7620, 17373, 24764, 9714, 1700, 30946, 22902, 8259, 26584, 16425, 635, 13642, 16777, 17415, 25509, 28041, 25100, 8732, 11086, 17939, 23948, 17751, 33284, 33057, 2988, 34215, 11962, 3888, 7278, 28469, 9420, 32921, 22101, 30787, 13387, 5757, 31623, 23708, 34054, 5181, 9448, 3013, 9437, 30400, 13488, 7873, 8557, 34191, 29640, 21115, 7637, 9900, 23248, 1325, 21247, 17428, 23921, 5231, 15989, 22914, 18508, 5040, 8329, 19318, 3010, 22310, 8294, 13505, 8231, 22458, 32140, 21408, 28351, 14922, 16708, 28251, 31120, 24110, 5604, 14037, 33563, 32757, 27756, 16675, 24116, 23761, 12103, 12620, 25560, 8288, 8130, 24107, 5413, 3211, 13410, 14502, 30270, 2748, 31009, 8933, 3147, 18820, 6554, 27665, 8144, 30999, 22029, 21966, 28601, 15144, 27967, 6913, 15411, 2391, 16331, 25498, 32303, 39, 18254, 33159, 18031, 4547, 9044, 9584, 5738, 11231, 10051, 17868, 31530, 27522, 26135, 19304, 15289, 6349, 27453, 28835, 32859, 17368, 26402, 4256, 4593, 11063, 19255, 29388, 8462, 24996, 30140, 31319, 31656, 34491, 29300, 12458, 15557, 28334, 10877, 31452, 17129, 24264, 14361, 8307, 20494, 17552, 33042, 29052, 23346, 441, 29787, 31542, 1542, 13928, 2505, 18052, 20473, 11529, 27633, 30351, 26175, 19761, 13443, 14772, 19495, 30288, 32725, 20010, 16128, 9775, 2688, 30677, 34046, 15332, 30769, 21589, 4347, 27129, 31973, 6204, 26372, 31862, 24336, 23079, 10870, 31688, 25095, 30318, 11284, 12238, 19639, 32477, 4327, 5035, 3394, 29602, 19098, 30108, 27807, 14341, 85, 588, 25773, 6702, 22785, 17856, 27891, 19692, 17404, 11886, 3816, 9847, 25948, 21481, 28424, 3095, 8317, 27829, 4983, 10356, 8726, 23375, 16307, 26808, 27186, 26336, 21942, 3449, 26819, 33274, 7500, 19202, 20845, 2756, 30439, 34035, 1821, 10600, 31469, 17484, 15121, 1077, 27677, 32112, 20378, 25320, 6494, 32582, 9533, 18567, 24357, 23317, 27288, 820, 1543, 8536, 22685, 16474, 31596, 23369, 17837, 19876, 33161, 33146, 19402, 21024, 4873, 33845, 17127, 27225, 29283, 17822, 12918, 12557, 3107, 27361, 10285, 30296, 22398, 18945, 28923, 9462, 30679, 22419, 32704, 1019, 31954, 9227, 25725, 2373, 31094, 590, 2353, 23178, 11740, 27448, 10003, 30049, 24961, 12909, 5087, 6410, 10987, 4767, 2823, 13575, 22197, 13574, 33396, 3581, 14293, 11099, 13293, 14035, 172, 20172, 10909, 14579, 6412, 11107, 4403, 14165, 3588, 7436, 5427, 29946, 14630, 33330, 24229, 29950, 3478, 6170, 10533, 34527, 12891, 1537, 25633, 20635, 21175, 14084, 12244, 33025, 17241, 24744, 18281, 18553, 31990, 29310, 9975, 33579, 20244, 21593, 31245, 14595, 18208, 15366, 15839, 20025, 18609, 28434, 10512, 2548, 11937, 22871, 2692, 21265, 18436, 3159, 19929, 7767, 23937, 1792, 18197, 33184, 27053, 19305, 27017, 19333, 19798, 33514, 13606, 17524, 17217, 30398, 10021, 26074, 12206, 9514, 25667, 30319, 26251, 7149, 20169, 5976, 33703, 11532, 31287, 9843, 9777, 14222, 13619, 33464, 12042, 5635, 4091, 16509, 7621, 27213, 15074, 23752, 32393, 24314, 13126, 8668, 17388, 267, 24079, 14827, 19961, 20414, 32069, 11190, 23197, 7831, 22273, 13151, 20396, 611, 18888, 7967, 13722, 16074, 9902, 24332, 7155, 27172, 10010, 24316, 15770, 23628, 15078, 1265, 4427, 17185, 2812, 10656, 24436, 1917, 264, 704, 26823, 6518, 26762, 20911, 26788, 5961, 29314, 9645, 7006, 24104, 196, 9101, 6672, 19577, 3210, 19281, 25885, 25782, 30221, 27742, 12335, 24917, 32086, 10940, 8779, 17015, 15944, 22041, 23407, 16460, 7921, 16225, 16845, 33407, 27721, 8383, 8821, 27529, 8299, 34492, 21035, 23382, 14190, 15573, 26796, 13989, 10760, 3143, 25690, 783, 3802, 23586, 10240, 3805, 19418, 20755, 33787, 2809, 25269, 10975, 20003, 14739, 11815, 13346, 30899, 7419, 7269, 26183, 17574, 287, 31670, 12634, 23731, 2062, 23493, 29849, 19744, 202, 27449, 21547, 14089, 8310, 20850, 31374, 34322, 372, 18389, 13725, 7493, 14154, 17583, 20997, 7202, 605, 9631, 34118, 31789, 17673, 30252, 14736, 19860, 638, 28331, 10642, 15599, 736, 27893, 8981, 29184, 27403, 8416, 9997, 34387, 19736, 9456, 2697, 11680, 3011, 524, 21179, 8461, 27241, 32126, 33116, 27802, 6564, 19527, 19040, 31167, 25876, 3260, 5127, 26793, 33949, 28597, 33571, 11873, 32229, 15505, 9429, 4281, 29611, 2098, 30180, 18923, 4699, 22824, 6897, 23547, 13955, 28816, 22303, 6246, 13202, 24825, 2316, 4914, 32531, 27162, 33207, 5243, 32660, 32927, 2761, 30692, 20117, 28020, 8550, 22744, 14422, 19547, 21812, 9517, 17836, 19836, 15562, 20080, 14947, 16814, 26443, 29577, 12413, 27202, 25580, 1089, 33198, 9609, 2122, 13855, 2103, 13360, 5909, 7016, 24978, 32468, 2753, 5686, 7799, 11490, 28073, 3203, 26294, 5475, 13688, 15120, 13823, 12926, 12048, 454, 12509, 3500, 13584, 33701, 14454, 32219, 24807, 6457, 20071, 30434, 18369, 27294, 4710, 25157, 34448, 341, 15986, 32335, 18082, 223, 30422, 7263, 11900, 24563, 19307, 11658, 370, 29483, 14569, 24174, 489, 18289, 8872, 19068, 31836, 2223, 18209, 32349, 25054, 22561, 16933, 8611, 6112, 22968, 9472, 11619, 24204, 29826, 26933, 27623, 7075, 9709, 26046, 24736, 4455, 22338, 20284, 14543, 22027, 16893, 22125, 1765, 23258, 16397, 31368, 3975, 31953, 5140, 9175, 7749, 8540, 13528, 33569, 19683, 22923, 21393, 31609, 4396, 24561, 7002, 4474, 1751, 30234, 27166, 2343, 7877, 32008, 2357, 14617, 29077, 24042, 32520, 29824, 29145, 21442, 9956, 15459, 10636, 5701, 28886, 28995, 8054, 32974, 28988, 29696, 20567, 15676, 31564, 7588, 31246, 29674, 3541, 13814, 3862, 4536, 26049, 28508, 5202, 26775, 31362, 11351, 5043, 16258, 3537, 17508, 22087, 12297, 10342, 25486, 15443, 24935, 2129, 2961, 29006, 28417, 7725, 15210, 2416, 26655, 24405, 10417, 29073, 7459, 26578, 17739, 31384, 8714, 28542, 7770, 3412, 18590, 24602, 16256, 7757, 11435, 14594, 10536, 27772, 28480, 20522, 4247, 19168, 21770, 19859, 19346, 29697, 12246, 31283, 24724, 285, 31555, 7834, 22918, 703, 24375, 15405, 4127, 13723, 11653, 26314, 10615, 32246, 10158, 22852, 959, 21098, 30897, 1080, 18053, 1929, 8826, 27276, 23433, 29063, 7434, 25898, 20423, 15105, 16952, 25483, 14303, 21944, 34161, 7919, 31773, 23808, 19914, 20481, 953, 9614, 14857, 2653, 22232, 29277, 12411, 23674, 14798, 29347, 23946, 21324, 47, 27754, 30812, 17376, 23469, 32984, 26524, 28828, 20060, 30011, 15641, 27666, 1904, 25090, 12616, 14774, 17526, 3106, 8273, 26967, 6147, 5516, 33495, 9075, 4087, 25010, 14661, 10427, 30663, 25071, 15183, 12519, 34515, 29508, 11015, 15228, 1930, 29021, 4678, 11371, 9761, 2591, 31610, 27526, 17166, 29008, 1202, 1800, 3955, 32014, 19994, 16305, 23681, 10962, 19075, 16792, 12940, 26225, 34115, 2332, 11225, 27953, 27131, 33022, 29772, 23882, 22839, 7566, 16101, 20666, 19644, 14933, 15528, 251, 5833, 21197, 24083, 2594, 23041, 31323, 33438, 15253, 24349, 22117, 15963, 24262, 3473, 16097, 6388, 12849, 21645, 28622, 7643, 25988, 17154, 4646, 27844, 9150, 26069, 4236, 14287, 15583, 11733, 12199, 18132, 16776, 26497, 26881, 26725, 1296, 17909, 18392, 17600, 3283, 29339, 22158, 22992, 2068, 19419, 112, 15270, 27507, 29354, 11866, 443, 29795, 5588, 1628, 5460, 2840, 7062, 31369, 31620, 4364, 3216, 3942, 18852, 28776, 14786, 29096, 32404, 18522, 32878, 3117, 17314, 19473, 12726, 13581, 25426, 15837, 28477, 26285, 16368, 9361, 11736, 22224, 32065, 20360, 30833, 23345, 23013, 8482, 23420, 34482, 8322, 20645, 521, 21709, 25924, 22652, 2242, 19959, 31198, 8908, 18038, 6980, 29178, 31001, 5032, 14040, 5134, 31049, 9250, 29867, 9327, 9802, 22569, 28582, 28267, 32666, 23959, 17235, 730, 21857, 23144, 2854, 2432, 20175, 21885, 19322, 22266, 10348, 1889, 15047, 16839, 25041, 4652, 242, 8257, 6664, 18283, 32505, 23289, 24587, 16345, 10044, 27723, 27910, 18651, 9450, 5814, 32646, 19606, 16525, 4576, 14161, 14864, 12429, 26043, 6662, 7891, 21888, 8125, 12214, 17167, 14201, 22676, 2032, 14455, 8713, 13768, 14313, 12135, 24165, 9722, 20865, 16599, 21941, 25780, 17320, 4306, 28381, 7229, 7063, 7629, 3870, 13268, 27163, 29475, 5408, 13532, 8936, 8862, 7368, 29332, 28550, 30463, 9058, 9332, 1082, 12774, 16504, 21824, 3499, 7821, 31522, 26486, 16573, 31243, 11116, 21198, 31349, 14210, 30830, 8509, 25070, 5840, 450, 28933, 6119, 33007, 23898, 442, 34475, 5954, 34034, 494, 29206, 16519, 25655, 5518, 17209, 25723, 1122, 8909, 18788, 5275, 25679, 28360, 11440, 26531, 4286, 34342, 18117, 8334, 9644, 34352, 6188, 30238, 26598, 17225, 26062, 26449, 5923, 27939, 31080, 17522, 25226, 34202, 14943, 28335, 1, 34249, 33486, 29200, 28186, 8089, 11296, 32005, 12391, 2715, 7828, 21860, 13538, 29989, 13698, 29433, 4796, 2496, 24351, 819, 6484, 21993, 5908, 722, 11882, 14033, 22755, 34368, 18920, 4946, 23867, 24099, 29269, 21311, 22628, 17946, 27050, 24720, 8654, 29774, 26737, 28847, 10481, 15772, 28610, 4285, 27402, 14992, 28266, 22447, 8939, 13646, 17455, 13968, 30307, 1663, 32914, 9241, 9511, 28299, 10526, 8824, 23156, 21081, 24599, 21990, 26031, 22394, 20313, 4503, 11957, 14406, 30416, 29127, 7844, 23282, 5010, 17809, 24476, 4749, 25857, 24095, 29636, 27798, 19810, 10393, 536, 1104, 26735, 2869, 19614, 30473, 17252, 32094, 33462, 14982, 16466, 19965, 32285, 33742, 27314, 19904, 30807, 24938, 18364, 15426, 8701, 8367, 7707, 29484, 32329, 15524, 12759, 31247, 30584, 573, 12114, 23397, 25121, 20973, 782, 7276, 20368, 6884, 19277, 13197, 12760, 18602, 30494, 25248, 27548, 14854, 19799, 2803, 21708, 3656, 13595, 5571, 12062, 34068, 2047, 16461, 1853, 6294, 33492, 2844, 26094, 14888, 6649, 10419, 3135, 1298, 16086, 19440, 19230, 29062, 360, 12070, 23767, 21968, 12085, 106, 5238, 11311, 11439, 12785, 4115, 8777, 9478, 16956, 11830, 7120, 15756, 6107, 28128, 16091, 13591, 21133, 15516, 19185, 11308, 26006, 21226, 7745, 23615, 29794, 13143, 1859, 10131, 12171, 19605, 25346, 29808, 20174, 6489, 28799, 30987, 30209, 8229, 21437, 19341, 2387, 16881, 13739, 32909, 29390, 1272, 26750, 14175, 4358, 31416, 14065, 3923, 20535, 30529, 22845, 3017, 9700, 27470, 1982, 6978, 9358, 2928, 26721, 5119, 17963, 27149, 33305, 27643, 18411, 5347, 2595, 17355, 8425, 19571, 34499, 3614, 24066, 31744, 23209, 6155, 33482, 9057, 5177, 8631, 22467, 7363, 470, 17348, 23694, 22124, 19615, 12522, 5558, 26400, 23643, 32444, 33021, 11180, 29766, 33141, 11475, 15872, 27238, 24404, 12535, 14163, 20584, 29803, 1443, 22514, 27297, 26715, 17176, 3038, 30231, 12920, 16349, 7619, 28007, 24393, 1101, 26504, 30913, 25772, 27145, 12012, 9207, 6381, 20749, 33768, 1239, 8308, 31552, 20688, 34379, 21143, 32621, 33324, 9571, 15856, 5219, 12930, 30022, 31814, 15401, 3865, 32307, 12952, 16596, 4042, 6073, 16485, 9135, 4208, 22994, 22825, 26692, 25918, 18075, 7699, 8775, 6229, 1931, 7654, 2431, 32240, 10331, 636, 33903, 24421, 20096, 33072, 18806, 23891, 30428, 25925, 32002, 23600, 22493, 4636, 24859, 33436, 18518, 31114, 26566, 31459, 32427, 34106, 21096, 12269, 15580, 24653, 30190, 14738, 10099, 13471, 23567, 21515, 17416, 20898, 34241, 10146, 25564, 27185, 33640, 19035, 22639, 2570, 27808, 31549, 19475, 27079, 23231, 12796, 13832, 5069, 15564, 10888, 33379, 3859, 15007, 10086, 31902, 20994, 30005, 7854, 1960, 24928, 19505, 30855, 27157, 7164, 26891, 6141, 18539, 21904, 16311, 30322, 31466, 7182, 32156, 8739, 2984, 30017, 28786, 4308, 30665, 14812, 19372, 21028, 20353, 27426, 9572, 23844, 17759, 20954, 27261, 25926, 890, 7168, 26799, 33233, 24016, 6976, 5306, 9948, 27215, 22809, 9820, 28735, 34177, 9867, 18744, 15589, 32353, 25542, 28585, 17527, 22874, 6195, 26342, 33634, 32611, 17107, 7013, 5526, 13064, 17187, 23165, 30034, 25953, 19458, 31939, 26759, 10062, 13988, 10355, 12883, 16780, 12552, 28233, 10976, 34315, 26860, 34207, 32820, 14442, 31796, 28911, 5223, 5927, 30509, 26489, 8039, 3629, 10279, 17940, 27635, 28195, 6057, 13744, 6128, 17181, 31418, 21659, 4202, 26003, 15302, 33074, 17131, 24315, 2348, 17747, 3715, 6492, 26647, 16248, 32288, 18641, 1152, 15008, 12972, 20902, 12357, 26169, 26345, 30399, 14171, 8174, 33336, 12364, 4355, 8638, 3427, 10473, 19976, 22355, 28711, 25178, 29007, 8173, 28380, 17788, 27573, 5649, 14655, 21348, 1888, 26320, 16088, 18541, 30568, 32670, 13534, 26893, 9055, 33621, 17393, 28294, 22255, 7226, 30831, 12934, 6083, 18464, 28057, 11341, 3813, 27006, 4762, 12851, 2547, 28894, 2575, 9612, 1323, 26131, 27118, 1150, 18699, 31635, 29580, 5260, 11811, 4387, 24779, 34364, 32481, 16601, 4829, 27108, 33127, 33153, 33189, 21735, 16830, 17663, 4492, 7380, 8260, 22520, 4686, 21102, 15089, 16876, 18814, 18401, 4419, 145, 16954, 24614, 30916, 33163, 5118, 33413, 27059, 23929, 10695, 27792, 5377, 23046, 15269, 962, 28898, 15581, 22481, 31148, 21845, 272, 25804, 12339, 9872, 28247, 34431, 7390, 7523, 5967, 4440, 24712, 18984, 17248, 6194, 3372, 30799, 29738, 17470, 30522, 11291, 13920, 12265, 2538, 22756, 20408, 4700, 5862, 31727, 2730, 31607, 30433, 28272, 20937, 22154, 6285, 1503, 32981, 12843, 27899, 27619, 9509, 32230, 16788, 34028, 14486, 21574, 23745, 20655, 10432, 11959, 28426, 25394, 3644, 30203, 24999, 34242, 6930, 10402, 13586, 31024, 23857, 12231, 1977, 19992, 15288, 7509, 23232, 6496, 27151, 21706, 25087, 2143, 18747, 32184, 5726, 33766, 4693, 4989, 4312, 13981, 24156, 28344, 11790, 10738, 5115, 7554, 33024, 5092, 31940, 9910, 21620, 2601, 4228, 24977, 640, 12311, 20900, 10716, 25014, 18410, 10792, 5760, 16823, 5121, 30783, 8376, 24725, 5861, 12320, 9909, 29730, 22337, 33613, 27981, 32391, 3062, 4993, 10487, 12162, 26271, 14544, 11085, 3663, 14216, 29520, 24213, 10520, 29744, 2647, 20188, 13456, 2568, 14301, 2124, 29961, 9313, 6745, 29581, 7839, 28427, 11589, 15361, 29763, 13714, 34263, 18456, 4481, 17397, 9169, 17768, 10448, 25159, 27189, 7860, 7691, 19817, 1141, 1392, 10814, 23672, 18385, 11362, 7308, 5106, 28921, 11143, 9933, 31602, 22452, 27909, 21813, 2043, 24753, 16094, 23876, 14869, 14271, 7987, 9934, 26048, 30191, 22305, 16380, 6249, 25111, 32674, 19752, 11399, 7557, 18351, 32969, 2247, 27366, 34479, 15744, 26205, 26605, 7132, 1697, 31517, 19944, 5338, 16228, 912, 13052, 32003, 20652, 1255, 8328, 16433, 10486, 25122, 24558, 22856, 28358, 7553, 15348, 16681, 19933, 13486, 26316, 10330, 31012, 29811, 16075, 21475, 30223, 17441, 1530, 2282, 30032, 5058, 27212, 17660, 19910, 16923, 13211, 29736, 3983, 15922, 15357, 14083, 25872, 19839, 19886, 11573, 6267, 17879, 16303, 19499, 3111, 19564, 27097, 520, 3406, 34067, 10556, 14160, 18079, 6853, 28236, 10216, 23314, 27260, 16677, 31528, 20498, 23312, 26458, 20167, 8214, 22318, 20474, 23605, 12221, 1410, 18891, 13405, 19595, 23307, 23295, 31633, 13461, 32122, 15810, 8186, 22974, 4446, 3000, 28718, 20517, 29910, 30349, 25824, 3878, 1825, 17227, 12793, 17230, 20783, 25735, 24159, 28634, 28731, 12110, 27801, 7634, 5771, 18496, 29871, 34131, 27570, 24372, 17246, 17625, 24070, 3232, 23240, 11675, 12692, 23463, 15767, 16566, 31063, 32172, 33748, 17840, 24841, 23906, 33106, 8198, 4890, 1726, 5148, 33674, 9454, 3084, 26503, 25470, 16373, 3934, 9117, 9724, 739, 25395, 20882, 30413, 31869, 523, 8013, 10979, 31881, 825, 24860, 29291, 6537, 22731, 29234, 13146, 25834, 20862, 10744, 20564, 31551, 7861, 11403, 13435, 22252, 8846, 11320, 2953, 31683, 30924, 33650, 16181, 20242, 25905, 6010, 23354, 4844, 30151, 1458, 15670, 604, 11115, 18967, 22950, 16542, 12574, 27339, 29783, 21316, 7106, 3894, 30094, 8886, 34164, 14984, 16744, 20789, 22661, 19315, 23732, 26971, 29724, 5904, 1517, 27971, 5547, 21346, 31291, 8680, 22105, 8666, 3831, 13809, 26564, 11747, 3592, 27204, 31673, 9977, 21305, 15977, 31743, 26416, 29649, 19097, 17208, 32466, 23423, 675, 12245, 14410, 27102, 11209, 4324, 14948, 15099, 20362, 30237, 7982, 31526, 22700, 8836, 4110, 20745, 5669, 24545, 19623, 17215, 26005, 21082, 27110, 22717, 11979, 8155, 19522, 10719, 20386, 9839, 33417, 23717, 4227, 5790, 5612, 23466, 15213, 13850, 12130, 17292, 9890, 27046, 1962, 15116, 27781, 10985, 25264, 20440, 10879, 33170, 17236, 21600, 22044, 3339, 1213, 27741, 26053, 30920, 9181, 17267, 25414, 2448, 31201, 11239, 21500, 3032, 27653, 14453, 15647, 25863, 17967, 24941, 11189, 29544, 13873, 21994, 309, 31482, 12000, 10964, 28640, 75, 34091, 24069, 17, 29790, 29651, 9102, 29261, 8823, 28190, 28803, 5363, 10697, 25097, 8393, 20265, 15934, 8265, 23773, 34254, 15532, 24755, 20193, 21703, 14695, 5053, 12045, 26036, 26894, 8420, 19290, 6304, 19506, 22070, 33697, 26741, 9035, 27105, 20490, 10435, 29757, 744, 1461, 4910, 992, 8034, 9562, 33576, 16721, 24549, 880, 22204, 24643, 17159, 27983, 19227, 5844, 4746, 3804, 32150, 7694, 10244, 28857, 11251, 18086, 5689, 21704, 15800, 9182, 26613, 23399, 30169, 30870, 27667, 21152, 13567, 18455, 32952, 32567, 24709, 19056, 14629, 16732, 4600, 25524, 3025, 4315, 3556, 8421, 13335, 29957, 10017, 15733, 6121, 20482, 10650, 2995, 9973, 28751, 21997, 21314, 9079, 6797, 6713, 12683, 13621, 4421, 25873, 18621, 11941, 21091, 3494, 12818, 33600, 1809, 1194, 11001, 14779, 1368, 24957, 27217, 4432, 23608, 16898, 29996, 20855, 10746, 17949, 12204, 26887, 10470, 28953, 9284, 33678, 7048, 23267, 3775, 6015, 3452, 30081, 25836, 23450, 30785, 28638, 25957, 12601, 11074, 25117, 33095, 19297, 12911, 16769, 9668, 30332, 10317, 24232, 16210, 20051, 23783, 25206, 4073, 2915, 32337, 18987, 26651, 33979, 6569, 32398, 21790, 7432, 18842, 30766, 32719, 17811, 2072, 22146, 21542, 7552, 25903, 1072, 22225, 12971, 6234, 2187, 4726, 20036, 7618, 14114, 10570, 11191, 14841, 14718, 17931, 25021, 13865, 22348, 32159, 20057, 1593, 22177, 23653, 30778, 2933, 29698, 3572, 16920, 15911, 7528, 28881, 17579, 32747, 6862, 8816, 16058, 20093, 9551, 14941, 26872, 8999, 9523, 20879, 34136, 27519, 33063, 23914, 10413, 4270, 34037, 23424, 29938, 23506, 33019, 13104, 1590, 33234, 12316, 4992, 7631, 33526, 7762, 5831, 26459, 17322, 29214, 7845, 18100, 12033, 15039, 33121, 2766, 11580, 8048, 17829, 13523, 32783, 11912, 3681, 737, 30335, 34139, 11373, 2714, 34523, 2592, 24867, 92, 22823, 31346, 25839, 27262, 25108, 28035, 26439, 23712, 4152, 4167, 22156, 19757, 26705, 16654, 25783, 14032, 12194, 14563, 24371, 31395, 5913, 23290, 31351, 13078, 6092, 15608, 8971, 12226, 13191, 12539, 21958, 30350, 6539, 13136, 3987, 33871, 21492, 32925, 11855, 23907, 16455, 28757, 16935, 17182, 9354, 22835, 19573, 418, 8835, 26727, 1815, 14264, 1816, 15264, 32488, 11668, 25941, 21533, 2556, 25589, 28887, 10563, 19038, 30995, 3009, 27638, 25310, 13561, 24146, 16602, 31118, 24036, 27877, 22753, 16219, 24877, 20255, 8045, 9192, 1021, 7633, 2907, 10033, 5493, 6334, 2580, 31843, 15117, 310, 6222, 27717, 25332, 30404, 16236, 29263, 30530, 22589, 7856, 8240, 21172, 27464, 12252, 17338, 20035, 6498, 8070, 7513, 23337, 25184, 10031, 6478, 2841, 17830, 17421, 19990, 28473, 1629, 15990, 15163, 32891, 4665, 18270, 26505, 13025, 16053, 5563, 28192, 27563, 11698, 21394, 32570, 19379, 20382, 27887, 10983, 5440, 27245, 33114, 3502, 18056, 19080, 14187, 2681, 20895, 34507, 17945, 1018, 31800, 25268, 30825, 6377, 3722, 14418, 9708, 11714, 11947, 18106, 34359, 8814, 33989, 21202, 3752, 31702, 16089, 2089, 6658, 4468, 11072, 29116, 20087, 22802, 33587, 7479, 5034, 4431, 25297, 9796, 24039, 3746, 22609, 108, 28705, 29522, 31956, 12563, 1183, 26103, 17580, 9576, 7123, 16276, 14466, 13711, 15739, 26167, 6080, 5867, 5776, 25658, 11360, 28635, 32665, 30683, 26571, 13428, 3401, 4295, 16874, 24223, 14750, 16392, 34508, 31278, 20110, 12189, 4970, 4972, 24112, 2283, 20723, 416, 7487, 6186, 21014, 17893, 6613, 11094, 13644, 5907, 32461, 33870, 11323, 18873, 13613, 4596, 7377, 14576, 2075, 32676, 11221, 31690, 3469, 12987, 18441, 16488, 32100, 18439, 28333, 15552, 15638, 16187, 14740, 31260, 1722, 23056, 12639, 5989, 30042, 5100, 25663, 20926, 1963, 14743, 16949, 5346, 7312, 34080, 5900, 17121, 404, 7115, 4412, 29233, 1586, 10560, 24265, 18271, 33547, 14704, 32028, 3270, 28061, 27917, 29080, 17539, 26420, 26435, 909, 31525, 12032, 17138, 15558, 19703, 23686, 27460, 27818, 10574, 15263, 12060, 15215, 10808, 33893, 17715, 25600, 34413, 21325, 4393, 29053, 32566, 5871, 5310, 1964, 2746, 2670, 28528, 7852, 2452, 29888, 28004, 25831, 32990, 12449, 18490, 13660, 23045, 26212, 7966, 25651, 422, 29491, 3246, 28155, 27306, 24532, 20355, 1182, 15841, 23125, 1959, 22986, 26301, 17482, 34459, 20685, 27689, 15828, 11213, 29645, 2069, 32154, 8791, 8689, 20527, 19468, 19949, 6035, 6520, 8187, 7512, 30944, 14762, 922, 30753, 13611, 24493, 20454, 22878, 21858, 19033, 30649, 31779, 18041, 33853, 6344, 30664, 11613, 1630, 1411, 981, 9292, 18191, 17694, 3960, 2258, 19070, 7341, 17546, 30186, 956, 8263, 2624, 31650, 11382, 31742, 2142, 33308, 14788, 14647, 32239, 6187, 14832, 268, 6375, 17436, 20451, 30868, 28644, 14620, 14977, 3413, 10425, 7869, 13804, 14508, 27419, 14237, 1611, 32617, 15229, 1609, 18843, 23840, 33854, 8129, 883, 20967, 11760, 11462, 9960, 4501, 28028, 3571, 10376, 12174, 6336, 11356, 20812, 6419, 25867, 16632, 18450, 16921, 25755, 16491, 5255, 14437, 7530, 303, 21689, 10052, 15955, 21262, 11997, 11458, 9236, 1939, 12102, 29020, 8630, 18935, 2596, 21561, 10733, 21666, 15204, 18600, 11500, 19183, 5409, 19141, 27641, 27866, 33859, 9803, 1645, 28809, 24629, 3087, 28423, 15217, 26246, 33792, 28915, 14021, 3324, 4375, 2726, 20424, 6247, 25092, 2004, 26161, 25120, 33488, 19611, 21977, 22112, 11283, 30368, 7093, 8507, 32490, 11505, 960, 5813, 28904, 8530, 14348, 265, 27175, 19010, 19783, 82, 16699, 7786, 24128, 16798, 13299, 2450, 12443, 4477, 10020, 1093, 16066, 18672, 1874, 6675, 17596, 33553, 16003, 14100, 8137, 28730, 31070, 9589, 10912, 23262, 29710, 25412, 13453, 31325, 28044, 14976, 20818, 23913, 7488, 32775, 32911, 13403, 33377, 25812, 30285, 30927, 22068, 17467, 9458, 13164, 7285, 31653, 1603, 31458, 20919, 17764, 25829, 22738, 19127, 11438, 3247, 24729, 16753, 5108, 10090, 20819, 1879, 16728, 10668, 244, 22319, 11353, 2012, 31381, 12874, 363, 32720, 32979, 20628, 10863, 5300, 6130, 33699, 18114, 31556, 15809, 19195, 23647, 6566, 8901, 3402, 22851, 9990, 9245, 22770, 17096, 13577, 22855, 22620, 21543, 7960, 9994, 30737, 5153, 28655, 34071, 26879, 23499, 19239, 1782, 7373, 8560, 14362, 9862, 8195, 3847, 28407, 8549, 30673, 25942, 17372, 4941, 21484, 21856, 29399, 2569, 20978, 30267, 32953, 32162, 15129, 12829, 22790, 34464, 12111, 9005, 13021, 8126, 12580, 8158, 19155, 6961, 774, 19494, 18859, 13181, 33667, 5380, 25061, 28016, 12422, 16131, 8457, 13031, 23656, 19629, 1761, 10275, 21243, 1946, 25974, 17377, 18304, 20471, 22120, 32957, 6037, 16992, 11102, 11017, 26307, 1601, 30933, 14746, 29967, 32350, 13128, 22202, 18206, 15926, 24539, 33188, 8312, 28199, 10766, 32987, 30801, 6798, 28396, 9985, 15419, 22784, 2704, 13113, 31934, 10510, 22638, 25258, 15914, 8841, 14733, 31240, 3204, 26395, 25350, 10362, 8136, 30985, 18127, 19128, 19722, 4221, 24434, 28371, 34394, 4319, 21210, 17802, 30877, 31993, 34369, 34438, 9856, 17731, 15322, 12251, 12448, 10298, 11437, 26874, 17979, 29446, 20719, 28040, 2574, 17042, 2896, 18363, 27077, 1973, 10823, 33832, 25605, 17950, 32056, 27784, 23515, 31680, 11428, 21402, 33937, 20197, 28067, 15817, 27890, 18232, 2472, 12835, 599, 3771, 8601, 23501, 9964, 1123, 34362, 7748, 23381, 14496, 16152, 11931, 4175, 27547, 6423, 21006, 23500, 13610, 27078, 23415, 21633, 21220, 4138, 7127, 30436, 24028, 25893, 21732, 9563, 17557, 12857, 22898, 21494, 31082, 24641, 16453, 32447, 21931, 31327, 9147, 32187, 1420, 4524, 11550, 7223, 18478, 22127, 31752, 28461, 13641, 16812, 19533, 19813, 19753, 28545, 11832, 10608, 26111, 3646, 11715, 11544, 33625, 29820, 25677, 5532, 6943, 1635, 29708, 21357, 5618, 5101, 5946, 7988, 33487, 10671, 28402, 6028, 14246, 7004, 3855, 14858, 20499, 25119, 28880, 12780, 27196, 5480, 8700, 21567, 9158, 13683, 2271, 21992, 444, 27369, 21644, 22529, 598, 31370, 18261, 17206, 15061, 5936, 22462, 9213, 20857, 21426, 11968, 8388, 25733, 16199, 14688, 32579, 31242, 9972, 3938, 6312, 11621, 21347, 26371, 13413, 30408, 31175, 31004, 14698, 312, 11985, 1218, 2622, 17364, 25375, 27586, 22036, 5698, 21233, 569, 18144, 29424, 2930, 33590, 33659, 22092, 29245, 4422, 18872, 13659, 31760, 28552, 12892, 33190, 25072, 22240, 19154, 24426, 15276, 9397, 7471, 4129, 30348, 33763, 32863, 30277, 9753, 583, 26731, 15306, 16571, 13229, 4041, 16963, 22357, 29376, 26191, 4362, 8250, 958, 25452, 34033, 21727, 31669, 29181, 25841, 16487, 9302, 18477, 1993, 3468, 3647, 34340, 29336, 17933, 34258, 14714, 25386, 6034, 13485, 2158, 11442, 34057, 34378, 15460, 3973, 34545, 24826, 27537, 33415, 30204, 27587, 15373, 846, 24293, 33491, 11058, 25265, 9500, 713, 33663, 28117, 1066, 32344, 8037, 4084, 9316, 27431, 13441, 33705, 13898, 3653, 32212, 24417, 16906, 15961, 6351, 15081, 17603, 27291, 11585, 29761, 25851, 30797, 14674, 33597, 19898, 21495, 19323, 8252, 5313, 19988, 24821, 8979, 10319, 14501, 31465, 22083, 15058, 15267, 9321, 20743, 17359, 9297, 27985, 7065, 26418, 27258, 29627, 30588, 15124, 6153, 6990, 16065, 11289, 15943, 15746, 26350, 21835, 32413, 29140, 23294, 23936, 17304, 427, 26428, 27342, 1132, 16252, 1843, 14639, 11133, 3839, 24166, 3330, 14113, 9258, 1386, 17842, 30562, 24430, 15340, 4212, 6308, 336, 12002, 4645, 2519, 13297, 27327, 5309, 7807, 30583, 16144, 4896, 21194, 19330, 32299, 22528, 33049, 5030, 666, 19541, 28858, 32186, 22654, 34288, 31797, 16209, 5000, 10799, 4753, 12341, 4336, 19895, 4213, 7153, 33443, 26546, 16741, 9824, 17882, 17913, 24953, 28649, 7754, 33319, 32715, 25093, 27521, 10191, 31711, 561, 33327, 15059, 33538, 27938, 7203, 25741, 25936, 2366, 24998, 34218, 25680, 3760, 7195, 11560, 25695, 26791, 19105, 11400, 27585, 23490, 9642, 14690, 16585, 9568, 1850, 8350, 34446, 21366, 12779, 20828, 25977, 18572, 31217, 12821, 12380, 9646, 28307, 12723, 7270, 7301, 8698, 23734, 17413, 4120, 24085, 30890, 2658, 23861, 7775, 4015, 2345, 12397, 1862, 23939, 28322, 7398, 7647, 29911, 23486, 20463, 14699, 11634, 13348, 16983, 2578, 11650, 12361, 26559, 19598, 24411, 13395, 2082, 6490, 18966, 9546, 11130, 25709, 33064, 25855, 11648, 1240, 19365, 7001, 28700, 24009, 11906, 10948, 25013, 34409, 2822, 4618, 22404, 5487, 33399, 31858, 34277, 8373, 32821, 28502, 6775, 11233, 11036, 8859, 8988, 23905, 29111, 21124, 3191, 13783, 4954, 5559, 2225, 21021, 18779, 4197, 8422, 14849, 32844, 8022, 88, 20985, 16155, 23181, 22659, 4660, 10637, 31288, 17794, 6505, 1404, 13006, 28819, 3264, 2602, 26633, 26544, 12116, 12727, 28489, 31046, 13862, 7969, 12508, 18918, 12044, 10189, 14644, 14987, 26993, 11370, 24780, 14729, 24542, 7573, 11195, 24187, 12107, 33572, 1120, 30515, 31991, 22877, 26124, 13383, 34293, 10072, 22410, 4428, 2550, 2407, 27272, 19303, 17321, 9634, 8234, 13630, 9530, 30044, 23652, 4624, 27826, 27540, 28901, 30465, 1427, 3102, 33004, 18430, 17047, 2863, 17122, 19521, 23871, 2207, 21013, 25636, 5724, 33789, 1633, 21617, 6286, 2107, 27451, 33534, 34111, 31084, 8637, 2164, 16623, 20252, 14194, 22228, 33334, 10692, 3642, 17462, 5748, 29758, 8868, 5385, 12471, 14708, 24380, 2338, 26071, 10251, 23974, 19733, 33380, 22581, 32850, 16141, 21060, 17343, 2770, 19732, 11139, 11199, 8274, 17991, 34201, 30386, 19594, 14265, 30786, 22119, 11343, 22373, 4443, 10544, 27026, 9907, 13072, 21696, 16907, 34542, 4394, 28882, 33743, 26660, 15477, 30808, 8495, 9647, 21861, 25921, 13874, 4704, 34015, 5078, 28865, 22408, 24705, 6359, 23531, 2409, 4009, 23303, 29348, 3426, 34162, 33813, 2800, 33666, 34483, 354, 28939, 18942, 1523, 10266, 13649, 4368, 13726, 9913, 16039, 1391, 23601, 3984, 18565, 15962, 15149, 6971, 11719, 30029, 23564, 22438, 21144, 29612, 24303, 15712, 15236, 458, 33296, 32936, 14664, 22371, 4022, 8564, 26289, 32939, 4471, 24064, 30316, 21385, 25005, 8610, 8633, 16259, 30722, 32591, 7007, 22973, 3636, 748, 29274, 793, 16338, 7930, 26293, 16357, 30892, 15046, 34198, 22740, 31645, 12605, 11384, 24669, 25639, 3643, 30658, 7135, 17273, 10440, 27606, 5444, 28361, 25230, 32325, 259, 19491, 12547, 30477, 7701, 597, 9641, 4679, 15290, 16480, 587, 30733, 34222, 971, 29186, 24473, 28972, 5396, 31399, 4839, 32819, 23668, 19409, 8609, 32541, 24365, 8093, 26555, 3806, 30181, 33272, 20551, 2674, 14239, 7853, 31723, 15331, 33266, 5502, 2799, 33440, 9851, 25012, 648, 20948, 584, 22508, 31102, 2235, 26393, 3467, 5257, 28580, 31764, 29226, 25186, 28116, 25152, 15522, 7355, 2218, 7422, 5085, 21420, 9737, 27749, 28422, 12196, 22395, 14356, 18652, 17800, 6197, 24424, 25593, 4081, 16547, 753, 3332, 32787, 1333, 10262, 30414, 5368, 28792, 1678, 32362, 20330, 16750, 5758, 18499, 13585, 5163, 8075, 6653, 14151, 4589, 1658, 4260, 6660, 2625, 34411, 27500, 25562, 28821, 13827, 26522, 10757, 5919, 10071, 12086, 26570, 11524, 17290, 2960, 12075, 34412, 22708, 726, 5249, 18320, 10039, 10880, 12133, 23968, 17608, 4472, 32052, 30635, 31775, 30123, 19642, 26045, 30071, 19073, 20017, 34445, 22763, 7321, 9015, 34045, 3660, 11757, 4553, 17103, 440, 17517, 28678, 157, 14343, 8314, 8243, 28349, 9929, 9619, 23324, 33110, 32732, 6764, 16008, 7761, 32695, 6973, 16477, 7472, 5808, 24552, 4634, 7160, 1179, 10589, 1342, 19094, 22209, 27553, 26668, 1897, 20337, 13419, 32515, 28742, 8552, 5702, 7102, 4637, 18850, 15455, 21042, 11401, 21805, 5360, 29683, 21893, 15861, 328, 7068, 5585, 26102, 724, 26220, 5184, 27340, 16938, 30638, 12393, 5183, 25628, 4527, 25293, 29628, 16083, 31859, 30900, 9379, 29313, 1215, 9187, 15969, 24656, 21161, 20315, 30484, 22102, 33738, 10418, 30983, 30711, 21865, 32811, 14168, 33158, 5017, 34051, 1498, 25291, 14759, 4923, 13740, 3655, 22565, 22437, 20182, 20415, 1492, 13464, 16213, 34402, 25402, 15916, 22016, 20104, 4134, 23058, 18589, 11924, 3197, 13412, 31497, 14970, 9027, 15077, 12108, 21532, 7384, 29350, 21127, 32501, 16603, 24081, 468, 21392, 1998, 14017, 15376, 23570, 20874, 32332, 33544, 20450, 27988, 34049, 10800, 22292, 4406, 10686, 1109, 5798, 1041, 23108, 15102, 33568, 5582, 22573, 34186, 4697, 8858, 26453, 26947, 10911, 26517, 24747, 32712, 1784, 3892, 8711, 14963, 6643, 13159, 31762, 29667, 2092, 6482, 29752, 28496, 2153, 29944, 10652, 29982, 3896, 5625, 31573, 3442, 19225, 17442]

    print("number of batches: " + str(len(init_plan)) + ", calculating cost......")
    cost, _, _ = New_Simulate_Cost(init_plan, planner.cached_rows, batches_id, batches_freq)
    print("Cost: " + str(cost) + ", ")
    '''

    # plan, cost, _, _ = New_Heuristic_Search(
    #     planner.log_path, 
    #     planner.cached_rows, 
    #     batches_id, 
    #     batches_freq, 
    #     warm_up_steps=planner.warm_up_steps, 
    #     init_plan=init_plan,
    #     search_limit=35000, 
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
    init_plan = None
    
    plan, cost = Segmented_LFU_Multiprocess_Search(
        planner.log_path, 
        planner.cached_rows, 
        batches_id,
        batches_freq,
        planner.warm_up_steps, 
        init_plan, 
        None,
        None,
        None, 
        search_limit=40,
        hotness_diff_threshold_base_relax_ratio=0.8,
        hotness_diff_threshold_update_window=10,
        hotness_diff_threshold_startup_cap=8,
        hotness_diff_threshold_increment_relax_ratio=0.001,
        hotness_diff_threshold_late_time_cap=0.3,
        hotness_diff_threshold_relax_ratio_penalty_rate=0.8,
        num_process=1,
        )

    '''------------------------ Save the generated plan ------------------------'''
    planner.plan = plan[:]
    if PLAN_FILE_NAME is not None:
        planner.to_parquet(PLAN_FILE_NAME)
    print("Cost: " + str(cost) + ".")    

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
    # print("Start converting training plan to batch info...")
    # input_path = os.path.join(AVAZU_PLAN_PATH, "training_plan-best.parquet")
    # output_path = os.path.join(AVAZU_PLAN_PATH, "id_to_prefetch.parquet")
    # Training_Plan_to_ID_of_Batches(input_path, output_path, batches_id, batches_freq)

    '''------------------------------- End ------------------------------'''

    planning_time = time.time() - dataloading_time - start_time
    print("dataloading_time: " + str(dataloading_time) + ", planning_time: " + str(planning_time))


    
