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
DATASET = "taobao"
BATCH_FILE_SUFFIX = "-1024"
LOG_PATH = "/root/files/coding/data_loading_planner/taobao_run_1"
PLAN_FILE_NAME = "-LFU-15-1024-4.1"
DLRM_GPU_CACHE_SIZE = int(33762577 * 0.05)                                                       # num of rows * cache_ratio                     # num of embedding entries = GB / float32 / 64 dimension
TBSM_GPU_CACHE_SIZE = int(5159457 * 0.15)                                                        # num of rows * cache_ratio
DLRM_DATA_PATH = "/root/files/coding/RecSys-Training-Planner/DLRM/input/kaggle/kaggleAdDisplayChallenge_processed.npz"
TBSM_DATA_PATH = "/root/files/coding/RecSys-Training-Planner/TBSM/output/taobao_train_t20.npz"
DLRM_PLAN_PATH = "/root/files/coding/RecSys-Training-Planner/DLRM/input/training_plan/"
TBSM_PLAN_PATH = "/root/files/coding/RecSys-Training-Planner/TBSM/input/training_plan/"

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
        if dataset == "criteo":
            self.load_criteo_data(data_path, batches)
        else:
            self.load_taobao_data(data_path, batches)
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
        
    def load_criteo_data(self, data_path, batches):
        """
        Returned data:
            self.batches (list of df.Series): Categorical objects contained in each batch. (This is basically all we need from the dataset.) 
        """
        # Read cat_data
        data = np.load(os.path.abspath(data_path))
        X_cat = data["X_cat"]                                                                                               # categorical feature
        # cat_data = df.DataFrame(X_cat).astype(int)                                                                          # each column is a row of categorical features in the original dataset
        
        # Count size of features.
        self.cat_counts = list(data["counts"])

        '''
        # Convert id from id_in_cat to unique_id_in_dataset
        cat_offsets = list()
        tmp_offset = 0
        for i in range(len(self.cat_counts)):
            cat_offsets.append(tmp_offset)
            tmp_offset = tmp_offset + self.cat_counts[i]
        cat_offsets = df.Series(cat_offsets)
        cat_data = cat_data + cat_offsets
        
        # Convert batches of indices to batches of ids
        self.batches = list()
        self.freq_batches = list()
        for i in range(0, self.batch_num - 1):
            # batch_indices = batches[i]
            # self.batches.append(cat_data.iloc[batch_indices].stack().unique())
            indices = batches[i].to_numpy()
            counted_batch = df.concat([df.Series(rotated_cat_data[indices].flatten())]).value_counts()
            indices_batch = df.Series(counted_batch.index)
            freq_batch = counted_batch.reset_index(drop=True)
            self.batches.append(indices_batch)
            self.freq_batches.append(freq_batch)
        # For batches[self.batch_num - 1], deal with -1s
        batch_indices = df.Series([i for i in batches[self.batch_num - 1].values.tolist() if i != -1])
        self.batches.append(cat_data.iloc[batch_indices].stack().unique())
        '''

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
        

    def load_taobao_data(self, data_path, batches):
        """
        Returned data:
            self.batches (list of df.Series): Categorical objects contained in each batch. (This is basically all we need from the dataset.) 
        """
        # Read cat_data
        data = np.load(os.path.abspath(data_path))
        X_cat = data["X_cat"]                                                                                               # categorical feature

        # Count size of features.
        self.cat_counts = [987994, 4162024, 9439]

        # Convert id from id_in_cat to unique_id_in_dataset
        cat_offsets = list()
        tmp_offset = 0
        for i in range(len(self.cat_counts)):
            cat_offsets.append(tmp_offset)
            tmp_offset = tmp_offset + self.cat_counts[i]
        for i in range(len(cat_offsets)):
            X_cat[i] = X_cat[i] + cat_offsets[i]
        
        # Calculate objs in batch
        rotated_cat_data = np.swapaxes(X_cat, 0, 1)
        self.batches = list()
        self.freq_batches = list()
        for i in range(batches.shape[1] - 1):
            indices = batches[i].to_numpy()
            counted_batch = df.concat([df.Series(rotated_cat_data[indices].flatten())]).value_counts()
            indices_batch = df.Series(counted_batch.index)
            freq_batch = counted_batch.reset_index(drop=True)
            self.batches.append(indices_batch)
            self.freq_batches.append(freq_batch)
        # last batch contains -1, should remove them first.
        indices = np.array([i for i in batches[self.batch_num - 1].values.tolist() if i != -1])
        counted_batch = df.concat([df.Series(rotated_cat_data[indices].flatten())]).value_counts()
        indices_batch = df.Series(counted_batch.index, dtype='int32')
        freq_batch = counted_batch.reset_index(drop=True)
        self.batches.append(indices_batch)
        self.freq_batches.append(freq_batch)

    def Simulate_Cost(self, plan: list, init_cache_state: Optional[df.Series], start_step: Optional[int], start_cost: Optional[int]):
        '''
        Calculate cost of a given route.
        '''

        num_steps = len(plan)
        if isinstance(init_cache_state, df.Series):
            gpu_cache = init_cache_state
            cost_total = start_cost
            num_available_rows = self.cached_rows - len(gpu_cache)
        else:
            gpu_cache = df.Series([-1] * self.cached_rows, dtype=int)
            cost_total = 0
            start_step = 0
            num_available_rows = self.cached_rows
        
        # batch_register = df.Series([-1] * self.cached_rows, dtype='int32')
        freq_register = df.Series([-1] * self.cached_rows, dtype='int32')
        
        # the main loop
        for step in range(start_step, num_steps):
            # Get ids in current batch
            batch_ids = self.batches[plan[step]]
            freq_batch_ids = self.freq_batches[plan[step]]

            # Find ids that need to be transfered to cache
            ids_to_comm = batch_ids[batch_ids.isin(gpu_cache) == False]
            freq_ids_to_comm = freq_batch_ids[ids_to_comm.index]
            num_ids_to_comm = len(ids_to_comm)                                                                              # Cost 1: cost of moving data in
            num_rows_to_evic = num_ids_to_comm - num_available_rows
            num_rows_to_evic = num_rows_to_evic if num_rows_to_evic > 0 else 0                                              # Cost 2: cost of moving data out

            # Evic first num_rows_to_evic rows of ids, since the gpu_cache is sorted by batch flags.
            if num_rows_to_evic > 0:
                evictable_row = gpu_cache[gpu_cache.isin(batch_ids) == False]
                nonempty_row = gpu_cache[evictable_row.index][gpu_cache[evictable_row.index] != -1]
                idx_freq_evictable_row = freq_register[nonempty_row.index].nsmallest(num_rows_to_evic, keep='last').index       # very expensive
                # idx_freq_evictable_row = evictable_row.index[:num_rows_to_evic]                                                 # a cheap equivalent, TODO: implement a O(nlogn) algorithm later
                if idx_freq_evictable_row.shape[0] < num_rows_to_evic:                                                                      # Temporarily fix. should replace this and the expensive line of code with a O(nlogn). Do this after the paper being submitted.
                    idx_freq_evictable_row = nonempty_row.index[:num_rows_to_evic]
                gpu_cache[idx_freq_evictable_row] = -1
                freq_register[idx_freq_evictable_row] = 0
                # gpu_cache = gpu_cache.iloc[num_rows_to_evic : -1]
            
            # Update gpu cache
            # gpu_cache = df.concat([gpu_cache, ids_to_comm], ignore_index=True)
            idx_update = gpu_cache[gpu_cache == -1].index[:num_ids_to_comm]
            gpu_cache[idx_update] = ids_to_comm.values
            freq_register[idx_update] = freq_register[idx_update].values + freq_ids_to_comm.values
            
            num_available_rows = num_available_rows + num_rows_to_evic - num_ids_to_comm
            cost_total = cost_total + num_ids_to_comm + num_rows_to_evic

        return cost_total, gpu_cache, freq_register
    
    def Heuristic_Search(self, init_plan: list = None, init_cache_state: df.Series = None, init_cost: int = None, process_id: int = None, init_freq_status: df.Series = None) -> tuple:
        if process_id is not None:
            log_file_path = os.path.join(self.log_path, ("search-log-" + str(process_id) + ".txt"))
        else:
            log_file_path = os.path.join(self.log_path, ("search-log.txt"))
        self.search_log = open(log_file_path, "w")
        # self.search_log_l2 = open(os.path.join(self.log_path, "search-log-level-2.txt"), "w")

        if isinstance(init_plan, list):
            plan = init_plan
        else:
            plan = list()
        
        unused_batch = [i for i in range(self.batch_num) if i not in plan]
        random.shuffle(unused_batch)

        # The warm up phase (randomly fill in some steps since the first few steps don't really matter that much.)
        time_warmup_start = time.time()
        if len(plan) < self.warm_up_steps:
            fill_in_length = self.warm_up_steps - len(plan)
            plan = plan + unused_batch[ : fill_in_length]
            del unused_batch[ : fill_in_length]
        if init_cache_state is not None and init_cost is not None and init_freq_status is not None:
            cost_total = init_cost
            cache_state = init_cache_state
            freq_state = init_freq_status
        else:
            cost_total, cache_state, freq_state = self.Simulate_Cost(plan, None, None, None)
        
        time_warmup_finished = time.time()

        # Set up the hotness difference threshold.
        hotness_diff_threshold = self.hotness_diff_threshold_init_value
        hotness_diff_history = [hotness_diff_threshold] * self.hotness_diff_threshold_update_period
        hotness_diff_history_idx = 0
        hotness_diff_threshold_dynamic_ratio = self.hotness_diff_threshold_base_relax_ratio
        hotness_diff_threshold_ratio_increment = 0

        output_str = "[Warm up phase] cost: " + str(cost_total) + ", cache_usage: " + str(len(cache_state) / self.cached_rows) + " warmup time: " + str(time_warmup_finished - time_warmup_start) + "\n[Startup plan] " + str(plan) + "\n[Start searching] hotness_diff threshold = " + str(hotness_diff_threshold)
        # print(output_str)
        self.search_log.write(output_str + "\n")

        # Searching
        time_last_step = time.time()
        for step in range(len(plan), self.batch_num):
            cost_best_choice = LARGE_NUMBER
            # cache_state_best_choice = None
            best_choice = 0
            cost_worst_choice = 0
            if (cache_state == -1).nunique() == 2:
                num_available_rows = int(cache_state.value_counts()[-1]) # self.cached_rows - len(cache_state)
            else:
                if (cache_state == -1)[0]:
                    num_available_rows = self.cached_rows
                else:
                    num_available_rows = 0

            # Recalibration
            if step % self.hotness_diff_threshold_recal_steps == 0:
                hotness_diff_threshold = self.hotness_diff_threshold_init_value
                hotness_diff_history = [hotness_diff_threshold] * self.hotness_diff_threshold_update_period

            # Search at most self.search_limit steps to find a choice
            for choice_idx in range(len(unused_batch)):
                # Suffix _tmp menas current step

                # Actually make the plan and calculate cost [hight cost, don't use this]
                # trial_plan = plan + [unused_batch[choice_idx]]
                # cost_total_tmp, cache_state_tmp = self.Simulate_Cost(trial_plan, cache_state.copy(), step, cost_total)
                # cost_tmp = cost_total_tmp - cost_total
                
                # Just calculate cost, don't update cache.
                batch_ids = self.batches[unused_batch[choice_idx]]
                ids_to_comm = batch_ids[batch_ids.isin(cache_state) == False]
                num_ids_to_comm = len(ids_to_comm)                                                                              # Cost 1: cost of moving data in
                num_rows_to_evic = num_ids_to_comm - num_available_rows
                num_rows_to_evic = num_rows_to_evic if num_rows_to_evic > 0 else 0                                              # Cost 2: cost of moving data out
                cost_tmp = num_ids_to_comm + num_rows_to_evic
                
                # Record a better/worse choice
                if cost_tmp < cost_best_choice:
                    cost_best_choice = cost_tmp
                    # cache_state_best_choice = cache_state_tmp
                    best_choice = choice_idx
                if cost_tmp > cost_worst_choice:
                    cost_worst_choice = cost_tmp
                
                # Early stop conditions
                hotness_diff_tmp = cost_worst_choice - cost_best_choice
                if choice_idx > self.hotness_diff_threshold_startup_cap and hotness_diff_tmp > hotness_diff_threshold:
                    hotness_diff_threshold_ratio_increment = hotness_diff_threshold_ratio_increment + self.hotness_diff_threshold_increment_relax_ratio
                    hotness_diff_threshold_dynamic_ratio = self.hotness_diff_threshold_base_relax_ratio + hotness_diff_threshold_ratio_increment
                    break

                # Late stop conditions
                if choice_idx > self.search_limit or choice_idx == (len(unused_batch) - 1):
                    break

                # Level 2 logging
                # output_str = "[Choice " + str(choice_idx) +" in Step " + str(step) + "] choice: " + str(unused_batch[choice_idx]) + ", cost: " + str(cost_tmp) + ", best_cost: " + str(cost_best_choice) + ", worst_cost: " + str(cost_worst_choice) + ", hotness_diff: " + str(hotness_diff_tmp)
                # self.search_log_l2.write(output_str + "\n")
            
            # Decide current step and actually update cache state.
            choice = unused_batch.pop(best_choice)
            plan.append(choice)
            # cache_state = cache_state_best_choice
            cost_total = cost_total + cost_best_choice
            random.shuffle(unused_batch)

            # Update cache
            batch_ids = self.batches[choice]
            freq_batch_ids = self.freq_batches[choice]
            ids_to_comm = batch_ids[batch_ids.isin(cache_state) == False]
            num_ids_to_comm = len(ids_to_comm)
            if num_ids_to_comm > self.cached_rows:
                print("Even batch with the smallest cost exceeds the capacity of the cache.")
            freq_ids_to_comm = freq_batch_ids[ids_to_comm.index]
            num_rows_to_evic = num_ids_to_comm - num_available_rows
            if num_rows_to_evic > 0:
                # cache_state = cache_state.iloc[num_rows_to_evic : -1]
                evictable_row = cache_state[cache_state.isin(batch_ids) == False]
                nonempty_row = cache_state[evictable_row.index][cache_state[evictable_row.index] != -1]
                # idx_freq_evictable_row = freq_state[evictable_row.index].nsmallest(len(evictable_row) - num_rows_to_evic, keep='last', ).index                      # Very expensive!!!
                idx_freq_evictable_row = freq_state[nonempty_row.index].nsmallest(num_rows_to_evic, keep='last', ).index                      # Very expensive!!!
                # idx_freq_evictable_row = evictable_row.index[:num_rows_to_evic]                                                                 # a cheap equivalent
                if idx_freq_evictable_row.shape[0] < num_rows_to_evic:                                                                      # Temporarily fix. should replace this and the expensive line of code with a O(nlogn). Do this after the paper being submitted.
                    idx_freq_evictable_row = nonempty_row.index[:num_rows_to_evic]
                cache_state[idx_freq_evictable_row] = -1
                freq_state[idx_freq_evictable_row] = 0
            # cache_state = df.concat([cache_state, ids_to_comm], ignore_index=True)
            idx_update = cache_state[cache_state == -1].index[:num_ids_to_comm]
            cache_state[idx_update] = ids_to_comm.values
            freq_state[idx_update] = freq_state[idx_update].values + freq_ids_to_comm.values

            # Update threshold
            hotness_diff_history[hotness_diff_history_idx] = cost_worst_choice - cost_best_choice
            hotness_diff_history_idx = (hotness_diff_history_idx + 1) % self.hotness_diff_threshold_update_period
            hotness_diff_mean = statistics.mean(hotness_diff_history)
            hotness_diff_threshold = hotness_diff_mean * hotness_diff_threshold_dynamic_ratio

            # Any step takes more than 0.3s should be considered as slightly late, thus no increment of ralax ratio
            step_time = time.time() - time_last_step
            time_last_step = time.time()
            if step_time > self.hotness_diff_threshold_late_time_cap:
                hotness_diff_threshold_ratio_increment = hotness_diff_threshold_ratio_increment * self.hotness_diff_threshold_relax_ratio_penalty_rate
                hotness_diff_threshold_dynamic_ratio = self.hotness_diff_threshold_base_relax_ratio + hotness_diff_threshold_ratio_increment

            # Logging
            
            output_str = "[Step " + str(step) + "] choice: " + str(choice) + ", cost: " + str(cost_best_choice) + ", hotness_diff: " + str(cost_worst_choice - cost_best_choice) + ", cache_usage(last run): " + str(1 - (num_available_rows/ self.cached_rows)) + ", step_time = " + str(step_time) + ", searched choices: " + str(choice_idx) + "\n             hotness_diff_history: " + str(hotness_diff_history) + "\n             mean hotness_diff: " + str(hotness_diff_mean) + ", ratio_increment: " + str(hotness_diff_threshold_ratio_increment) + ", dynamic threshold ratio: " + str(hotness_diff_threshold_dynamic_ratio) + ", new threshold: " + str(hotness_diff_threshold)
            # print(output_str)
            self.search_log.write(output_str + "\n")
        
        # output_str = "[cost: " + str(cost_total) + "] Training plan generated: " + str(plan)
        # print(output_str)
        # self.search_log.write(output_str + "\n")

        self.search_log.close()
        # self.search_log_l2.close()
        self.plan = plan
        return cost_total, cache_state, freq_state

    def Grouped_Search(self, num_groups: int, init_plan: list = None) -> int:
        time_stamp_1 = time.time()
        print("Constructing grouped batches...")

        # Replace the batches by grouped batches
        backup_batches = self.batches[:]
        backup_num_batch = self.batch_num
        self.batch_num = num_groups
        original_batches = self.batches[:]
        self.batches = list()
        indices_batch = list(range(len(original_batches)))

        if init_plan is not None:
            # First group contains the init_plan
            num_item_per_group = int((len(original_batches) - len(init_plan)) / (num_groups - 1))
            start_group = 1
            backup_warm_up_steps = self.warm_up_steps
            self.warm_up_steps = int((self.warm_up_steps - len(init_plan)) / num_item_per_group)
            if self.warm_up_steps < 0:
                self.warm_up_steps = 0
            new_init_plan = [0]
            
            tmp = df.Series(dtype=original_batches[0].dtype)
            for idx in init_plan:
                tmp = df.concat([tmp, original_batches[idx]], ignore_index=True)
            self.batches.append(tmp.unique())
            tmp_init_plan = sorted(init_plan)
            for i in range(len(init_plan)):
                tmp = original_batches.pop(tmp_init_plan[len(init_plan) - 1 - i])
                indices_batch.pop(tmp_init_plan[len(init_plan) - 1 - i])
                del tmp
            
        else:
            num_item_per_group = int(len(original_batches) / num_groups)
            start_group = 0
            backup_warm_up_steps = self.warm_up_steps
            self.warm_up_steps = int(self.warm_up_steps / num_item_per_group)
            new_init_plan = None
            init_plan = list()
        
        # Grouping
        for i in range(start_group, num_groups - 1):
            tmp = df.Series(dtype=original_batches[0].dtype)
            for _ in range(num_item_per_group):
                fst_batch = original_batches.pop(0)
                tmp = df.concat([tmp, fst_batch], ignore_index=True)
            tmp = tmp.unique()
            self.batches.append(tmp)
        
        # Dealing with the last group
        tmp = df.Series(dtype=original_batches[0].dtype)
        for _ in range(len(original_batches)):
            fst_batch = original_batches.pop(0)
            tmp = df.concat([tmp, fst_batch], ignore_index=True)
        tmp = tmp.unique()
        self.batches.append(tmp)

        time_stamp_2 = time.time()
        print("Construction complete! (" + str(time_stamp_2 - time_stamp_1) + "s)\nStart actual planning...")

        # Calling search function
        cost, _, _ = self.Heuristic_Search(new_init_plan)
        # cost = -1
        # self.plan = list(range(len(self.batches)))
        
        time_stamp_3 = time.time()
        print("Planning complete! (" + str(time_stamp_3 - time_stamp_2) + "s)\nRecovering environment...")

        # Recover the real plan
        grouped_plan = self.plan
        self.plan = init_plan
        for i in range(start_group, num_groups):
            if grouped_plan[i] != (num_groups - 1):
                self.plan = self.plan + indices_batch[(grouped_plan[i] - start_group) * num_item_per_group:((grouped_plan[i] - start_group) + 1) * num_item_per_group]
            else:
                self.plan = self.plan + indices_batch[(num_groups - 1 - start_group) * num_item_per_group:]

        # Recover parameters
        self.batches = backup_batches[:]
        self.warm_up_steps = backup_warm_up_steps
        self.batch_num = backup_num_batch

        time_stamp_4 = time.time()
        print("Recovery complete! (" + str(time_stamp_4 - time_stamp_3) + ")")

        return cost

    def Segmented_Search(self, chunk_size: int, init_plan: list = None) -> int:
        # Check chunk_size and make sure it is reasonable
        assert self.warm_up_steps < chunk_size
        assert self.batch_num > chunk_size

        # Save environment (protect the scene)
        backup_batches = self.batches[:]
        backup_num_batches = self.batch_num
        backup_warm_up_steps = self.warm_up_steps
        
        # Initialization. Do the first loop to handle num_warm_up_steps and init_plan.
        idx_unused_batches = list(range(self.batch_num))
        plan = list()
        
        chunk_count = 0
        start_time = time.time()

        self.batch_num = chunk_size
        self.plan = None
        self.batches = list()
        batch_indices = list()
        
        if init_plan is not None:
            # Move indices in init_plan to the beginning of the chunk
            sorted_init_plan = sorted(init_plan)
            new_init_plan = [i[0] for i in sorted(enumerate(init_plan), key=lambda x:x[1])]
            for i in range(len(init_plan)):
                tmp = idx_unused_batches[i]
                idx_unused_batches = sorted_init_plan[i]
                idx_unused_batches[sorted_init_plan[i]] = tmp
        else:
            new_init_plan = None
        
        for i in range(chunk_size):
            idx = idx_unused_batches.pop(0)
            batch_indices.append(idx)
            self.batches.append(backup_batches[idx])
        
        cost, cache_state, freq_state = self.Heuristic_Search(new_init_plan)

        for i in range(chunk_size):
            plan.append(batch_indices[self.plan[i]])

        end_time = time.time()
        print("[Chunk " + str(chunk_count) + "] Searched. Cost = " + str(cost) + ". (time: " + str(end_time - start_time) + "s)")
        chunk_count = chunk_count + 1
        
        self.warm_up_steps = 0

        # Segment unused batches (create new scene) and search
        while len(idx_unused_batches) > chunk_size:
            start_time = time.time()

            self.plan = None
            self.batches = list()
            batch_indices = list()

            for i in range(chunk_size):
                idx = idx_unused_batches.pop(0)
                batch_indices.append(idx)
                self.batches.append(backup_batches[idx])
            
            cost, cache_state, freq_state = self.Heuristic_Search(init_cache_state=cache_state, init_cost=cost, init_freq_status=freq_state)
            
            for i in range(chunk_size):
                plan.append(batch_indices[self.plan[i]])
            
            end_time = time.time()
            print("[Chunk " + str(chunk_count) + "] Searched. Cost = " + str(cost) + ". (time: " + str(end_time - start_time) + "s)")
            chunk_count = chunk_count + 1
        
        # Last chunk: len(unused_batches) <= chunk_size
        start_time = time.time()

        self.batch_num = len(idx_unused_batches)
        self.plan = None
        batch_indices = idx_unused_batches[:]
        self.batches = list()

        for i in range(len(idx_unused_batches)):
            self.batches.append(backup_batches[idx_unused_batches[i]])
        
        cost, cache_state, freq_state = self.Heuristic_Search(init_cache_state=cache_state, init_cost=cost, init_freq_status=freq_state)

        for i in range(len(idx_unused_batches)):
            plan.append(batch_indices[self.plan[i]])
        
        end_time = time.time()
        print("[Chunk " + str(chunk_count) + " (the last chunk)] Searched. Cost = " + str(cost) + ". (time: " + str(end_time - start_time) + "s)")
        
        # Restore environment
        self.batches = backup_batches[:]
        self.batch_num = backup_num_batches
        self.warm_up_steps = backup_warm_up_steps
        self.plan = plan

        return cost
                
    def Search_Wrapper(self, q: Queue, process_id: int = None, init_plan: list = None) -> None:
        cost, _, _ = self.Heuristic_Search(init_plan=init_plan, process_id=process_id)
        q.put(cost)
        q.put(self.plan)

    def Multiprocess_Search(self, chunk_size: int, init_plan: list = None) -> int:
        # Check chunk_size and make sure it is reasonable
        assert self.warm_up_steps < chunk_size
        assert self.batch_num > chunk_size

        # Save environment (protect the scene)
        backup_batches = self.batches[:]
        backup_freq_batches = self.freq_batches[:]
        backup_num_batches = self.batch_num
        backup_warm_up_steps = self.warm_up_steps
        
        # Initialization. Do the first loop to handle num_warm_up_steps and init_plan.
        ctx = get_context("spawn")
        
        idx_unused_batches = list(range(self.batch_num))
        plan = list()
        processes = list()
        queues = list()
        list_batch_indices = list()
        
        chunk_count = 0

        self.batch_num = chunk_size
        self.plan = None
        self.batches = list()
        self.freq_batches = list()
        batch_indices = list()
        
        if init_plan is not None:
            # Move indices in init_plan to the beginning of the chunk
            sorted_init_plan = sorted(init_plan)
            new_init_plan = [i[0] for i in sorted(enumerate(init_plan), key=lambda x:x[1])]
            for i in range(len(init_plan)):
                tmp = idx_unused_batches[i]
                idx_unused_batches = sorted_init_plan[i]
                idx_unused_batches[sorted_init_plan[i]] = tmp
        else:
            new_init_plan = None
        
        for i in range(chunk_size):
            idx = idx_unused_batches.pop(0)
            batch_indices.append(idx)
            self.batches.append(backup_batches[idx])
            self.freq_batches.append(backup_freq_batches[idx])
        
        # Issue the first searching process
        queues.append(ctx.Queue())
        list_batch_indices.append(batch_indices)
        processes.append(ctx.Process(target=self.Search_Wrapper, args=(queues[chunk_count], chunk_count, new_init_plan, )))
        processes[chunk_count].start()

        print("[Chunk " + str(chunk_count) + "] Issued.")
        chunk_count = chunk_count + 1
        
        self.warm_up_steps = 0

        # Segment unused batches (create new scene) and search
        while len(idx_unused_batches) > chunk_size:
            self.plan = None
            self.batches = list()
            batch_indices = list()
            self.freq_batches = list()

            for i in range(chunk_size):
                idx = idx_unused_batches.pop(0)
                batch_indices.append(idx)
                self.batches.append(backup_batches[idx])
                self.freq_batches.append(backup_freq_batches[idx])
            
            queues.append(ctx.Queue())
            list_batch_indices.append(batch_indices)
            processes.append(ctx.Process(target=self.Search_Wrapper, args=(queues[chunk_count], chunk_count, )))
            processes[chunk_count].start()

            print("[Chunk " + str(chunk_count) + "] Issued.")
            chunk_count = chunk_count + 1
        
        # Last chunk: len(unused_batches) <= chunk_size
        start_time = time.time()

        self.batch_num = len(idx_unused_batches)
        self.plan = None
        batch_indices = idx_unused_batches[:]
        self.batches = list()
        self.freq_batches = list()

        for i in range(len(idx_unused_batches)):
            self.batches.append(backup_batches[idx_unused_batches[i]])
            self.freq_batches.append(backup_freq_batches[idx_unused_batches[i]])
        
        last_cost, _, _= self.Heuristic_Search(process_id=chunk_count)
        
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
        for i in range(len(idx_unused_batches)):
            plan.append(batch_indices[self.plan[i]])
        
        print("[Chunk " + str(chunk_count) + "  (the last chunk)] Collected. Subcost: " + str(last_cost) + ", length of plan: " + str(len(self.plan)) + ".")
        
        # Restore environment
        self.batches = backup_batches[:]
        self.freq_batches = backup_freq_batches[:]
        self.batch_num = backup_num_batches
        self.warm_up_steps = backup_warm_up_steps
        self.plan = plan

        return cost

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
        ):
    plan, cost, _, _ = New_Heuristic_Search(
        log_path=log_path, 
        cached_rows=cached_rows, 
        batches_id=batches_id, 
        batches_freq=batches_freq, 
        warm_up_steps=0, 
        init_plan=None,
        init_cache_state=None,
        init_cost=None,
        process_id=process_id,
        init_freq_status=None,
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
    training_plan = data['plan'].to_arrow().to_pylist()
    # training_plan = [i for i in range(len(id_batches))]
    import pdb; pdb.set_trace()
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
            import pdb; pdb.set_trace()
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
    if DATASET == "criteo":
        planner = Planner(
            dataset="criteo",
            plan_path=DLRM_PLAN_PATH, 
            data_path=DLRM_DATA_PATH, 
            log_path=LOG_PATH, 
            cached_rows=DLRM_GPU_CACHE_SIZE, 
            warm_up_steps=0, 
            search_limit=2000, 
            hotness_diff_threshold_base_relax_ratio=0.8, 
            hotness_diff_threshold_relax_ratio_penalty_rate=0.8, 
            hotness_diff_threshold_increment_relax_ratio=0.001, 
            hotness_diff_threshold_late_time_cap=1,
            hotness_diff_threshold_startup_cap=12,
            hotness_diff_threshold_recal_steps=0,
            batch_file=BATCH_FILE_SUFFIX
            )
    else:    
        planner = Planner(
            dataset="taobao",
            plan_path=TBSM_PLAN_PATH, 
            data_path=TBSM_DATA_PATH, 
            log_path=LOG_PATH, 
            cached_rows=TBSM_GPU_CACHE_SIZE, 
            warm_up_steps=0, 
            search_limit=2000, 
            hotness_diff_threshold_base_relax_ratio=0.8, 
            hotness_diff_threshold_relax_ratio_penalty_rate=0.8, 
            hotness_diff_threshold_increment_relax_ratio=0.001, 
            hotness_diff_threshold_late_time_cap=1,
            hotness_diff_threshold_startup_cap=15,
            hotness_diff_threshold_recal_steps=0,
            batch_file=BATCH_FILE_SUFFIX,
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
    # planner.Read_plan("training_plan-15-1024-2")
    # print("number of batches: " + str(len(planner.plan)) + ", calculating cost......")
    
    # # cost, _, _ = New_Simulate_Cost(planner.plan, planner.cached_rows, batches_id, batches_freq)
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

    # Comment date: 20240312
    init_plan_path = "/root/files/coding/data_loading_planner/taobao_run_1/live_backup-lfu--LFU-15-1024-4.json"
    with open(init_plan_path, "r") as backup_file:
        init_plan = json.load(backup_file)
    
    print("Initialized plan has been loaded.")

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

    plan, cost, _, _ = New_LFU_Multiprocess_Search(
        LOG_PATH,
        planner.cached_rows, 
        batches_id, 
        batches_freq,
        warm_up_steps=planner.warm_up_steps, 
        search_limit=planner.search_limit, 
        hotness_diff_threshold_base_relax_ratio=planner.hotness_diff_threshold_base_relax_ratio, 
        hotness_diff_threshold_relax_ratio_penalty_rate=planner.hotness_diff_threshold_relax_ratio_penalty_rate,
        hotness_diff_threshold_increment_relax_ratio=planner.hotness_diff_threshold_increment_relax_ratio,
        hotness_diff_threshold_late_time_cap=planner.hotness_diff_threshold_late_time_cap,
        hotness_diff_threshold_startup_cap=planner.hotness_diff_threshold_startup_cap,
        num_process=40,
        init_plan=init_plan,
    )
    planner.plan = plan[:]

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
    
    # planner.plan = plan[:]

    '''------------------------ New multiprocess planner ------------------------'''
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
    #     simulators.append(Process(target=New_None_LFU_Cost_Wrapper, args=(costs[i], i, random_route, planner.cached_rows, batches_id)))
    #     simulators[i].start()

    # for i in range(num_loop):
    #     simulators[i].join()
    #     cost = costs[i].get()
    #     accumulated_cost = accumulated_cost + cost
    #     print("[random route " + str(i + 1) + "] cost = " + str(cost))
    # cost = accumulated_cost / num_loop

    '''------------------------------- Convertor ------------------------------'''
    # input_path = os.path.join(TBSM_PLAN_PATH, "training_plan-29.parquet")
    # output_path = os.path.join(TBSM_PLAN_PATH, "sequence-id_to_prefetch.parquet")
    # Training_Plan_to_ID_of_Batches(input_path, output_path, batches_id, batches_freq)

    

    '''------------------------------- End ------------------------------'''

    planning_time = time.time() - dataloading_time - start_time
    print("Cost: " + str(cost) + ", ")
    print("dataloading_time: " + str(dataloading_time) + ", planning_time: " + str(planning_time))
    if PLAN_FILE_NAME is not None:
        planner.to_parquet(PLAN_FILE_NAME)


    
