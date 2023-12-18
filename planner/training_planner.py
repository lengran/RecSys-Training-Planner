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

LARGE_NUMBER = 10000000

DLRM_GPU_CACHE_SIZE = int(33762577 * 0.05 * 3)                                                       # num of rows * cache_ratio                     # num of embedding entries = GB / float32 / 64 dimension
TBSM_GPU_CACHE_SIZE = int(5159457 * 0.05 * 3)                                                       # num of rows * cache_ratio
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
        loading_begin = time.time()
        print("Loading data...")
        batches = df.read_parquet(os.path.join(os.path.abspath(plan_path), "batches.parquet")).transpose()             # each column is a batch of indices of dataset's row
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
        cat_data = df.DataFrame(X_cat).astype(int)                                                                          # each column is a row of categorical features in the original dataset
        
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
                idx_freq_evictable_row = freq_register[evictable_row.index].nlargest(num_rows_to_evic, keep='last').index
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
        print(output_str)
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
            freq_batch_ids = self.freq_batches[plan[step]]
            ids_to_comm = batch_ids[batch_ids.isin(cache_state) == False]
            num_ids_to_comm = len(ids_to_comm)
            freq_ids_to_comm = freq_batch_ids[ids_to_comm.index]
            num_rows_to_evic = len(ids_to_comm) - num_available_rows
            if num_rows_to_evic > 0:
                # cache_state = cache_state.iloc[num_rows_to_evic : -1]
                evictable_row = cache_state[cache_state.isin(batch_ids) == False]
                idx_freq_evictable_row = freq_state[evictable_row.index].nlargest(num_rows_to_evic, keep='last').index
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

            for i in range(chunk_size):
                idx = idx_unused_batches.pop(0)
                batch_indices.append(idx)
                self.batches.append(backup_batches[idx])
            
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

        for i in range(len(idx_unused_batches)):
            self.batches.append(backup_batches[idx_unused_batches[i]])
        
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
        self.batch_num = backup_num_batches
        self.warm_up_steps = backup_warm_up_steps
        self.plan = plan

        return cost

    def Read_plan(self, plan_file: str = None):
        if plan_file is None:
            plan_file = ""
        full_path = os.path.join(self.plan_path, ("training_plan" + plan_file + ".parquet"))

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
    
    # planner = Planner(
    #     dataset="criteo",
    #     plan_path=DLRM_PLAN_PATH, 
    #     data_path=DLRM_DATA_PATH, 
    #     log_path="/root/files/coding/data_loading_planner/kaggle_run_2", 
    #     cached_rows=DLRM_GPU_CACHE_SIZE, 
    #     warm_up_steps=150, 
    #     search_limit=2000, 
    #     hotness_diff_threshold_base_relax_ratio=0.9, 
    #     hotness_diff_threshold_relax_ratio_penalty_rate=0.5, 
    #     hotness_diff_threshold_increment_relax_ratio=0.001, 
    #     hotness_diff_threshold_late_time_cap=1,
    #     hotness_diff_threshold_startup_cap=100,
    #     hotness_diff_threshold_recal_steps=0,
    #     )
    
    planner = Planner(
        dataset="taobao",
        plan_path=TBSM_PLAN_PATH, 
        data_path=TBSM_DATA_PATH, 
        log_path="/root/files/coding/data_loading_planner/taobao_run_1", 
        cached_rows=TBSM_GPU_CACHE_SIZE, 
        warm_up_steps=10, 
        search_limit=250, 
        hotness_diff_threshold_base_relax_ratio=0.8, 
        hotness_diff_threshold_relax_ratio_penalty_rate=0.8, 
        hotness_diff_threshold_increment_relax_ratio=0.001, 
        hotness_diff_threshold_late_time_cap=1,
        hotness_diff_threshold_startup_cap=12,
        hotness_diff_threshold_recal_steps=0,
        )
    
    dataloading_time = time.time() - start_time
    
    print("Planner initialized. Dataset loaded. (" + str(dataloading_time) + "s)\nStart planning...")

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
    # cost, _ = planner.Simulate_Cost(planner.plan, None, None, None)

    # Chunked Search
    # cost = planner.Segmented_Search(2000)
    cost = planner.Multiprocess_Search(3500)

    # Check cost
    # planner.Read_plan("-8")
    # cost, _ = planner.Simulate_Cost(planner.plan, None, None, None)
    # print("number of batches: " + str(len(planner.plan)))

    planning_time = time.time() - dataloading_time - start_time

    print("Cost: " + str(cost) + ", dataloading_time: " + str(dataloading_time) + ", planning_time: " + str(planning_time))
    planner.to_parquet("-15-256-2")
