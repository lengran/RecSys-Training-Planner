# Data related code for the pytorch implementation of TBSM model

# private packages
from tbsm_data_pytorch import TBSMDataset
from dlrm.cached_embeddingbag import CacheManager

# miscellaneous
from os import path
from typing import Sized, Optional, Iterator, List

# public packages
import torch
import pandas
import time
import threading

class TBSMDataset_WithListIndexSupport(TBSMDataset):
    def __getitem__(self, index):
        if isinstance(index, list):
            return [self[idx] for idx in index]

        return super().__getitem__(index)

class PlanedSampler(torch.utils.data.Sampler[List[int]]):
    r""" Generate batches according to loaded training plan or from a normal batch sampler.

    Args:
        ready (bool): Work mode. When Ture, the sampler reads batches from a training plan. Otherwise, generate batches from a batch sampler.
        data_path (os.path): Path to the directory where the sampler read plan and batches in / write batches to.
        batch_size (int): A parameter to pass into the batch sampler.
        shuffle (bool): If the sequence of data need to be shuffled before passing into the batch sampler.
        drop_last (bool): A parameter to pass into the batch sampler.
        dataset (Dataset): A parameter to pass into the batch sampler.
    """
    def __init__(
            self, 
            ready: bool, 
            data_path: str, 
            batch_size: Optional[int], 
            shuffle: Optional[bool], 
            drop_last: Optional[bool], 
            dataset: Optional[Sized]
            ) -> None:
        self.data_path = path.abspath(data_path)
        if not path.exists(self.data_path):
            raise ValueError(f"The directory to write batches provided doesn't exist. Path ={data_path}")
        self.ready = ready

        if ready:
            # If already have a training plan, just load it.
            self.batches = pandas.read_parquet(path.join(self.data_path, "batches.parquet")).astype(int).values.tolist()
            self.training_plan = pandas.read_parquet(path.join(self.data_path, "training_plan.parquet")).astype(int).squeeze().tolist()
            self.length = len(self.training_plan)
        else:
            # If not, create samplers to generate batches (which is in method 'generate_batches').
            if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
                raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
            if not isinstance(drop_last, bool):
                raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")
            if shuffle:
                shuffle_sampler = torch.utils.data.RandomSampler(dataset)
                self.batch_sampler = torch.utils.data.BatchSampler(shuffle_sampler, batch_size, drop_last)
            else:
                sequential_sampler = torch.utils.data.SequentialSampler(dataset)
                self.batch_sampler = torch.utils.data.BatchSampler(sequential_sampler, batch_size, drop_last)
            self.batches = list(self.batch_sampler)
            self.length = len(self.batches) 
            self.training_plan = [i for i in range(self.length)]
    
    def generate_batches(self):
        # Write batches to a file so planner can read it and generate a plan.
        dataframe = pandas.DataFrame(self.batches, columns=[str(i) for i in range(len(self.batches[0]))]).fillna(-1).astype(int)
        dataframe.to_parquet(path.join(self.data_path, "batches.parquet"))

    def __iter__(self) -> Iterator[int]:
        # Return a batch each time.
        for i in range(self.length):
            batch_idx = self.training_plan[i]
            if batch_idx != (self.length - 1):
                yield self.batches[batch_idx]
            else:
                # Deal with placebo -1 before acutally returning data indices.
                batch = self.batches[batch_idx]
                filtered_batch = [i for i in batch if i != -1 ]
                yield filtered_batch

    def __len__(self) -> int:
        return self.length

class DataManager():
    # creates a loader (train, val or test data) to be used in the main training loop
    # or during inference step
    def __init__(self, args):
        """
        if mode == "train":
            raw = args.raw_train_file
            proc = args.pro_train_file
            numpts = args.num_train_pts
            batchsize = args.mini_batch_size
            doshuffle = True
        elif mode == "val":
            raw = args.raw_train_file
            proc = args.pro_val_file
            numpts = args.num_val_pts
            batchsize = 25000
            doshuffle = True
        else:
            raw = args.raw_test_file
            proc = args.pro_test_file
            numpts = 1
            batchsize = 25000
            doshuffle = False
        """

        self.train_data = TBSMDataset_WithListIndexSupport(
            args.datatype,
            "train",
            args.ts_length,
            args.points_per_user,
            args.numpy_rand_seed,
            args.raw_train_file,
            args.pro_train_file,
            args.arch_embedding_size,
            args.num_train_pts,
        )

        self.val_data = TBSMDataset_WithListIndexSupport(
            args.datatype,
            "val",
            args.ts_length,
            args.points_per_user,
            args.numpy_rand_seed,
            args.raw_train_file,
            args.pro_val_file,
            args.arch_embedding_size,
            args.num_val_pts,
        )

        # Create empty cache
        self.batch_pointer = 0      # infinite prefetching count
        self.loader_count = 0       # dataloader count
        self.num_cat_counts = [987994, 4162024, 9439]        # I got these three values from the dataset init method.
        num_embedding_rows = 0
        self.offsets = list()
        for i in range(len(self.num_cat_counts)):
            self.offsets.append(num_embedding_rows)
            num_embedding_rows = num_embedding_rows + self.num_cat_counts[i]
        embedding_dim = args.arch_sparse_feature_size
        self.gpu_cache = CacheManager(num_embeddings=num_embedding_rows, embedding_dim=embedding_dim, cache_ratio=args.cache_ratio, pin_weight=True, async_copy=True, buffer_size=0)

        # Create a special sampler to support import and export of training plans. TODO: make this switchable from outside this script, elegantly.
        custom_train_sampler = PlanedSampler(True, args.training_plan_dir, None, None, None, None)
        # custom_train_sampler = PlanedSampler(False, args.training_plan_dir, batch_size=args.mini_batch_size, shuffle=True, drop_last=False, dataset=self.train_data)

        val_sampler = PlanedSampler(False, args.training_plan_dir, batch_size=15000, shuffle=True, drop_last=False, dataset=self.val_data)

        '''
        # debug
        self.train_count = 0
        self.val_count = 0
        '''
        
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            # batch_size=args.mini_batch_size,
            num_workers=0,
            collate_fn=self.collate_wrapper_tbsm,
            # shuffle=True,
            sampler=custom_train_sampler,
        )

        self.val_loader = torch.utils.data.DataLoader(
            self.val_data,
            # batch_size=25000,
            num_workers=0,
            collate_fn=self.collate_wrapper_tbsm,
            # collate_fn=self.collate_wrapper_tbsm_1,             # debug
            # shuffle=True,
            sampler=val_sampler,
        )

        # debug
        self.prefetching_signal = threading.Event()             # Use this to block prefetching
        self.prefetching_ready = threading.Event()              # Use this to temporarily pause prefetching when there isn't enough space in cache.
        self.training_ready = threading.Event()

        self.infinite_prefetch = True

        if self.infinite_prefetch:
            self.prefetch_wait_step_limit = 75
            self.prefetch_wait_step_count = self.prefetch_wait_step_limit
            self.prefetching_ready.set()
            self.prefetching_thread = threading.Thread(target=self.CacheRowLoader, args=[args.training_plan_dir])
            self.prefetching_thread.start()
        else:
            self.training_ready.set()

        # return loader, len(data)

    # defines transform to be performed during each call to batch,
    # used by loader
    # It calls the cache update method
    def collate_wrapper_tbsm(self, list_of_tuples):
        '''
        # debug
        print("===========================[calling train dataloader (" + str(self.train_count) + ")]===========================")
        self.train_count = self.train_count + 1
        '''

        # turns tuple into X, S_o, S_i, take last ts_length items
        data = list(zip(*list_of_tuples[0]))                                # don't know why there is an extra list wrapping actual batch
        all_cat = torch.tensor(data[0], dtype=torch.long, device=torch.cuda.current_device())
        all_int = torch.tensor(data[1], dtype=torch.float, device=torch.cuda.current_device())

        
        if self.infinite_prefetch:
            # Update cache and batch pointer (Replaced by infinite prefetching)
            # print("[Dataloader] Loading batch " + str(self.loader_count) + ", current prefetching step is " + str(self.batch_pointer))
            if self.loader_count >= self.batch_pointer:
                self.prefetch_wait_step_count = self.prefetch_wait_step_limit
                # print("[Warning] Training waits!")
                # start_time = time.time()
                self.training_ready.clear()
                self.training_ready.wait()
                # end_time = time.time()
                # print("[Warning] Training has waited for " + str(end_time - start_time) + " seconds.")
    
            self.loader_count += 1
        else:
            # start_time = time.time()
            tmp = 0
            tmp_offset = list()
            for i in range(len(self.num_cat_counts)):
                tmp_offset.append([tmp] * all_cat.shape[2])
                tmp = tmp + self.num_cat_counts[i]
            tmp_tensor = torch.tensor(tmp_offset, device=torch.cuda.current_device())
            ids = all_cat + tmp_tensor
            unique_ids = torch.unique(ids.flatten())
            self.gpu_cache.none_lfu_prepare_ids(unique_ids, self.batch_pointer)
            self.batch_pointer = self.batch_pointer + 1
            # end_time = time.time()
            # print("[Synchronized fetching] Finished for batch " + str(self.batch_pointer) + " (" + str(end_time - start_time) + "s)")

        # print("shapes:", all_cat.shape, all_int.shape)

        num_den_fea = all_int.shape[1]
        num_cat_fea = all_cat.shape[1]
        batchSize = all_cat.shape[0]
        ts_len = all_cat.shape[2]
        all_int = torch.reshape(all_int, (batchSize, num_den_fea * ts_len))

        X = []
        lS_i = []
        lS_o = []

        # transform data into the form used in dlrm nn
        for j in range(ts_len):

            lS_i_h = []
            for i in range(num_cat_fea):
                lS_i_h.append(all_cat[:, i, j])

            lS_o_h = [torch.tensor(range(batchSize), device=torch.cuda.current_device()) for _ in range(len(lS_i_h))]
            # lS_o_h = [torch.tensor(range(batchSize)) for _ in range(len(lS_i_h))]

            lS_i.append(lS_i_h)
            lS_o.append(lS_o_h)
            X.append(all_int[:, j].view(-1, 1))

        T = torch.tensor(data[2], dtype=torch.float32).view(-1, 1)

        return X, lS_o, lS_i, T
    
    '''
    def collate_wrapper_tbsm_1(self, list_of_tuples):
        # debug
        print("===========================[calling val dataloader (" + str(self.val_count) + ")]===========================")
        self.val_count = self.val_count + 1

        # turns tuple into X, S_o, S_i, take last ts_length items
        data = list(zip(*list_of_tuples[0]))                                # don't know why there is an extra list wrapping actual batch
        all_cat = torch.tensor(data[0], dtype=torch.long, device=torch.cuda.current_device())
        all_int = torch.tensor(data[1], dtype=torch.float, device=torch.cuda.current_device())

        # Update cache and batch pointer
        tmp = 0
        tmp_offset = list()
        for i in range(len(self.num_cat_counts)):
            tmp_offset.append([tmp] * all_cat.shape[2])
            tmp = tmp + self.num_cat_counts[i]
        tmp_tensor = torch.tensor(tmp_offset, device=torch.cuda.current_device())
        ids = all_cat + tmp_tensor
        unique_ids = torch.unique(ids.flatten())
        self.gpu_cache.prepare_ids(unique_ids, self.batch_pointer)
        self.batch_pointer = self.batch_pointer + 1

        # print("shapes:", all_cat.shape, all_int.shape)

        num_den_fea = all_int.shape[1]
        num_cat_fea = all_cat.shape[1]
        batchSize = all_cat.shape[0]
        ts_len = all_cat.shape[2]
        all_int = torch.reshape(all_int, (batchSize, num_den_fea * ts_len))

        X = []
        lS_i = []
        lS_o = []

        # transform data into the form used in dlrm nn
        for j in range(ts_len):

            lS_i_h = []
            for i in range(num_cat_fea):
                lS_i_h.append(all_cat[:, i, j])

            lS_o_h = [torch.tensor(range(batchSize), device=torch.cuda.current_device()) for _ in range(len(lS_i_h))]
            # lS_o_h = [torch.tensor(range(batchSize)) for _ in range(len(lS_i_h))]

            lS_i.append(lS_i_h)
            lS_o.append(lS_o_h)
            X.append(all_int[:, j].view(-1, 1))

        T = torch.tensor(data[2], dtype=torch.float32).view(-1, 1)

        return X, lS_o, lS_i, T
    '''

    def PlannedCacheRowLoader(self):
        """
        A separate thread, prefetching planed embedding entries as far as it can.
        Issue prefetching request with specified rows operations.
        NOTE: This function hasn't been tested.
        """
        # Lists of list.
        ids_to_move_in_batches = pandas.read_parquet(path.join(self.data_path, "ids.parquet")).astype(int).values.tolist()
        slots_to_evict_batches = pandas.read_parquet(path.join(self.data_path, "slots_to_evict.parquet")).astype(int).values.tolist()
        slots_to_move_in_batches = pandas.read_parquet(path.join(self.data_path, "slots_to_move_in.parquet")).astype(int).values.tolist()
        slots_to_update_batches = pandas.read_parquet(path.join(self.data_path, "slots_to_update.parquet")).astype(int).values.tolist()
        batch_pointer = 0

        while True:
            print("Prefetching embedding entries for batch " + str(batch_pointer) + ".")
            ids_to_move_in = ids_to_move_in_batches[batch_pointer]
            slots_to_evict = slots_to_evict_batches[batch_pointer]
            slots_to_move_in = slots_to_move_in_batches[batch_pointer]
            slots_to_update = slots_to_update_batches[batch_pointer]

            # move rows to gpu (a temporary place not directly to cache)
            # TODO: Why are these code here? Shouldn't them belong to the cache manager?
            cpu_row_idxs_to_move_in = self.idx_map.index_select(0, ids_to_move_in.to(torch.cuda.current_device()))
            cpu_row_idxs_to_move_in_copy = cpu_row_idxs_to_move_in.cpu()
            if self._async_copy:
                if self.buffer_size == 0:
                    evict_in_rows_gpu = (
                        self.weight.view(self.num_embeddings, -1).index_select(0, cpu_row_idxs_to_move_in_copy).pin_memory()
                    )
                    with torch.cuda.stream(self._memcpy_stream):
                        evict_in_rows_gpu = evict_in_rows_gpu.to(torch.cuda.current_device(), non_blocking=True)
                else:
                    raise NotImplemented


            succeed = self.gpu_cache.new_prepare_ids(ids_to_move_in, batch_pointer, slots_to_evict, slots_to_move_in, slots_to_update, evict_in_rows_gpu)

            if succeed:
                batch_pointer += 1
            else:
                while not succeed:
                    time.sleep(self.sleep_interval)
    
    @torch.no_grad()
    def NoneLFUCacheRowLoader(self, plan_path: str):
        """
        A separate thread, prefetching embedding entries as far as it can.
        Issue prefetching request.
        """
        # Lists of list.
        ids_batches = pandas.read_parquet(path.join(plan_path, "id_to_prefetch.parquet"))["id_planed_batches"].tolist()       # type(ids_batches) = list, type(ids_batches) = np.ndarray
        
        self.batch_pointer = 0

        for _ in range(len(ids_batches)):
            # start_time = time.time()
            # First compute which ids need to be transfered.
            self.prefetching_signal.wait()
            # print("[Infinite Prefetching] Prefetching embedding entries for batch " + str(self.batch_pointer) + ".")
            ids_batch = torch.tensor(ids_batches[self.batch_pointer], device=torch.cuda.current_device())
            cpu_row_idxs = self.gpu_cache.idx_map.index_select(0, ids_batch)                                                    # The indices are unique.
            # self.gpu_cache.evict_backlist = cpu_row_idxs
            mask_comm_cpu_row_idxs = torch.isin(cpu_row_idxs, self.gpu_cache.cached_idx_map, invert=True)
            comm_cpu_row_idxs = cpu_row_idxs[mask_comm_cpu_row_idxs]
            update_cpu_row_idxs = cpu_row_idxs[torch.logical_not(mask_comm_cpu_row_idxs)]

            # Update batch flags to protect useful rows from being evicted.
            gpu_idxs_to_update_flag = self.gpu_cache.inverted_cached_idx[update_cpu_row_idxs]
            self.gpu_cache.batch_flags.index_fill_(0, gpu_idxs_to_update_flag, self.batch_pointer)

            # Move rows to gpu (to a temporary buffer, not directly to cache rows)
            self.prefetching_signal.wait()
            comm_cpu_row_idxs_copy = comm_cpu_row_idxs.cpu()
            if self.gpu_cache._async_copy:
                if self.gpu_cache.buffer_size == 0:
                    evict_in_rows_gpu = self.gpu_cache.weight.view(self.gpu_cache.num_embeddings, -1).index_select(0, comm_cpu_row_idxs_copy).pin_memory()
                    with torch.cuda.stream(self.gpu_cache._memcpy_stream):
                        evict_in_rows_gpu = evict_in_rows_gpu.to(torch.cuda.current_device(), non_blocking=True)
                else:
                    raise NotImplemented

            # Receive data on GPU. This part could be blocked and could be resumed by the training script.
            succeed = self.gpu_cache.none_lfu_new_prepare_ids_part_2(comm_cpu_row_idxs, self.batch_pointer, evict_in_rows_gpu)
            # begin_wait = time.time()
            # waiting_time = 0
            self.prefetch_wait_step_count = 0
            while not succeed:
                self.prefetching_ready.clear()
                
                while self.prefetch_wait_step_count < self.prefetch_wait_step_limit:
                    self.prefetching_ready.wait()
                    self.prefetch_wait_step_count += 1
                    # if self.prefetch_wait_step_count < self.prefetch_wait_step_limit:
                    self.prefetching_ready.clear()

                # end_wait = time.time()
                # waiting_time = end_wait - begin_wait
                succeed = self.gpu_cache.none_lfu_new_prepare_ids_part_2(comm_cpu_row_idxs, self.batch_pointer, evict_in_rows_gpu)
        
            self.batch_pointer += 1
            self.training_ready.set()
            # end_time = time.time()
            # print("[Infinite Prefetching] Finished prefetching for batch " + str(self.batch_pointer) + " (" + str(end_time - start_time - waiting_time) + "s, waiting time " + str(waiting_time) + ")")

    @torch.no_grad()
    def CacheRowLoader(self, plan_path: str):
        """
        A separate thread, prefetching embedding entries as far as it can.
        Issue prefetching request.
        """
        # Lists of list.
        data = pandas.read_parquet(path.join(plan_path, "id_to_prefetch.parquet"))
        ids_batches = data["id_planed_batches"].tolist()       # type(ids_batches) = list, type(ids_batches) = np.ndarray
        freq_batches = data["freq_planed_batches"].tolist()
        
        self.batch_pointer = 0

        for _ in range(len(ids_batches)):
            # start_time = time.time()
            # First compute which ids need to be transfered.
            self.prefetching_signal.wait()
            # print("[Infinite Prefetching] Prefetching embedding entries for batch " + str(self.batch_pointer) + ".")
            ids_batch = torch.tensor(ids_batches[self.batch_pointer], device=torch.cuda.current_device())
            freq_batch = torch.tensor(freq_batches[self.batch_pointer], device=torch.cuda.current_device(), dtype=self.gpu_cache.freq_cnter.dtype)
            cpu_row_idxs = self.gpu_cache.idx_map.index_select(0, ids_batch)                                                    # The indices are unique.
            # self.gpu_cache.evict_backlist = cpu_row_idxs
            mask_comm_cpu_row_idxs = torch.isin(cpu_row_idxs, self.gpu_cache.cached_idx_map, invert=True)
            comm_cpu_row_idxs = cpu_row_idxs[mask_comm_cpu_row_idxs]
            mask_update_cpu_row_idxs = torch.logical_not(mask_comm_cpu_row_idxs)

            # Update batch flags to protect useful rows from being evicted.
            update_cpu_row_idxs = cpu_row_idxs[mask_update_cpu_row_idxs]
            update_freq = freq_batch[mask_update_cpu_row_idxs]
            gpu_idxs_to_update_flag = self.gpu_cache.inverted_cached_idx[update_cpu_row_idxs]
            self.gpu_cache.batch_flags.index_fill_(0, gpu_idxs_to_update_flag, self.batch_pointer)
            self.gpu_cache.freq_cnter.scatter_add_(0, gpu_idxs_to_update_flag, update_freq)

            # Move rows to gpu (to a temporary buffer, not directly to cache rows)
            comm_cpu_row_idxs_copy = comm_cpu_row_idxs.cpu()
            if self.gpu_cache._async_copy and self.gpu_cache.buffer_size == 0:
                evict_in_rows_gpu = self.gpu_cache.weight.view(self.gpu_cache.num_embeddings, -1).index_select(0, comm_cpu_row_idxs_copy).pin_memory()
                with torch.cuda.stream(self.gpu_cache._memcpy_stream):
                    evict_in_rows_gpu = evict_in_rows_gpu.to(torch.cuda.current_device(), non_blocking=True)
            else:
                raise NotImplemented

            # Receive data on GPU. This part could be blocked and could be resumed by the training script.
            comm_freq = freq_batch[mask_comm_cpu_row_idxs]
            succeed = self.gpu_cache.new_prepare_ids_part_2(comm_cpu_row_idxs, self.batch_pointer, evict_in_rows_gpu, comm_freq)
            # begin_wait = time.time()
            # waiting_time = 0
            self.prefetch_wait_step_count = 0
            while not succeed:
                # print("[Infinite Prefetching] No enough rows. Prefetching waits.")
                self.prefetching_ready.clear()
                self.prefetching_ready.wait()
                
                # Potential risk of a dead lock situation
                while self.prefetch_wait_step_count < self.prefetch_wait_step_limit:
                    self.prefetching_ready.wait()
                    self.prefetch_wait_step_count += 1
                    # if self.prefetch_wait_step_count < self.prefetch_wait_step_limit:
                    self.prefetching_ready.clear()

                # end_wait = time.time()
                # waiting_time = end_wait - begin_wait
                succeed = self.gpu_cache.new_prepare_ids_part_2(comm_cpu_row_idxs, self.batch_pointer, evict_in_rows_gpu, comm_freq)
        
            self.batch_pointer += 1
            self.training_ready.set()
            # end_time = time.time()
            # print("[Infinite Prefetching] Finished prefetching for batch " + str(self.batch_pointer - 1) + " (" + str(end_time - start_time - waiting_time) + "s, waiting time " + str(waiting_time) + ")")