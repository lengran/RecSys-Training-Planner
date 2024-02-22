from os import path
from dlrm_data_pytorch import CriteoDataset
from cached_embeddingbag import CacheManager
import torch
from typing import Sized, Optional, Iterator, List
import pandas
import numpy as np
from concurrent.futures import ThreadPoolExecutor as ConcurrentPoolExecutor             # rename this to support future transparent change to process based parallel execution
import time

class CriteoDataset_WithListIndexSupport(CriteoDataset):
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

class Data_Manager(object):
    '''
    Contains datasets, cache managers and dataloaders.
    '''

    def __init__(self, args) -> None:
        if args.mlperf_logging and args.memory_map and args.data_set == "terabyte":
            # more efficient for larger batches
            data_directory = path.dirname(args.raw_data_file)

            if args.mlperf_bin_loader:
                raise NotImplementedError()
            else:
                '''
                data_filename = args.raw_data_file.split("/")[-1]

                train_data = CriteoDataset(
                    args.data_set,
                    args.max_ind_range,
                    args.data_sub_sample_rate,
                    args.data_randomize,
                    "train",
                    args.raw_data_file,
                    args.processed_data_file,
                    args.memory_map,
                    args.dataset_multiprocessing
                )

                test_data = CriteoDataset(
                    args.data_set,
                    args.max_ind_range,
                    args.data_sub_sample_rate,
                    args.data_randomize,
                    "test",
                    args.raw_data_file,
                    args.processed_data_file,
                    args.memory_map,
                    args.dataset_multiprocessing
                )

                train_loader = data_loader_terabyte.DataLoader(
                    data_directory=data_directory,
                    data_filename=data_filename,
                    days=list(range(23)),
                    batch_size=args.mini_batch_size,
                    max_ind_range=args.max_ind_range,
                    split="train"
                )

                test_loader = data_loader_terabyte.DataLoader(
                    data_directory=data_directory,
                    data_filename=data_filename,
                    days=[23],
                    batch_size=args.test_mini_batch_size,
                    max_ind_range=args.max_ind_range,
                    split="test"
                )
                '''
                raise NotImplementedError()
        else:
            # Datasets
            self.train_data = CriteoDataset_WithListIndexSupport(
                args.data_set,
                args.max_ind_range,
                args.data_sub_sample_rate,
                None, # args.data_randomize,
                "train",
                args.raw_data_file,
                args.processed_data_file,
                args.memory_map,
                args.dataset_multiprocessing
            )

            self.test_data = CriteoDataset_WithListIndexSupport(
                args.data_set,
                args.max_ind_range,
                args.data_sub_sample_rate,
                None, #args.data_randomize,
                "test",
                args.raw_data_file,
                args.processed_data_file,
                args.memory_map,
                args.dataset_multiprocessing
            )

            # Create empty cache now so the dataloader can callback them later.
            self.batch_pointer = 0
            num_embedding_list = self.train_data.counts
            total_embedding_row = 0
            self.offsets = list()
            for i in range(num_embedding_list.size):
                self.offsets.append(total_embedding_row)
                total_embedding_row = total_embedding_row + num_embedding_list[i]

            embedding_dim = args.arch_sparse_feature_size
            self.gpu_cache = CacheManager(num_embeddings=total_embedding_row, embedding_dim=embedding_dim, cache_ratio=args.cache_ratio, pin_weight=True)#, async_copy=True, buffer_size=0)

            self.data_path = args.training_plan_dir
            # First initiate a Sampler that read training plan from a directory, then pass it to DataLoaders
            # TODO: Deal with this gracefully. 1. Use arg.data_randomize. 2. Add another arg to choose whether import training plan or not.
            # custom_sampler = PlanedSampler(True, self.data_path, None, None, None, None)
            custom_sampler = PlanedSampler(False, self.data_path, batch_size=args.mini_batch_size, shuffle=False, drop_last=False, dataset=self.train_data)

            # Dataloaders
            self.train_loader = torch.utils.data.DataLoader(
                self.train_data,
                num_workers=args.num_workers,
                collate_fn=self.collate_cached_wrapper_criteo,
                pin_memory=False,
                sampler=custom_sampler
            )
            # batch_size=args.mini_batch_size,
            # shuffle=False,
            # drop_last=False,  # True

            self.test_loader = torch.utils.data.DataLoader(
                self.test_data,
                batch_size=args.test_mini_batch_size,
                shuffle=False,
                num_workers=args.test_num_workers,
                collate_fn=self.collate_cached_wrapper_criteo,
                pin_memory=False,
                drop_last=False,  # True
            )

            self.sleep_interval = 10        # a large number (seconds)
            self._load_time_stamp = time.time()

    def collate_cached_wrapper_criteo(self, list_of_tuples):
        '''
        Customized collate function that update cache.
        '''
        # where each tuple is (X_int, X_cat, y)
        transposed_data = list(zip(*list_of_tuples[0]))

        X_int = torch.log(torch.tensor(transposed_data[0], dtype=torch.float32) + 1)
        X_cat = torch.tensor(transposed_data[1], dtype=torch.long)
        T = torch.tensor(transposed_data[2], dtype=torch.float32).view(-1, 1)

        batchSize = X_cat.shape[0]
        featureCnt = X_cat.shape[1]

        # Update cache and the batch pointer
        for i in range(featureCnt):
            offseted_ids = X_cat[:, i].flatten().add(self.offsets[i])           # A naive way to hash ids in diff cats, TODO: design a better hash function
            self.gpu_cache.prepare_ids(offseted_ids, self.batch_pointer)
        self.batch_pointer = self.batch_pointer + 1

        lS_i = [X_cat[:, i] for i in range(featureCnt)]
        lS_o = [torch.tensor(range(batchSize)) for _ in range(featureCnt)]
        _tmp_time_stamp = time.time()
        self.sleep_interval = min(self.sleep_interval, _tmp_time_stamp - self._load_time_stamp)
        self._load_time_stamp = _tmp_time_stamp

        return X_int, torch.stack(lS_o), torch.stack(lS_i), T

    # def PlannedCacheRowLoader(self):
    #     """
    #     A separate thread, prefetching planed embedding entries as far as it can.
    #     Issue prefetching request with specified rows operations.
    #     NOTE: This function hasn't been tested.
    #     """
    #     # Lists of list.
    #     ids_to_move_in_batches = pandas.read_parquet(path.join(self.data_path, "ids.parquet")).astype(int).values.tolist()
    #     slots_to_evict_batches = pandas.read_parquet(path.join(self.data_path, "slots_to_evict.parquet")).astype(int).values.tolist()
    #     slots_to_move_in_batches = pandas.read_parquet(path.join(self.data_path, "slots_to_move_in.parquet")).astype(int).values.tolist()
    #     slots_to_update_batches = pandas.read_parquet(path.join(self.data_path, "slots_to_update.parquet")).astype(int).values.tolist()
    #     batch_pointer = 0

    #     import pdb; pdb.set_trace()
    #     while True:
    #         print("Prefetching embedding entries for batch " + str(batch_pointer) + ".")
    #         ids_to_move_in = ids_to_move_in_batches[batch_pointer]
    #         slots_to_evict = slots_to_evict_batches[batch_pointer]
    #         slots_to_move_in = slots_to_move_in_batches[batch_pointer]
    #         slots_to_update = slots_to_update_batches[batch_pointer]

    #         # move rows to gpu (a temporary place not directly to cache)
    #         # TODO: Why are these code here? Shouldn't them belong to the cache manager?
    #         cpu_row_idxs_to_move_in = self.idx_map.index_select(0, ids_to_move_in.to(torch.cuda.current_device()))
    #         cpu_row_idxs_to_move_in_copy = cpu_row_idxs_to_move_in.cpu()
    #         if self._async_copy:
    #             if self.buffer_size == 0:
    #                 evict_in_rows_gpu = (
    #                     self.weight.view(self.num_embeddings, -1).index_select(0, cpu_row_idxs_to_move_in_copy).pin_memory()
    #                 )
    #                 with torch.cuda.stream(self._memcpy_stream):
    #                     evict_in_rows_gpu = evict_in_rows_gpu.to(torch.cuda.current_device(), non_blocking=True)
    #             else:
    #                 raise NotImplemented


    #         succeed = self.gpu_cache.new_prepare_ids(ids_to_move_in, batch_pointer, slots_to_evict, slots_to_move_in, slots_to_update, evict_in_rows_gpu)

    #         if succeed:
    #             batch_pointer += 1
    #         else:
    #             while not succeed:
    #                 time.sleep(self.sleep_interval)
    
    # def CacheRowLoader(self):
    #     """
    #     A separate thread, prefetching embedding entries as far as it can.
    #     Issue prefetching request.
    #     """
    #     # Lists of list.
    #     ids_batches = pandas.read_parquet(path.join(self.data_path, "ids.parquet")).values.tolist()
        
    #     batch_pointer = 0

    #     import pdb; pdb.set_trace()
    #     while True:
    #         print("Prefetching embedding entries for batch " + str(batch_pointer) + ".")
    #         ids_batch = ids_batches[batch_pointer]

    #         # move rows to gpu (a temporary place not directly to cache)
    #         cpu_row_idxs_to_move_in = self.idx_map.index_select(0, ids_to_move_in.to(torch.cuda.current_device()))
    #         cpu_row_idxs_to_move_in_copy = cpu_row_idxs_to_move_in.cpu()
    #         if self._async_copy:
    #             if self.buffer_size == 0:
    #                 evict_in_rows_gpu = (
    #                     self.weight.view(self.num_embeddings, -1).index_select(0, cpu_row_idxs_to_move_in_copy).pin_memory()
    #                 )
    #                 with torch.cuda.stream(self._memcpy_stream):
    #                     evict_in_rows_gpu = evict_in_rows_gpu.to(torch.cuda.current_device(), non_blocking=True)
    #             else:
    #                 raise NotImplemented


    #         succeed = self.gpu_cache.new_prepare_ids(ids_to_move_in, batch_pointer, slots_to_evict, slots_to_move_in, slots_to_update, evict_in_rows_gpu)

    #         if succeed:
    #             batch_pointer += 1
    #         else:
    #             while not succeed:
    #                 time.sleep(self.sleep_interval)