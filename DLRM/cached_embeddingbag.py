import torch
import sys
from typing import Optional

class LimitBuffIndexCopyer(object):
    """LimitBuffIndexCopyer
    Index Copy using limited temp buffer on CUDA.

    Args:
        size (int): buffer size
    """

    def __init__(self, size: int) -> None:
        self._buff_size = size

    @torch.no_grad()
    def index_copy(self, dim: int, src_index: torch.LongTensor, tgt_index: torch.LongTensor, src: torch.Tensor, tgt: torch.Tensor):
        """copy
        src tensor[src_index] -(index_select)-> tmp -(index_copy_)-> tgt tensor [tgt_index]
        The valid rows in the src tensor are continuous, while rows in tgt tensor is scattered.

        Args:
            dim (int):  dimension along which to index
            src_index (int): indices of src tensor to select from
            tgt_index (int): indices of tgt tensor to select from
            src (torch.Tensor):  the tensor containing values to copy
            tgt (torch.Tensor):  the tensor to be copied
        """
        # tgt.index_copy_(dim, index, src)
        assert dim == 0, "only support index_copy on dim 0"
        assert tgt.dim() == 2
        assert src.dim() == 2
        tgt_device = tgt.device
        src_device = src.device

        assert src_index.numel() == tgt_index.numel()
        dim_size = src_index.numel()
        src_index = src_index.to(src_device)
        for begin_pos in range(0, dim_size, self._buff_size):
            cur_len = min(self._buff_size, dim_size - begin_pos)
            src_idx_piece = src_index.narrow(0, begin_pos, cur_len)
            if src_device.type == "cpu" and tgt_device.type == "cuda":
                cpu_tmp_buffer = src.index_select(dim, src_idx_piece).pin_memory()
                tmp_buffer = torch.empty_like(cpu_tmp_buffer, device=tgt_device)
                tmp_buffer.copy_(cpu_tmp_buffer)
            else:
                tmp_buffer = src.index_select(dim, src_idx_piece).to(tgt_device)
            tgt_idx_piece = tgt_index.narrow(0, begin_pos, cur_len)
            tgt.index_copy_(dim, tgt_idx_piece, tmp_buffer)

def _wait_for_data(t, stream: Optional[torch.cuda.streams.Stream]) -> None:
    if stream is None:
        return
    torch.cuda.current_stream().wait_stream(stream)
    # As mentioned in https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html,
    # PyTorch uses the "caching allocator" for memory allocation for tensors. When a tensor is
    # freed, its memory is likely to be reused by newly constructed tensors.  By default,
    # this allocator traces whether a tensor is still in use by only the CUDA stream where it
    # was created.   When a tensor is used by additional CUDA streams, we need to call record_stream
    # to tell the allocator about all these streams.  Otherwise, the allocator might free the
    # underlying memory of the tensor once it is no longer used by the creator stream.  This is
    # a notable programming trick when we write programs using multi CUDA streams.
    cur_stream = torch.cuda.current_stream()
    assert isinstance(t, torch.Tensor)
    t.record_stream(cur_stream)

class CacheManager(torch.nn.Module):
    """
    A software cache that manages parameter storage and transfer between CPU memory and GPU memory.
    Args:
        weight (torch.Tensor)
        cuda_row_num (int, optional): Size of GPU cache, number of embedding vectors it contains. Defaults to 0.
        buffer_size (int, optional): The number of rows in a data transmitter buffer. Defaults to 50_000.
        pin_weight (bool, optional): Use pin memory to store the cpu weight. If set `True`, the cpu memory usage will increase largely. Defaults to False.
        async_copy (bool, optional): Enable asynchronize data transfer between CPU and GPU. Defaults to False.
    """

    def __init__(
        self,
        weight: torch.Tensor = None,
        cuda_row_num: int = 0,
        buffer_size: int = 0,
        pin_weight: bool = False,
        async_copy: bool = False
    ) -> None:
        '''
        A fake init method to enable "create an object before actually use it".
        Create the cache object before dataloaders exist so it can callback to update cache rows.
        The actual init of the cache is after embedding layer created)
        '''
        super().__init__()

        if weight is not None:
            self.init(weight=weight, cuda_row_num=cuda_row_num, buffer_size=buffer_size, pin_weight=pin_weight, async_copy=async_copy)

    def init(
        self,
        weight: torch.Tensor,
        cuda_row_num: int = 0,
        buffer_size: int = 0,
        pin_weight: bool = False,
        async_copy: bool = False
    ) -> None:
        self.buffer_size = buffer_size
        self.num_embeddings, self.embedding_dim = weight.shape
        self.cuda_row_num = cuda_row_num
        self._cuda_available_row_num = self.cuda_row_num
        self.pin_weight = pin_weight
        self.elem_size_in_byte = weight.element_size()

        self._init_weight(weight)

        self._async_copy = async_copy
        if self._async_copy:
            self._memcpy_stream = torch.cuda.Stream()
        
        # Init comm stats
        self._elapsed_dict = {}
        self._show_cache_miss = True
        self._reset_comm_stats()

        # Perf log
        self.num_hits_history = []
        self.num_miss_history = []
        self.num_write_back_history = []

        # Register for batch flags (gpu_row_idx -> batch_flag)
        self.register_buffer(
            "batch_flags",
            torch.empty(self.cuda_row_num, device=torch.cuda.current_device(), dtype=torch.long).fill_(-1),
            persistent=False,
        )
        self._finished_batch = -1
    
    def _init_weight(self, weight) -> None:
        if self.cuda_row_num > 0:
            # Enable cache with introducing auxiliary data structures
            self.cuda_cached_weight = torch.nn.Parameter(
                torch.zeros(
                    self.cuda_row_num, self.embedding_dim, device=torch.cuda.current_device(), dtype=weight.dtype
                )
            )

            # Pin memory cpu for higher CPU-GPU copy bandwidth
            self.weight = weight.pin_memory() if self.pin_weight else weight
            
            # Map IDs and indices 
            # id -> cpu_row_idx
            self.register_buffer(
                "idx_map",
                torch.arange(self.num_embeddings, dtype=torch.long, device=torch.cuda.current_device()),
                persistent=False,
            )

            # gpu_row_idx -> cpu_row_idx
            self.register_buffer(
                "cached_idx_map",
                torch.empty(self.cuda_row_num, device=torch.cuda.current_device(), dtype=torch.long).fill_(-1),
                persistent=False,
            )

            # cpu_row_id -> gpu_row_idx.
            # gpu_row_idx as -1 means cpu_row_id not in CUDA.
            self.register_buffer(
                "inverted_cached_idx",
                torch.zeros(self.num_embeddings, device=torch.cuda.current_device(), dtype=torch.long).fill_(-1),
                persistent=False,
            )

            self.evict_backlist = torch.tensor([], device=torch.cuda.current_device())

            # index copy buffer size should less than 10% of cuda weight.
            if self.buffer_size > 0:
                self.limit_buff_index_copyer = LimitBuffIndexCopyer(self.buffer_size)

        else:
            # Disable cache so that FreqCacheEmbedding is compatible with vanilla EmbeddingBag
            # self.weight = torch.nn.Parameter(weight)
            # self.cuda_cached_weight = self.weight
            raise NotImplementedError()

    def _reset_comm_stats(self) -> None:
        for k in self._elapsed_dict.keys():
            self._elapsed_dict[k] = 0

        self._cpu_to_cuda_numel = 0
        self._cuda_to_cpu_numel = 0
        if self._show_cache_miss:
            self._cache_miss = 0
            self._total_cache = 0

    def prepare_ids(self, ids: torch.Tensor, batch_pointer: torch.Tensor) -> torch.Tensor:
        """
        move the cpu embedding rows w.r.t. ids into CUDA memory
        Args:
            ids (torch.Tensor): the ids to be computed
            batch_pointer (torch.LongTensor): the pointer to (or index of) the current loading batch
        Returns:
            torch.Tensor: indices on the cuda_cached_weight.
        """
        torch.cuda.synchronize()
        with self.timer("cache_op") as gtimer:
            # identify cpu rows to cache
            with self.timer("1_identify_cpu_row_idxs") as timer:
                with torch.profiler.record_function("(cache) get unique indices"):
                    cpu_row_idxs, repeat_times = torch.unique(self.idx_map.index_select(0, ids), return_counts=True)

                    assert len(cpu_row_idxs) <= self.cuda_row_num, (
                        f"You move {len(cpu_row_idxs)} embedding rows from CPU to CUDA. "
                        f"It is larger than the capacity of the cache, which at most contains {self.cuda_row_num} rows, "
                        f"Please increase cuda_row_num or decrease the training batch size."
                    )
                    self.evict_backlist = cpu_row_idxs
                    tmp = torch.isin(cpu_row_idxs, self.cached_idx_map, invert=True)
                    comm_cpu_row_idxs = cpu_row_idxs[tmp]           # indices to be transfered

                    if self._show_cache_miss:
                        self._cache_miss += torch.sum(repeat_times[tmp])
                        self._total_cache += ids.numel()

            self.num_hits_history.append(len(cpu_row_idxs) - len(comm_cpu_row_idxs))
            self.num_miss_history.append(len(comm_cpu_row_idxs))
            self.num_write_back_history.append(0)

            # move sure the cuda rows will not be evicted!
            with torch.profiler.record_function("(cache) prepare_rows_on_cuda"):
                with self.timer("prepare_rows_on_cuda") as timer:
                    self._prepare_rows_on_cuda(comm_cpu_row_idxs)

            self.evict_backlist = torch.tensor([], device=cpu_row_idxs.device, dtype=cpu_row_idxs.dtype)

            with self.timer("6_update_cache") as timer:
                with torch.profiler.record_function("6_update_cache"):
                    gpu_row_idxs = self._id_to_cached_cuda_id(ids)
                
                # Update batch flag
                unique_gpu_row_idxs = self.inverted_cached_idx[cpu_row_idxs]
                self.batch_flags.index_fill_(0, unique_gpu_row_idxs, batch_pointer)

        return gpu_row_idxs

    @torch.no_grad()
    def _prepare_rows_on_cuda(self, cpu_row_idxs: torch.Tensor) -> None:
        """prepare rows in cpu_row_idxs on CUDA memory
        Args:
            cpu_row_idxs (torch.Tensor): the rows to be placed on CUDA
        """
        evict_num = cpu_row_idxs.numel() - self.cuda_available_row_num

        cpu_row_idxs_copy = cpu_row_idxs.cpu()

        # move evict in rows to gpu
        if self._async_copy:
            if self.buffer_size == 0:
                evict_in_rows_gpu = (
                    self.weight.view(self.num_embeddings, -1).index_select(0, cpu_row_idxs_copy).pin_memory()
                )
                with torch.cuda.stream(self._memcpy_stream):
                    evict_in_rows_gpu = evict_in_rows_gpu.to(torch.cuda.current_device(), non_blocking=True)
            else:
                raise NotImplemented

        if evict_num > 0:
            with self.timer("2_identify_cuda_row_idxs") as timer:
                mask_cpu_row_idx = torch.isin(self.cached_idx_map, self.evict_backlist)
                invalid_idxs = torch.nonzero(mask_cpu_row_idx).squeeze(1)
                
                # Batch flag based evict strategy
                with self.timer("2_find_evict_gpu_idxs") as timer:
                    evict_gpu_row_idxs = self._find_evict_gpu_idxs(evict_num)

                if self._async_copy:
                    evict_out_rows_gpu = self.cuda_cached_weight.view(self.cuda_row_num, -1).index_select(
                        0, evict_gpu_row_idxs
                    )
                    evict_out_rows_cpu = torch.empty_like(evict_out_rows_gpu, device="cpu", pin_memory=True)
                    with torch.cuda.stream(None):
                        evict_out_rows_cpu.copy_(evict_out_rows_gpu, non_blocking=True)

                evict_info = self.cached_idx_map[evict_gpu_row_idxs]

            with self.timer("3_evict_out") as timer:
                if self.buffer_size > 0:
                    self.limit_buff_index_copyer.index_copy(
                        0,
                        src_index=evict_gpu_row_idxs,
                        tgt_index=evict_info.cpu(),
                        src=self.cuda_cached_weight.view(self.cuda_row_num, -1),
                        tgt=self.weight.view(self.num_embeddings, -1),
                    )
                else:
                    # allocate tmp memory on CPU and copy rows on CUDA to CPU.
                    # TODO async gpu -> cpu
                    if self._async_copy:
                        _wait_for_data(evict_out_rows_cpu, None)
                    else:
                        with self.timer("3_1_evict_out_index_select") as timer:
                            evict_out_rows_cpu = self.cuda_cached_weight.view(self.cuda_row_num, -1).index_select(
                                0, evict_gpu_row_idxs
                            )
                        with self.timer("3_2_evict_out_gpu_to_cpu_copy") as timer:
                            evict_out_rows_cpu = evict_out_rows_cpu.cpu()

                    with self.timer("3_2_evict_out_cpu_copy") as timer:
                        self.weight.view(self.num_embeddings, -1).index_copy_(0, evict_info.cpu(), evict_out_rows_cpu)

                self.cached_idx_map.index_fill_(0, evict_gpu_row_idxs, -1)
                self.inverted_cached_idx.index_fill_(0, evict_info, -1)
                # self.freq_cnter.index_fill(0, evict_gpu_row_idxs, sys.maxsize) # unnecessary
                self._cuda_available_row_num += evict_num

                weight_size = evict_gpu_row_idxs.numel() * self.embedding_dim
                self._cuda_to_cpu_numel += weight_size
            # print(f"evict embedding weight: {weight_size*self.elem_size_in_byte/1e6:.2f} MB")

        # slots of cuda weight to evict in
        with self.timer("4_identify_cuda_slot") as timer:
            slots = torch.nonzero(self.cached_idx_map == -1).squeeze(1)[: cpu_row_idxs.numel()]

        # TODO wait for optimize
        with self.timer("5_evict_in") as timer:
            # Here also allocate extra memory on CUDA. #cpu_row_idxs
            if self.buffer_size > 0:
                self.limit_buff_index_copyer.index_copy(
                    0,
                    src_index=cpu_row_idxs_copy,
                    tgt_index=slots,
                    src=self.weight.view(self.num_embeddings, -1),
                    tgt=self.cuda_cached_weight.view(self.cuda_row_num, -1),
                )
            else:
                if self._async_copy:
                    _wait_for_data(evict_in_rows_gpu, self._memcpy_stream)
                else:
                    with self.timer("5_1_evict_in_index_select") as timer:
                        # narrow index select to a subset of self.weight
                        # tmp = torch.narrow(self.weight.view(self.num_embeddings, -1), 0, min(cpu_row_idxs).cpu(), max(cpu_row_idxs) - min(cpu_row_idxs) + 1)
                        # evict_in_rows_gpu = tmp.index_select(0, cpu_row_idxs_copy - min(cpu_row_idxs).cpu())
                        evict_in_rows_gpu = (
                            self.weight.view(self.num_embeddings, -1).index_select(0, cpu_row_idxs_copy).pin_memory()
                        )

                    with self.timer("5_2_evict_in_gpu_to_cpu_copy") as timer:
                        evict_in_rows_gpu = evict_in_rows_gpu.cuda()

                    with self.timer("5_3_evict_in_index_copy") as timer:
                        self.cuda_cached_weight.view(self.cuda_row_num, -1).index_copy_(0, slots, evict_in_rows_gpu)

        with self.timer("6_update_cache") as timer:
            self.cached_idx_map[slots] = cpu_row_idxs
            self.inverted_cached_idx.index_copy_(0, cpu_row_idxs, slots)
            self._cuda_available_row_num -= cpu_row_idxs.numel()
            # Don't need to update batch flags because this method only cares the transfered rows. Do it in prepare_id method instead.

        weight_size = cpu_row_idxs.numel() * self.embedding_dim
        self._cpu_to_cuda_numel += weight_size
        # print(f"admit embedding weight: {weight_size*self.elem_size_in_byte/1e6:.2f} MB")

    
    def _find_evict_gpu_idxs(self, evict_num: int) -> torch.Tensor:
        """_find_evict_gpu_idxs
        Find the gpu idxs to be evicted, according to their batch flags and current self._finished_batch flag.
        Args:
            evict_num (int): how many rows has to be evicted
        Returns:
            torch.Tensor: a list tensor (1D), contains the gpu_row_idxs.
        """
        victim_gpu_rows = torch.nonzero(self.batch_flags < self._finished_batch).squeeze(1)[ : evict_num]
        return victim_gpu_rows

    @torch.no_grad()
    def _id_to_cached_cuda_id(self, ids: torch.Tensor) -> torch.Tensor:
        """
        convert ids to indices in self.cuda_cached_weight.
        Implemented with parallel operations on GPU.
        Args:
            ids (torch.Tensor): ids from the dataset
        Returns:
            torch.Tensor: contains indices in self.cuda_cached_weight
        """
        ids = self.idx_map.index_select(0, ids.view(-1))
        ret = self.inverted_cached_idx.index_select(0, ids)
        return ret

    @torch.no_grad()
    def update_batch_flag(self, batch_pointer: int = None) -> None:
        if batch_pointer is not None:
            self._finished_batch = batch_pointer
        else:
            self._finished_batch = self._finished_batch + 1

    @property
    def cuda_available_row_num(self):
        return self._cuda_available_row_num

class CachedEmbeddingBag(torch.nn.Module):
    """GPU cached EmbeddingBag."""

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            max_norm: Optional[float] = None, 
            norm_type: float = 2., 
            scale_grad_by_freq: bool = False,
            mode: str = 'mean', 
            sparse: bool = False, 
            _weight: Optional[torch.Tensor] = None,
            include_last_offset: bool = False, 
            padding_idx: Optional[int] = None,
            device: torch.device = None, 
            dtype: torch.dtype = None,
            cached_ratio: float = 0.01,
            buffer_size: int = 50_000,
            pin_weight: bool = False
    ) -> None:
        # Factory Pytorch init code (_weight init code removed)
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.mode = mode
        self.sparse = sparse
        self.include_last_offset = include_last_offset

        # Initialize weight and GPU cache related things.
        assert cached_ratio <= 1.0, f'cache ratio {cached_ratio} should be smaller than 1.0' 
        cuda_row_num = int(num_embeddings * cached_ratio)
        if _weight is None:
            _weight = torch.empty(self.num_embeddings, self.embedding_dim, dtype=dtype, device=device)
            torch.nn.init.normal_(_weight)
            if self.padding_idx is not None:
                with torch.no_grad():
                    _weight[self.padding_idx].fill_(0)
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], 'Shape of weight does not match num_embeddings and embedding_dim'
        self.cache_weight_mgr = CacheManager(_weight, cuda_row_num, buffer_size, pin_weight)
        self.cache_op = True

    def set_cache_mgr_async_copy(self, flag):
        self.cache_weight_mgr._async_copy = flag
    
    def forward(self, input, offsets=None, per_sample_weights=None, shape_hook=None):
        if self.cache_op:
            with torch.no_grad():
                input = self.cache_weight_mgr.prepare_ids(input)
        
        embeddings = torch.nn.functional.embedding_bag(
            input.cuda(),
            self.cache_weight_mgr.cuda_cached_weight,
            offsets,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.mode,
            self.sparse,
            per_sample_weights,
            self.include_last_offset,
            self.padding_idx,
        )
        if shape_hook is not None:
            embeddings = shape_hook(embeddings)
        
        # Shall we update batch flag here?
        self.cache_weight_mgr.update_batch_flag()

        return embeddings
    
    @property
    def weight(self):
        return self.cache_weight_mgr.weight
    
    ############################# Perf Log ###################################

    @property
    def num_hits_history(self):
        return self.cache_weight_mgr.num_hits_history

    @property
    def num_miss_history(self):
        return self.cache_weight_mgr.num_miss_history

    @property
    def num_write_back_history(self):
        return self.cache_weight_mgr.num_write_back_history

    @property
    def swap_in_bandwidth(self):
        if self.cache_weight_mgr._cpu_to_cuda_numel > 0:
            return (
                self.cache_weight_mgr._cpu_to_cuda_numel
                * self.cache_weight_mgr.elem_size_in_byte
                / 1e6
                / self.cache_weight_mgr._cpu_to_cuda_elapse
            )
        else:
            return 0

    @property
    def swap_out_bandwidth(self):
        if self.cache_weight_mgr._cuda_to_cpu_numel > 0:
            return (
                self.cache_weight_mgr._cuda_to_cpu_numel
                * self.cache_weight_mgr.elem_size_in_byte
                / 1e6
                / self.cache_weight_mgr._cuda_to_cpu_elapse
            )
        return 0