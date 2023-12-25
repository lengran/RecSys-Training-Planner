import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cudf as df
import os
import time
from multiprocessing import Process, Pool, Queue

DLRM_DATA_PATH = "/root/files/coding/RecSys-Training-Planner/DLRM/input/kaggle/kaggleAdDisplayChallenge_processed.npz"
TBSM_DATA_PATH = "/root/files/coding/RecSys-Training-Planner/TBSM/output/taobao_train_t20.npz"
DLRM_PLAN_PATH = "/root/files/coding/RecSys-Training-Planner/DLRM/input/training_plan/"
TBSM_PLAN_PATH = "/root/files/coding/RecSys-Training-Planner/TBSM/input/training_plan/"
PLOT_PATH = "/root/files/coding/data_loading_planner/CDF"

class BatchLoader(object):
    def __init__(
            self, 
            dataset: str,
            plan_path: str, 
            data_path: str, 
            batchsize: int = None,
            ) -> None:
        '''
        args:
            dataset (str): Should be "criteo" or "taobao".
            plan_path (str): Directory to read batches from and write generated plan to.
            data_path (str): Path to the dataset file (a numpy xyz file).
        '''
        # Read the batches.
        if batchsize is None:
            batchsize_str = ""
        else:
            batchsize_str = "-" + str(batchsize)
        loading_begin = time.time()
        print("Loading data...")
        batches = df.read_parquet(os.path.join(os.path.abspath(plan_path), "batches" + batchsize_str + ".parquet")).transpose()             # each column is a batch of indices of dataset's row
        self.batch_num = batches.shape[1]
        self.batch_size = batches.shape[0]

        # Read dataset and process it.
        if dataset == "criteo":
            self.load_criteo_data(data_path, batches)
        else:
            self.load_taobao_data(data_path, batches)
        loading_end = time.time()
        print("Data loading finished. (" + str(loading_end - loading_begin) + "s)")

    def load_criteo_data(self, data_path, batches):
        """
        Returned data:
            self.batches (list of df.Series): Categorical objects contained in each batch. (This is basically all we need from the dataset.) 
        """
        # Read cat_data
        data = np.load(os.path.abspath(data_path))
        X_cat = data["X_cat"]                                                                                               # categorical feature
        
        # Count size of features.
        self.cat_counts = list(data["counts"])

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

class DatasetLoader(object):
    def __init__(
            self, 
            dataset: str,
            # plan_path: str, 
            data_path: str, 
            ) -> None:
        '''
        args:
            dataset (str): Should be "criteo" or "taobao".
            plan_path (str): Directory to read batches from and write generated plan to.
            data_path (str): Path to the dataset file (a numpy xyz file).
        '''
        # Read the batches.
        loading_begin = time.time()
        print("Loading data...")
        # batches = df.read_parquet(os.path.join(os.path.abspath(plan_path), "batches.parquet")).transpose()             # each column is a batch of indices of dataset's row
        # self.batch_num = batches.shape[1]
        # self.batch_size = batches.shape[0]

        # Read dataset and process it.
        if dataset == "criteo":
            self.load_criteo_data(data_path)
        else:
            self.load_taobao_data(data_path)
        loading_end = time.time()
        print("Data loading finished. (" + str(loading_end - loading_begin) + "s)")

    def load_criteo_data(self, data_path):
        """
        Returned data:
            self.cat_data
            self.cat_counts
            self.num_cats
            self.num_cat_objs
        """
        # Read cat_data
        data = np.load(os.path.abspath(data_path))
        X_cat = data["X_cat"]                                                                                               # categorical feature
        self.cat_data = X_cat.swapaxes(0, 1)
        
        # Count size of features.
        self.cat_counts = list(data["counts"])
        self.num_cats = self.cat_data.shape[0]
        self.num_cat_objs = 0
        for i in range(len(self.cat_counts)):
            self.num_cat_objs = self.num_cat_objs + self.cat_counts[i]

    def load_taobao_data(self, data_path):
        """
        Returned data:
            self.cat_data
            self.cat_counts
            self.num_cats
            self.num_cat_objs
        """
        # Read cat_data
        data = np.load(os.path.abspath(data_path))
        X_cat = data["X_cat"]                                                                                               # categorical feature
        self.cat_data = X_cat.reshape(X_cat.shape[0], X_cat.shape[1] * X_cat.shape[2])
        

        # Count size of features.
        self.cat_counts = [987994, 4162024, 9439]
        self.num_cats = 3

        # Convert id from id_in_cat to unique_id_in_dataset
        # cat_offsets = list()
        # tmp_offset = 0
        # for i in range(len(self.cat_counts)):
        #     cat_offsets.append(tmp_offset)
        #     tmp_offset = tmp_offset + self.cat_counts[i]
        # for i in range(len(cat_offsets)):
        #     X_cat[i] = X_cat[i] + cat_offsets[i]
        
        # Calculate objs in batch
        # rotated_cat_data = np.swapaxes(X_cat, 0, 1)
        # self.batches = list()
        # for i in range(batches.shape[1]):
        #     indices = batches[i].to_numpy()
        #     self.batches.append(df.concat([df.Series(rotated_cat_data[indices].flatten())]).unique())

"""
def calculate_obj_overlap(batches: list) -> list:
    '''
    Calculate number of overlapping objects between batches.
    Input: batches of objects
    Output: A 2 dimension matrix contains number of overlapping objects.
    '''
    print("Calculating...")
    overlapping_heatmap = list()
    num_batches = len(batches)

    # Calculate from the beginning to the end
    for i in range(num_batches):
        overlapping_heatmap.append([-1] * (i + 1))
        overlapping_heatmap[i][i] = len(batches[i])
        for j in range(i + 1, num_batches):
            common_objs = batches[i][batches[i].isin(batches[j]) == True]
            num_common_objs = len(common_objs)
            overlapping_heatmap[i].append(num_common_objs)

    # Fill in the mirrored cells in the heatmap
    for i in range(num_batches):
        for j in range(i):
            overlapping_heatmap[i][j] = overlapping_heatmap[j][i]
    
    return overlapping_heatmap
"""

def Calculate_hit_rate(plan: list, cached_rows: int, batches: list, freq_batches: list):
    '''
    Calculate cache hit rate of a given route.
    '''
    
    num_steps = len(plan)
    gpu_cache = df.Series([-1] * cached_rows, dtype=int)
    cost_total = 0
    start_step = 0
    num_available_rows = cached_rows
    hit_rate_track = list()
    
    # batch_register = df.Series([-1] * self.cached_rows, dtype='int32')
    freq_register = df.Series([-1] * cached_rows, dtype='int32')
    
    # the main loop
    for step in range(start_step, num_steps):
        if (step % int(num_steps / 10)) == 0:
            print("Progress: " + str(int((step + 1)  / (num_steps / 10))) + "0%...")
        # Get ids in current batch
        batch_ids = batches[plan[step]]
        freq_batch_ids = freq_batches[plan[step]]
        num_total_access = int(freq_batch_ids.sum())

        # Find ids that need to be transfered to cache
        ids_to_comm = batch_ids[batch_ids.isin(gpu_cache) == False]
        freq_ids_to_comm = freq_batch_ids[ids_to_comm.index]
        num_miss_access = int(freq_ids_to_comm.sum())
        hit_rate = (num_total_access - num_miss_access) / num_total_access
        hit_rate_track.append(hit_rate)
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

    return cost_total, hit_rate_track

def Generate_CDF_hitrate(data: list, output_path: str):
    cdf = sorted(data)
    # pdf=data/np.sum(data)
    # cdf = np.cumsum(pdf)
    # np.count_nonzero()

    plt.style.use('_mpl-gallery')

    fig, ax = plt.subplots(figsize=(4, 1.76),)
    plt.xscale('log')
    # plt.yscale('log')
    ax.set_ylabel("Cache hit rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("Mini-batch")

    # ax.ecdf(cats_count, label="CDF")
    # ax.hist(cats_count, 1, density=True, histtype='step', cumulative=True, label="CDF")
    ax.plot(cdf)

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def Generate_combined_CDF(criteo_cats_count: list, taobao_cats_count: list):
    criteo_cats_count.sort(reverse=True)
    criteo_entry_count = list(range(len(criteo_cats_count)))
    taobao_cats_count.sort(reverse=True)
    taobao_entry_count = list(range(len(taobao_cats_count)))

    plt.style.use('_mpl-gallery')

    fig, ax = plt.subplots(figsize=(6, 1.5),)
    plt.xscale('log')
    ax.set_ylabel("Access Ratio")
    ax.set_xlabel("Entry Count")

    # plt.yscale('log')

    # ax.ecdf(cats_count, label="CDF")
    # ax.hist(cats_count, 1, density=True, histtype='step', cumulative=True, label="CDF")
    ax.plot(criteo_entry_count, criteo_cats_count, 'b', linestyle='--', label='Criteo dataset for DLRM')
    ax.plot(taobao_entry_count, taobao_cats_count, 'g', linestyle='-.', label='Taobao dataset for TBSM')
    ax.legend()

    plt.savefig(os.path.join(PLOT_PATH, "skewness.pdf"), bbox_inches='tight')
    plt.close()
    
def count_access_rate(cat_data: np.ndarray):
    # access_count = [0] * cat_count
    # for i in range(num_rows):
    #     access_count[cat_data[i]] = access_count[cat_data[i]] + 1
    #     if (i - 1) % int(num_rows / 10) == 0:
    #         print("[Count access of cat " + str(id) + " in dataset] " + str(int((i - 1) / int(num_rows / 10))) + "0%. (" + str(i) + " rows)")
    # data = df.Series(cat_data, dtype='int32')
    # counted_data = data.value_counts()
    # unordered_data = counted_data.index.to_arrow().to_pylist()
    # unordered_count = counted_data.to_arrow().to_pylist()
    # pair = list(zip(unordered_data, unordered_count))
    # pair.sort(key=lambda x: x[0])
    # count = list(map(lambda x: x[1], pair))
    objs, obj_counts = np.unique(cat_data, return_counts=True)
    pair_type = [('obj', int), ('count', int)]
    import pdb; pdb.set_trace()
    tmp_pair = list(zip(objs, obj_counts))
    pair = np.array(tmp_pair, dtype=pair_type)
    sorted_pair = np.sort(pair, order='count')

    return 
    # q.put(access_count)
    
def calculate_obj_overlap(loader: BatchLoader, sample_rate: float = 1.0 ) -> np.ndarray:
    '''
    Calculate number of overlapping objects between batches.
    Input: batches of objects
    Output: A 2 dimension matrix contains number of overlapping objects.
    '''
    
    sample_unit = int(1 / sample_rate)
    idx_batch = [i for i in range(len(loader.batches)) if i % sample_unit == 0]
    num_batches = len(idx_batch)
    overlapping_heatmap = np.empty((num_batches, num_batches))
    print(str(num_batches) + " sampled batches.\nExtracting data...")

    batches_id = list()
    batches_freq = list()
    for i in range(num_batches):
        idx = idx_batch[i]
        batch_id = loader.batches[idx].to_numpy()
        batch_freq = loader.freq_batches[idx].to_numpy()
        batches_id.append(batch_id)
        batches_freq.append(batch_freq)
    
    print("Calculating....")

    # Calculate from the beginning to the end
    for i in range(num_batches):
        if i > 0:
            # overlapping_heatmap[i][:i] = overlapping_heatmap[:i][i]
            for j in range(i):
                overlapping_heatmap[i][j] = overlapping_heatmap[j][i]
        overlapping_heatmap[i][i] = batches_freq[i].sum()
        for j in range(i + 1, num_batches):
            # common_objs = batches_id[i][batches_id[i].isin(batches_id[j]) == True]
            mask_common_objs = np.isin(batches_id[i], batches_id[j], assume_unique=True)
            num_common_objs = batches_freq[i][mask_common_objs].sum()
            overlapping_heatmap[i][j] = num_common_objs
        
        if (i % int(num_batches / 10)) == 0:
            print("0." + str(int(i / (num_batches / 10))) + " calculated...")
    
    return overlapping_heatmap

def Filter_heatmap(data: np.ndarray):
    avg = np.average(np.average(data))
    for i in range(data.shape[0]):
        data[i][i] = avg
    avg = np.average(np.average(data))
    for i in range(data.shape[0]):
        data[i][i] = np.nan# avg
    
    return data.shape[0]

def Generate_heatmap(data: np.ndarray):    
    # plt.style.use('_mpl-gallery')
    fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(data)
    # im, cbar = draw_heatmap(data, ax=ax, cmap="YlGn", cbarlabel="Overlapping")
    # texts = annotate_heatmap(im, valfmt="{x:.1f} t")

    cbar = ax.figure.colorbar(im, ax=ax,)
    cbar.ax.set_ylabel("Percentage of overlapped objects in sampled mini-batches (%)", rotation=-90, va="bottom")

    plt.savefig(os.path.join(PLOT_PATH, "heatmap.pdf"), bbox_inches='tight')

def Generate_histogram(data: np.ndarray, num_bins: int, output_path: str, num_elemets: int):
    max_count = num_elemets * num_elemets - num_elemets
    # data = data / max_count

    # plot
    plt.style.use('_mpl-gallery')
    fig, ax = plt.subplots(figsize=(4, 1.76))

    # ax.set_ylabel("Occurance")
    ax.set_ylabel("Percentage of samples (%)")
    ax.set_xlabel("Percentage of overlapping (%)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=max_count))

    ax.hist(data, num_bins)
    plt.savefig(output_path, bbox_inches='tight')

if __name__ == "__main__":
    '''
    Heatmap
    '''
    
    # log_path = "/root/files/coding/data_loading_planner/kaggle_run_0"
    log_path = "/root/files/coding/data_loading_planner/taobao_run"
    # criteo =BatchLoader(
    #     dataset="criteo",
    #     plan_path=DLRM_PLAN_PATH, 
    #     data_path=DLRM_DATA_PATH, 
    #     batchsize=1024,
    #     )
    
    # taobao = BatchLoader(
    #     dataset="taobao",
    #     plan_path=TBSM_PLAN_PATH, 
    #     data_path=TBSM_DATA_PATH,
    #     batchsize=1024,
    #     )
    
    # start_time = time.time()
    # heatmap = calculate_obj_overlap(criteo, 0.01)
    # end_time = time.time()
    # print("Calculation finished. (" + str(end_time - start_time) + "s)")

    # # save heatmap
    # np.save(os.path.join(log_path, "heatmap"), heatmap)

    # print("Loading map data...")
    # heatmap = np.load(os.path.join(log_path, "heatmap.npy"))
    # print("Processing...")
    # full_overlapping = np.max(heatmap[0])
    # minimum_overlapping = np.min(heatmap)
    # heatmap = heatmap / full_overlapping * 100
    # num_elements = Filter_heatmap(heatmap)
    # print("Max(overlapping with itself): " + str(full_overlapping) + ", min: " + str(minimum_overlapping) + "\nDrawing...")
    # # Generate_heatmap(heatmap)
    # Generate_histogram(heatmap.flatten(), 100, os.path.join(PLOT_PATH, "heatmap_plot.pdf"), num_elements)
    # print("Done!")

    '''
    CDF figures
    '''
    log_path_criteo = "/root/files/coding/data_loading_planner/kaggle_run_0"
    log_path_taobao = "/root/files/coding/data_loading_planner/taobao_run"

    # criteo = DatasetLoader(
    #     dataset="criteo",
    #     data_path=DLRM_DATA_PATH, 
    #     )
    # taobao = DatasetLoader(
    #     dataset="taobao",
    #     data_path=TBSM_DATA_PATH, 
    #     )
    
    # criteo_cats_count = list()
    # for i in range(criteo.num_cats):
    #     print("[Count access of cat " + str(i) + " in criteo dataset] Number of rows: " + str(criteo.cat_data[i].shape[0]) +". Counting.....")
    #     # cat_count = count_access_rate(criteo.cat_data[i])
    #     _, cat_count = np.unique(criteo.cat_data[i], return_counts=True)
    #     criteo_cats_count = criteo_cats_count + cat_count.tolist()
    #     print("[Count access of cat " + str(i) + " in dataset] Done!")

    # taobao_cats_counts = list()
    # for i in range(taobao.num_cats):
    #     print("[Count access of cat " + str(i) + " in taobao dataset] Number of rows: " + str(taobao.cat_data[i].shape[0]) +". Counting.....")
    #     # cat_count = count_access_rate(taobao.cat_data[i])
    #     _, cat_count = np.unique(taobao.cat_data[i], return_counts=True)
    #     taobao_cats_counts = taobao_cats_counts + cat_count.tolist()
    #     print("[Count access of cat " + str(i) + " in dataset] Done!")
    
    # np.save(os.path.join(log_path_criteo, "cats_counts"), criteo_cats_count)
    # np.save(os.path.join(log_path_taobao, "cats_counts"), taobao_cats_counts)
    
    criteo_cats_count = np.load(os.path.join(log_path_criteo, "cats_counts.npy")).astype(np.int32).tolist()
    taobao_cats_counts = np.load(os.path.join(log_path_taobao, "cats_counts.npy")).astype(np.int32).tolist()
    print("Drawing...")
    Generate_combined_CDF(criteo_cats_count, taobao_cats_counts)
    print("Done!")
    
    '''
    Hit rate on 2 neighbouring batchse
    '''
    # criteo = BatchLoader(
    #     dataset="criteo",
    #     plan_path=DLRM_PLAN_PATH, 
    #     data_path=DLRM_DATA_PATH, 
    #     batchsize=256
    #     )

    # taobao = BatchLoader(
    #     dataset="taobao",
    #     plan_path=TBSM_PLAN_PATH, 
    #     data_path=TBSM_DATA_PATH, 
    #     batchsize=1024
    #     )

    log_path = "/root/files/coding/data_loading_planner/taobao_run"
    
    # plan = list(range(len(taobao.batches)))
    # plan = [i for i in range(len(taobao.batches)) if i % 10 == 0]
    
    # print("Calculating hit rates.")
    # cost, hit_rate_track = Calculate_hit_rate(plan, int(33762577 * 0.01), taobao.batches, taobao.freq_batches)
    # # print("Done! cost: " + str(cost))
    # np.save(os.path.join(log_path, "hitrate-0.01-1024-0.1sr"), hit_rate_track)

    # data = np.load(os.path.join(log_path, "hitrate-0.01-1024-0.1sr.npy"))
    # data = data / 0.567486029778531
    # data = np.clip(data, 0, 1)
    # hit_rate_track = data.tolist()

    # print("Drawing...")
    # Generate_CDF_hitrate(hit_rate_track, os.path.join(PLOT_PATH, "hitrate.png"))
    # print("Done!")


