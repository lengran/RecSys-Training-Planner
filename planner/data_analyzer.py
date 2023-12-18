import matplotlib.pyplot as plt
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
        for i in range(0, self.batch_num - 1):
            batch_indices = batches[i]
            self.batches.append(cat_data.iloc[batch_indices].stack().unique())
        # For batches[self.batch_num - 1], deal with -1s
        batch_indices = df.Series([i for i in batches[self.batch_num - 1].values.tolist() if i != -1])
        self.batches.append(cat_data.iloc[batch_indices].stack().unique())

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
        for i in range(batches.shape[1]):
            indices = batches[i].to_numpy()
            self.batches.append(df.concat([df.Series(rotated_cat_data[indices].flatten())]).unique())

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


def Generate_CDF(cats_count: list, dataset: str):
    cats_count.sort(reverse=True)

    plt.style.use('_mpl-gallery')

    fig, ax = plt.subplots(figsize=(5, 4),)
    plt.xscale('log')
    # plt.yscale('log')

    # ax.ecdf(cats_count, label="CDF")
    # ax.hist(cats_count, 1, density=True, histtype='step', cumulative=True, label="CDF")
    ax.plot(cats_count)

    plt.savefig(os.path.join(PLOT_PATH, dataset + ".png"), bbox_inches='tight')
    plt.close()

def Generate_combined_CDF(criteo_cats_count: list, taobao_cats_count: list):
    criteo_cats_count.sort(reverse=True)
    criteo_entry_count = list(range(len(criteo_cats_count)))
    taobao_cats_count.sort(reverse=True)
    taobao_entry_count = list(range(len(taobao_cats_count)))

    plt.style.use('_mpl-gallery')

    fig, ax = plt.subplots(figsize=(5, 2.2),)
    plt.xscale('log')
    ax.set_ylabel("Access Ratio")
    ax.set_xlabel("Entry Count")

    # plt.yscale('log')

    # ax.ecdf(cats_count, label="CDF")
    # ax.hist(cats_count, 1, density=True, histtype='step', cumulative=True, label="CDF")
    ax.plot(criteo_entry_count, criteo_cats_count, 'b', linestyle='--', label='Criteo dataset for DLRM')
    ax.plot(taobao_entry_count, taobao_cats_count, 'g', linestyle='-.', label='Taobao dataset for TBSM')
    ax.legend()

    plt.savefig(os.path.join(PLOT_PATH, "combined.pdf"), bbox_inches='tight')
    plt.close()
    

def count_access_rate(id: int, cat_data: np.ndarray):
    num_rows = cat_data.shape[0]

    print("[Count access of cat " + str(id) + " in dataset] Number of rows: " + str(num_rows) +". Counting.....")
    # access_count = [0] * cat_count
    # for i in range(num_rows):
    #     access_count[cat_data[i]] = access_count[cat_data[i]] + 1
    #     if (i - 1) % int(num_rows / 10) == 0:
    #         print("[Count access of cat " + str(id) + " in dataset] " + str(int((i - 1) / int(num_rows / 10))) + "0%. (" + str(i) + " rows)")
    data = df.Series(cat_data, dtype='int32')
    counted_data = data.value_counts()
    unordered_data = counted_data.index.to_arrow().to_pylist()
    unordered_count = counted_data.to_arrow().to_pylist()
    pair = list(zip(unordered_data, unordered_count))
    pair.sort(key=lambda x: x[0])
    count = list(map(lambda x: x[1], pair))
    
    print("[Count access of cat " + str(id) + " in dataset] Done!")

    return count
    # q.put(access_count)
    

if __name__ == "__main__":
    # plt.style.use('_mpl-gallery')
    
    # criteo =BatchLoader(
    #     dataset="criteo",
    #     plan_path=DLRM_PLAN_PATH, 
    #     data_path=DLRM_DATA_PATH, 
    #     )
    
    # taobao = BatchLoader(
    #     dataset="taobao",
    #     plan_path=TBSM_PLAN_PATH, 
    #     data_path=TBSM_DATA_PATH, 
    #     )
    
    # start_time = time.time()
    # heatmap = calculate_obj_overlap(criteo.batches)
    # end_time = time.time()
    # print("Calculation finished. (" + str(end_time - start_time) + "s)")

    # save heatmap
    # np_heatmap = np.array(heatmap)
    # np.save(os.path.join(TBSM_PLAN_PATH, "heatmap.npy"), np_heatmap)

    # print(np_heatmap)
    

    '''
    Prepare data for CDF figures
    '''
    criteo = DatasetLoader(
        dataset="criteo",
        data_path=DLRM_DATA_PATH, 
        )
    taobao = DatasetLoader(
        dataset="taobao",
        data_path=TBSM_DATA_PATH, 
        )
    
    criteo_cats_count = list()
    for i in range(criteo.num_cats):
        cat_count = count_access_rate(i, criteo.cat_data[i])
        criteo_cats_count = criteo_cats_count + cat_count

    taobao_cats_counts = list()
    for i in range(taobao.num_cats):
        cat_count = count_access_rate(i, taobao.cat_data[i])
        taobao_cats_counts = taobao_cats_counts + cat_count
    
    Generate_combined_CDF(criteo_cats_count, taobao_cats_counts)
    