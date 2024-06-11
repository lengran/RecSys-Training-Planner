# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: generate inputs and targets for the dlrm benchmark
# The inpts and outputs are generated according to the following three option(s)
# 1) random distribution
# 2) synthetic distribution, based on unique accesses and distances between them
#    i) R. Hassan, A. Harris, N. Topham and A. Efthymiou "Synthetic Trace-Driven
#    Simulation of Cache Memory", IEEE AINAM'07
# 3) public data set
#    i)  Criteo Kaggle Display Advertising Challenge Dataset
#    https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset
#    ii) Criteo Terabyte Dataset
#    https://labs.criteo.com/2013/12/download-terabyte-click-logs


from __future__ import absolute_import, division, print_function, unicode_literals

# others
from os import path
import bisect
import collections

import data_utils_avazu as data_utils

# numpy
import numpy as np
from numpy import random as ra


# pytorch
import torch
from torch.utils.data import Dataset, RandomSampler

import data_loader_terabyte


# Kaggle Display Advertising Challenge Dataset
# dataset (str): name of dataset (Kaggle or Terabyte)
# randomize (str): determines randomization scheme
#            "none": no randomization
#            "day": randomizes each day"s data (only works if split = True)
#            "total": randomizes total dataset
# split (bool) : to split into train, test, validation data-sets
class AvazuDataset(Dataset):

    def __init__(
            self,
            dataset,
            max_ind_range,
            sub_sample_rate,
            randomize,
            split="train",
            raw_path="",
            pro_data="",
            memory_map=False,
            dataset_multiprocessing=False
    ):
        # dataset
        # tar_fea = 1   # single target
        den_fea = 2  # 2 dense  features: date and time
        # spa_fea = 22  # 22 sparse features: all features except time(date + time) and click
        # tad_fea = tar_fea + den_fea
        # tot_fea = tad_fea + spa_fea
        out_file = "avazu_processed"
        self.max_ind_range = max_ind_range
        self.memory_map = memory_map

        # split the datafile into path and filename
        lstr = raw_path.split("/")
        self.d_path = "/".join(lstr[0:-1]) + "/"
        self.d_file = lstr[-1].split(".")[0]
        self.npzfile = self.d_path + self.d_file
        self.trafile = self.d_path + self.d_file + "_fea"

        # check if pre-processed data is available
        data_ready = True
        if memory_map:
            reo_data = self.npzfile + "_reordered.npz"
            if not path.exists(str(reo_data)):
                data_ready = False
        else:
            if not path.exists(str(pro_data)):
                data_ready = False

        # pre-process data if needed
        # WARNNING: when memory mapping is used we get a collection of files
        if data_ready:
            print("Reading pre-processed data=%s" % (str(pro_data)))
            file = str(pro_data)
        else:
            print("Reading raw data=%s" % (str(raw_path)))
            # file = data_utils.getAvazuData(
            #     raw_path,
            #     out_file,
            #     max_ind_range,
            #     sub_sample_rate,
            #     days,
            #     split,
            #     randomize,
            #     dataset == "kaggle",
            #     memory_map,
            #     dataset_multiprocessing
            # )
            file = data_utils.getAvazuData(raw_path)

        # get a number of samples per day
        total_file = self.d_path + self.d_file + "_count.npz"
        with np.load(total_file) as data:
            total_count = data["total_count"]
        # compute offsets per file
        # self.offset_per_file = np.array([0] + [x for x in total_per_file])
        # for i in range(days):
            # self.offset_per_file[i + 1] += self.offset_per_file[i]
        # print(self.offset_per_file)
        self.total_count = total_count
        print("Total_count: " + str(total_count))

        # setup data
        if memory_map:
            # setup the training/testing split
            self.split = split
            if split == 'none' or split == 'train':
                # self.day = 0
                # self.max_day_range = days if split == 'none' else days - 1
                self.train_size = int(7/8 * total_count)
            elif split == 'test' or split == 'val':
                # self.day = days - 1
                # num_samples = self.offset_per_file[days] - \
                            #   self.offset_per_file[days - 1]
                # self.test_size = int(np.ceil(num_samples / 2.))
                self.test_size = int((total_count - int(7/8 * total_count)) / 2)
                self.val_size = total_count - int(7/8 * total_count) - self.test_size
            else:
                sys.exit("ERROR: dataset split is neither none, nor train or test.")

            '''
            # text
            print("text")
            for i in range(days):
                fi = self.npzfile + "_{0}".format(i)
                with open(fi) as data:
                    ttt = 0; nnn = 0
                    for _j, line in enumerate(data):
                        ttt +=1
                        if np.int32(line[0]) > 0:
                            nnn +=1
                    print("day=" + str(i) + " total=" + str(ttt) + " non-zeros="
                          + str(nnn) + " ratio=" +str((nnn * 100.) / ttt) + "%")
            # processed
            print("processed")
            for i in range(days):
                fi = self.npzfile + "_{0}_processed.npz".format(i)
                with np.load(fi) as data:
                    yyy = data["y"]
                ttt = len(yyy)
                nnn = np.count_nonzero(yyy)
                print("day=" + str(i) + " total=" + str(ttt) + " non-zeros="
                      + str(nnn) + " ratio=" +str((nnn * 100.) / ttt) + "%")
            # reordered
            print("reordered")
            for i in range(days):
                fi = self.npzfile + "_{0}_reordered.npz".format(i)
                with np.load(fi) as data:
                    yyy = data["y"]
                ttt = len(yyy)
                nnn = np.count_nonzero(yyy)
                print("day=" + str(i) + " total=" + str(ttt) + " non-zeros="
                      + str(nnn) + " ratio=" +str((nnn * 100.) / ttt) + "%")
            '''

            # load unique counts
            with np.load(self.d_path + self.d_file + "_fea_count.npz") as data:
                self.counts = data["counts"]                                                    # number of different values in a feature
            self.m_den = den_fea  # X_int.shape[1]                                              # number of dense features
            self.n_emb = len(self.counts)                                                       # number of categorical features
            print("Sparse features= %d, Dense features= %d" % (self.n_emb, self.m_den))

            # Load the test data
            # Only a single day is used for testing
            if self.split == 'test' or self.split == 'val':
                # only a single day is used for testing
                fi = self.npzfile + ".npz" #"_reordered.npz"
                with np.load(fi) as data:
                    self.X_int = data["X_int"]  # continuous  feature
                    self.X_cat = data["X_cat"]  # categorical feature
                    self.y = data["y"]          # target

        else:
            print("DEBUG: load and preprocess data")
            # load and preprocess data
            with np.load(file) as data:
                X_int = data["X_int"]  # continuous  feature
                X_cat = data["X_cat"]  # categorical feature
                y = data["y"]          # target
            with np.load(self.d_path + self.d_file + "_fea_count.npz") as data:
                self.counts = data["counts"]
            self.m_den = X_int.shape[1]  # den_fea
            self.n_emb = len(self.counts)
            print("Sparse fea = %d, Dense fea = %d" % (self.n_emb, self.m_den))

            # create reordering
            indices = np.arange(len(y))

            if split == "none":
                # randomize all data
                if randomize == "total":
                    indices = np.random.permutation(indices)
                    print("Randomized indices...")

                X_int[indices] = X_int
                X_cat[indices] = X_cat
                y[indices] = y

            else:
                # indices = np.array_split(indices, self.offset_per_file[1:-1])

                # # randomize train data (per day)
                # if randomize == "day":  # or randomize == "total":
                #     for i in range(len(indices) - 1):
                #         indices[i] = np.random.permutation(indices[i])
                #     print("Randomized indices per day ...")

                # train_indices = np.concatenate(indices[:-1])
                # test_indices = indices[-1]
                # test_indices, val_indices = np.array_split(test_indices, 2)
                train_indices = int(7/8 * total_count)
                test_indices = int((total_count - train_indices) / 2)
                val_indices = np.arange(train_indices + test_indices, total_count)
                test_indices = np.arange(train_indices, train_indices + test_indices)
                train_indices = np.arange(train_indices)

                print("Defined %s indices..." % (split))

                # randomize train data (across days)
                if randomize == "total":
                    train_indices = np.random.permutation(train_indices)
                    print("Randomized indices ...")

                # create training, validation, and test sets
                if split == 'train':
                    self.X_int = [X_int[i] for i in train_indices]
                    self.X_cat = [X_cat[i] for i in train_indices]
                    self.y = [y[i] for i in train_indices]
                elif split == 'val':
                    self.X_int = [X_int[i] for i in val_indices]
                    self.X_cat = [X_cat[i] for i in val_indices]
                    self.y = [y[i] for i in val_indices]
                elif split == 'test':
                    self.X_int = [X_int[i] for i in test_indices]
                    self.X_cat = [X_cat[i] for i in test_indices]
                    self.y = [y[i] for i in test_indices]

            print("Split data according to indices...")

    def __getitem__(self, index):

        if isinstance(index, slice):
            return [
                self[idx] for idx in range(
                    index.start or 0, index.stop or len(self), index.step or 1
                )
            ]

        # if self.memory_map:
        #     if self.split == 'none' or self.split == 'train':
        #         # check if need to swicth to next day and load data
        #         if index == self.offset_per_file[self.day]:
        #             # print("day_boundary switch", index)
        #             self.day_boundary = self.offset_per_file[self.day]
        #             fi = self.npzfile + "_{0}_reordered.npz".format(
        #                 self.day
        #             )
        #             # print('Loading file: ', fi)
        #             with np.load(fi) as data:
        #                 self.X_int = data["X_int"]  # continuous  feature
        #                 self.X_cat = data["X_cat"]  # categorical feature
        #                 self.y = data["y"]          # target
        #             self.day = (self.day + 1) % self.max_day_range

        #         i = index - self.day_boundary
        #     elif self.split == 'test' or self.split == 'val':
        #         # only a single day is used for testing
        #         i = index + (0 if self.split == 'test' else self.test_size)
        #     else:
        #         sys.exit("ERROR: dataset split is neither none, nor train or test.")
        # else:
        i = index

        if self.max_ind_range > 0:
            return self.X_int[i], self.X_cat[i] % self.max_ind_range, self.y[i]
        else:
            return self.X_int[i], self.X_cat[i], self.y[i]

    # def _default_preprocess(self, X_int, X_cat, y):                                                                   # What is this for?
    #     X_int = torch.log(torch.tensor(X_int, dtype=torch.float) + 1)
    #     if self.max_ind_range > 0:
    #         X_cat = torch.tensor(X_cat % self.max_ind_range, dtype=torch.long)
    #     else:
    #         X_cat = torch.tensor(X_cat, dtype=torch.long)
    #     y = torch.tensor(y.astype(np.float32))

    #     return X_int, X_cat, y

    def __len__(self):
        if self.memory_map:
            if self.split == 'none':
                # return self.offset_per_file[-1]
                return self.total_count
            elif self.split == 'train':
                # return self.offset_per_file[-2]
                return self.train_size
            elif self.split == 'test':
                return self.test_size
            elif self.split == 'val':
                return self.val_size
            else:
                sys.exit("ERROR: dataset split is neither none, nor train nor test.")
        else:
            return len(self.y)


def collate_wrapper_avazu(list_of_tuples):
    # where each tuple is (X_int, X_cat, y)
    transposed_data = list(zip(*list_of_tuples))
    X_int = torch.log(torch.tensor(transposed_data[0], dtype=torch.float) + 1)
    X_cat = torch.tensor(transposed_data[1], dtype=torch.long)
    T = torch.tensor(transposed_data[2], dtype=torch.float32).view(-1, 1)

    batchSize = X_cat.shape[0]
    featureCnt = X_cat.shape[1]

    lS_i = [X_cat[:, i] for i in range(featureCnt)]
    lS_o = [torch.tensor(range(batchSize)) for _ in range(featureCnt)]

    return X_int, torch.stack(lS_o), torch.stack(lS_i), T


def ensure_dataset_preprocessed(args, d_path):
    _ = AvazuDataset(
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

    _ = AvazuDataset(
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

    for split in ['train', 'val', 'test']:
        print('Running preprocessing for split =', split)

        train_files = ['{}_{}_reordered.npz'.format(args.raw_data_file, day)
                       for
                       day in range(0, 23)]

        test_valid_file = args.raw_data_file + '_23_reordered.npz'

        output_file = d_path + '_{}.bin'.format(split)

        input_files = train_files if split == 'train' else [test_valid_file]
        data_loader_terabyte.numpy_to_binary(input_files=input_files,
                                             output_file_path=output_file,
                                             split=split)


def make_avazu_data_and_loaders(args):

    # if args.mlperf_logging and args.memory_map and args.data_set == "terabyte":
    #     # more efficient for larger batches
    #     data_directory = path.dirname(args.raw_data_file)

    #     if args.mlperf_bin_loader:
    #         lstr = args.processed_data_file.split("/")
    #         d_path = "/".join(lstr[0:-1]) + "/" + lstr[-1].split(".")[0]
    #         train_file = d_path + "_train.bin"
    #         test_file = d_path + "_test.bin"
    #         # val_file = d_path + "_val.bin"
    #         counts_file = args.raw_data_file + '_fea_count.npz'

    #         if any(not path.exists(p) for p in [train_file,
    #                                             test_file,
    #                                             counts_file]):
    #             ensure_dataset_preprocessed(args, d_path)

    #         train_data = data_loader_terabyte.CriteoBinDataset(
    #             data_file=train_file,
    #             counts_file=counts_file,
    #             batch_size=args.mini_batch_size,
    #             max_ind_range=args.max_ind_range
    #         )

    #         train_loader = torch.utils.data.DataLoader(
    #             train_data,
    #             batch_size=None,
    #             batch_sampler=None,
    #             shuffle=False,
    #             num_workers=0,
    #             collate_fn=None,
    #             pin_memory=False,
    #             drop_last=False,
    #             sampler=RandomSampler(train_data) if args.mlperf_bin_shuffle else None
    #         )

    #         test_data = data_loader_terabyte.CriteoBinDataset(
    #             data_file=test_file,
    #             counts_file=counts_file,
    #             batch_size=args.test_mini_batch_size,
    #             max_ind_range=args.max_ind_range
    #         )

    #         test_loader = torch.utils.data.DataLoader(
    #             test_data,
    #             batch_size=None,
    #             batch_sampler=None,
    #             shuffle=False,
    #             num_workers=0,
    #             collate_fn=None,
    #             pin_memory=False,
    #             drop_last=False,
    #         )
    #     else:
    #         data_filename = args.raw_data_file.split("/")[-1]

    #         train_data = CriteoDataset(
    #             args.data_set,
    #             args.max_ind_range,
    #             args.data_sub_sample_rate,
    #             args.data_randomize,
    #             "train",
    #             args.raw_data_file,
    #             args.processed_data_file,
    #             args.memory_map,
    #             args.dataset_multiprocessing
    #         )

    #         test_data = CriteoDataset(
    #             args.data_set,
    #             args.max_ind_range,
    #             args.data_sub_sample_rate,
    #             args.data_randomize,
    #             "test",
    #             args.raw_data_file,
    #             args.processed_data_file,
    #             args.memory_map,
    #             args.dataset_multiprocessing
    #         )

    #         train_loader = data_loader_terabyte.DataLoader(
    #             data_directory=data_directory,
    #             data_filename=data_filename,
    #             days=list(range(23)),
    #             batch_size=args.mini_batch_size,
    #             max_ind_range=args.max_ind_range,
    #             split="train"
    #         )

    #         test_loader = data_loader_terabyte.DataLoader(
    #             data_directory=data_directory,
    #             data_filename=data_filename,
    #             days=[23],
    #             batch_size=args.test_mini_batch_size,
    #             max_ind_range=args.max_ind_range,
    #             split="test"
    #         )
    # else:
    if True:
        train_data = AvazuDataset(
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

        test_data = AvazuDataset(
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

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.mini_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_wrapper_avazu,
            pin_memory=False,
            drop_last=False,  # True
        )

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.test_mini_batch_size,
            shuffle=False,
            num_workers=args.test_num_workers,
            collate_fn=collate_wrapper_avazu,
            pin_memory=False,
            drop_last=False,  # True
        )

    return train_data, train_loader, test_data, test_loader

def load_avazu_test_data_and_loaders(args):

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
    test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.test_mini_batch_size,
            shuffle=False,
            num_workers=args.test_num_workers,
            collate_fn=collate_wrapper_avazu,
            pin_memory=False,
            drop_last=False,  # True
        )

    return test_loader

def load_avazu_preprocessed_data_and_loaders(args, train_hot, train_normal):

    train_hot_loader = torch.utils.data.DataLoader(
        train_hot,
        batch_size=args.mini_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_avazu,
        pin_memory=False,
        drop_last=False,  # True
    )

    train_normal_loader = torch.utils.data.DataLoader(
        train_normal,
        batch_size=args.mini_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_avazu,
        pin_memory=False,
        drop_last=False,  # True
    )

    return train_hot_loader, train_normal_loader

def generate_random_output_batch(n, num_targets, round_targets=False):
    # target (probability of a click)
    if round_targets:
        P = np.round(ra.rand(n, num_targets).astype(np.float32)).astype(np.float32)
    else:
        P = ra.rand(n, num_targets).astype(np.float32)

    return torch.tensor(P)


# uniform ditribution (input data)
def generate_uniform_input_batch(
    m_den,
    ln_emb,
    n,
    num_indices_per_lookup,
    num_indices_per_lookup_fixed,
):
    # dense feature
    Xt = torch.tensor(ra.rand(n, m_den).astype(np.float32))

    # sparse feature (sparse indices)
    lS_emb_offsets = []
    lS_emb_indices = []
    # for each embedding generate a list of n lookups,
    # where each lookup is composed of multiple sparse indices
    for size in ln_emb:
        lS_batch_offsets = []
        lS_batch_indices = []
        offset = 0
        for _ in range(n):
            # num of sparse indices to be used per embedding (between
            if num_indices_per_lookup_fixed:
                sparse_group_size = np.int64(num_indices_per_lookup)
            else:
                # random between [1,num_indices_per_lookup])
                r = ra.random(1)
                sparse_group_size = np.int64(
                    np.round(max([1.0], r * min(size, num_indices_per_lookup)))
                )
            # sparse indices to be used per embedding
            r = ra.random(sparse_group_size)
            sparse_group = np.unique(np.round(r * (size - 1)).astype(np.int64))
            # reset sparse_group_size in case some index duplicates were removed
            sparse_group_size = np.int64(sparse_group.size)
            # store lengths and indices
            lS_batch_offsets += [offset]
            lS_batch_indices += sparse_group.tolist()
            # update offset for next iteration
            offset += sparse_group_size
        lS_emb_offsets.append(torch.tensor(lS_batch_offsets))
        lS_emb_indices.append(torch.tensor(lS_batch_indices))

    return (Xt, lS_emb_offsets, lS_emb_indices)


# synthetic distribution (input data)
def generate_synthetic_input_batch(
    m_den,
    ln_emb,
    n,
    num_indices_per_lookup,
    num_indices_per_lookup_fixed,
    trace_file,
    enable_padding=False,
):
    # dense feature
    Xt = torch.tensor(ra.rand(n, m_den).astype(np.float32))

    # sparse feature (sparse indices)
    lS_emb_offsets = []
    lS_emb_indices = []
    # for each embedding generate a list of n lookups,
    # where each lookup is composed of multiple sparse indices
    for i, size in enumerate(ln_emb):
        lS_batch_offsets = []
        lS_batch_indices = []
        offset = 0
        for _ in range(n):
            # num of sparse indices to be used per embedding (between
            if num_indices_per_lookup_fixed:
                sparse_group_size = np.int64(num_indices_per_lookup)
            else:
                # random between [1,num_indices_per_lookup])
                r = ra.random(1)
                sparse_group_size = np.int64(
                    max(1, np.round(r * min(size, num_indices_per_lookup))[0])
                )
            # sparse indices to be used per embedding
            file_path = trace_file
            line_accesses, list_sd, cumm_sd = read_dist_from_file(
                file_path.replace("j", str(i))
            )
            # debug prints
            # print("input")
            # print(line_accesses); print(list_sd); print(cumm_sd);
            # print(sparse_group_size)
            # approach 1: rand
            # r = trace_generate_rand(
            #     line_accesses, list_sd, cumm_sd, sparse_group_size, enable_padding
            # )
            # approach 2: lru
            r = trace_generate_lru(
                line_accesses, list_sd, cumm_sd, sparse_group_size, enable_padding
            )
            # WARNING: if the distribution in the file is not consistent
            # with embedding table dimensions, below mod guards against out
            # of range access
            sparse_group = np.unique(r).astype(np.int64)
            minsg = np.min(sparse_group)
            maxsg = np.max(sparse_group)
            if (minsg < 0) or (size <= maxsg):
                print(
                    "WARNING: distribution is inconsistent with embedding "
                    + "table size (using mod to recover and continue)"
                )
                sparse_group = np.mod(sparse_group, size).astype(np.int64)
            # sparse_group = np.unique(np.array(np.mod(r, size-1)).astype(np.int64))
            # reset sparse_group_size in case some index duplicates were removed
            sparse_group_size = np.int64(sparse_group.size)
            # store lengths and indices
            lS_batch_offsets += [offset]
            lS_batch_indices += sparse_group.tolist()
            # update offset for next iteration
            offset += sparse_group_size
        lS_emb_offsets.append(torch.tensor(lS_batch_offsets))
        lS_emb_indices.append(torch.tensor(lS_batch_indices))

    return (Xt, lS_emb_offsets, lS_emb_indices)


def generate_stack_distance(cumm_val, cumm_dist, max_i, i, enable_padding=False):
    u = ra.rand(1)
    if i < max_i:
        # only generate stack distances up to the number of new references seen so far
        j = bisect.bisect(cumm_val, i) - 1
        fi = cumm_dist[j]
        u *= fi  # shrink distribution support to exclude last values
    elif enable_padding:
        # WARNING: disable generation of new references (once all have been seen)
        fi = cumm_dist[0]
        u = (1.0 - fi) * u + fi  # remap distribution support to exclude first value

    for (j, f) in enumerate(cumm_dist):
        if u <= f:
            return cumm_val[j]


# WARNING: global define, must be consistent across all synthetic functions
cache_line_size = 1


def trace_generate_lru(
    line_accesses, list_sd, cumm_sd, out_trace_len, enable_padding=False
):
    max_sd = list_sd[-1]
    l = len(line_accesses)
    i = 0
    ztrace = []
    for _ in range(out_trace_len):
        sd = generate_stack_distance(list_sd, cumm_sd, max_sd, i, enable_padding)
        mem_ref_within_line = 0  # floor(ra.rand(1)*cache_line_size) #0

        # generate memory reference
        if sd == 0:  # new reference #
            line_ref = line_accesses.pop(0)
            line_accesses.append(line_ref)
            mem_ref = np.uint64(line_ref * cache_line_size + mem_ref_within_line)
            i += 1
        else:  # existing reference #
            line_ref = line_accesses[l - sd]
            mem_ref = np.uint64(line_ref * cache_line_size + mem_ref_within_line)
            line_accesses.pop(l - sd)
            line_accesses.append(line_ref)
        # save generated memory reference
        ztrace.append(mem_ref)

    return ztrace


def trace_generate_rand(
    line_accesses, list_sd, cumm_sd, out_trace_len, enable_padding=False
):
    max_sd = list_sd[-1]
    l = len(line_accesses)  # !!!Unique,
    i = 0
    ztrace = []
    for _ in range(out_trace_len):
        sd = generate_stack_distance(list_sd, cumm_sd, max_sd, i, enable_padding)
        mem_ref_within_line = 0  # floor(ra.rand(1)*cache_line_size) #0
        # generate memory reference
        if sd == 0:  # new reference #
            line_ref = line_accesses.pop(0)
            line_accesses.append(line_ref)
            mem_ref = np.uint64(line_ref * cache_line_size + mem_ref_within_line)
            i += 1
        else:  # existing reference #
            line_ref = line_accesses[l - sd]
            mem_ref = np.uint64(line_ref * cache_line_size + mem_ref_within_line)
        ztrace.append(mem_ref)

    return ztrace


def trace_profile(trace, enable_padding=False):
    # number of elements in the array (assuming 1D)
    # n = trace.size

    rstack = []  # S
    stack_distances = []  # SDS
    line_accesses = []  # L
    for x in trace:
        r = np.uint64(x / cache_line_size)
        l = len(rstack)
        try:  # found #
            i = rstack.index(r)
            # WARNING: I believe below is the correct depth in terms of meaning of the
            #          algorithm, but that is not what seems to be in the paper alg.
            #          -1 can be subtracted if we defined the distance between
            #          consecutive accesses (e.g. r, r) as 0 rather than 1.
            sd = l - i  # - 1
            # push r to the end of stack_distances
            stack_distances.insert(0, sd)
            # remove r from its position and insert to the top of stack
            rstack.pop(i)  # rstack.remove(r)
            rstack.insert(l - 1, r)
        except ValueError:  # not found #
            sd = 0  # -1
            # push r to the end of stack_distances/line_accesses
            stack_distances.insert(0, sd)
            line_accesses.insert(0, r)
            # push r to the top of stack
            rstack.insert(l, r)

    if enable_padding:
        # WARNING: notice that as the ratio between the number of samples (l)
        # and cardinality (c) of a sample increases the probability of
        # generating a sample gets smaller and smaller because there are
        # few new samples compared to repeated samples. This means that for a
        # long trace with relatively small cardinality it will take longer to
        # generate all new samples and therefore obtain full distribution support
        # and hence it takes longer for distribution to resemble the original.
        # Therefore, we may pad the number of new samples to be on par with
        # average number of samples l/c artificially.
        l = len(stack_distances)
        c = max(stack_distances)
        padding = int(np.ceil(l / c))
        stack_distances = stack_distances + [0] * padding

    return (rstack, stack_distances, line_accesses)


# auxiliary read/write routines
def read_trace_from_file(file_path):
    try:
        with open(file_path) as f:
            if args.trace_file_binary_type:
                array = np.fromfile(f, dtype=np.uint64)
                trace = array.astype(np.uint64).tolist()
            else:
                line = f.readline()
                trace = list(map(lambda x: np.uint64(x), line.split(", ")))
            return trace
    except Exception:
        print("ERROR: no input trace file has been provided")


def write_trace_to_file(file_path, trace):
    try:
        if args.trace_file_binary_type:
            with open(file_path, "wb+") as f:
                np.array(trace).astype(np.uint64).tofile(f)
        else:
            with open(file_path, "w+") as f:
                s = str(trace)
                f.write(s[1 : len(s) - 1])
    except Exception:
        print("ERROR: no output trace file has been provided")


def read_dist_from_file(file_path):
    try:
        with open(file_path, "r") as f:
            lines = f.read().splitlines()
    except Exception:
        print("Wrong file or file path")
    # read unique accesses
    unique_accesses = [int(el) for el in lines[0].split(", ")]
    # read cumulative distribution (elements are passed as two separate lists)
    list_sd = [int(el) for el in lines[1].split(", ")]
    cumm_sd = [float(el) for el in lines[2].split(", ")]

    return unique_accesses, list_sd, cumm_sd


def write_dist_to_file(file_path, unique_accesses, list_sd, cumm_sd):
    try:
        with open(file_path, "w") as f:
            # unique_acesses
            s = str(unique_accesses)
            f.write(s[1 : len(s) - 1] + "\n")
            # list_sd
            s = str(list_sd)
            f.write(s[1 : len(s) - 1] + "\n")
            # cumm_sd
            s = str(cumm_sd)
            f.write(s[1 : len(s) - 1] + "\n")
    except Exception:
        print("Wrong file or file path")


if __name__ == "__main__":
    import sys
    import operator
    import argparse

    ### parse arguments ###
    parser = argparse.ArgumentParser(description="Generate Synthetic Distributions")
    parser.add_argument("--trace-file", type=str, default="./input/trace.log")
    parser.add_argument("--trace-file-binary-type", type=bool, default=False)
    parser.add_argument("--trace-enable-padding", type=bool, default=False)
    parser.add_argument("--dist-file", type=str, default="./input/dist.log")
    parser.add_argument(
        "--synthetic-file", type=str, default="./input/trace_synthetic.log"
    )
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--print-precision", type=int, default=5)
    args = parser.parse_args()

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)

    ### read trace ###
    trace = read_trace_from_file(args.trace_file)
    # print(trace)

    ### profile trace ###
    (_, stack_distances, line_accesses) = trace_profile(
        trace, args.trace_enable_padding
    )
    stack_distances.reverse()
    line_accesses.reverse()
    # print(line_accesses)
    # print(stack_distances)

    ### compute probability distribution ###
    # count items
    l = len(stack_distances)
    dc = sorted(
        collections.Counter(stack_distances).items(), key=operator.itemgetter(0)
    )

    # create a distribution
    list_sd = list(map(lambda tuple_x_k: tuple_x_k[0], dc))  # x = tuple_x_k[0]
    dist_sd = list(
        map(lambda tuple_x_k: tuple_x_k[1] / float(l), dc)
    )  # k = tuple_x_k[1]
    cumm_sd = []  # np.cumsum(dc).tolist() #prefixsum
    for i, (_, k) in enumerate(dc):
        if i == 0:
            cumm_sd.append(k / float(l))
        else:
            # add the 2nd element of the i-th tuple in the dist_sd list
            cumm_sd.append(cumm_sd[i - 1] + (k / float(l)))

    ### write stack_distance and line_accesses to a file ###
    write_dist_to_file(args.dist_file, line_accesses, list_sd, cumm_sd)

    ### generate correspondinf synthetic ###
    # line_accesses, list_sd, cumm_sd = read_dist_from_file(args.dist_file)
    synthetic_trace = trace_generate_lru(
        line_accesses, list_sd, cumm_sd, len(trace), args.trace_enable_padding
    )
    # synthetic_trace = trace_generate_rand(
    #     line_accesses, list_sd, cumm_sd, len(trace), args.trace_enable_padding
    # )
    write_trace_to_file(args.synthetic_file, synthetic_trace)
