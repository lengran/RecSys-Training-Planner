from os import path
from dlrm_data_pytorch import CriteoDataset
from cached_embeddingbag import CacheManager
import torch

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
            self.train_data = CriteoDataset(
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

            self.test_data = CriteoDataset(
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

            # Create empty cache now so the dataloader can callback them later.
            self.batch_pointer = 0
            num_embedding_list = self.train_data.counts
            total_embedding_row = 0
            self.offsets = list()
            for i in range(num_embedding_list.size):
                self.offsets.append(total_embedding_row)
                total_embedding_row = total_embedding_row + num_embedding_list[i]

            embedding_dim = args.arch_sparse_feature_size
            self.gpu_cache = CacheManager(num_embeddings=total_embedding_row, embedding_dim=embedding_dim, cache_ratio=args.cache_ratio, pin_weight=True)

            # Dataloaders
            self.train_loader = torch.utils.data.DataLoader(
                self.train_data,
                batch_size=args.mini_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=self.collate_cached_wrapper_criteo,
                pin_memory=False,
                drop_last=False,  # True
            )

            self.test_loader = torch.utils.data.DataLoader(
                self.test_data,
                batch_size=args.test_mini_batch_size,
                shuffle=False,
                num_workers=args.test_num_workers,
                collate_fn=self.collate_cached_wrapper_criteo,
                pin_memory=False,
                drop_last=False,  # True
            )

    def collate_cached_wrapper_criteo(self, list_of_tuples):
        '''
        Customized collate function that update cache.
        '''
        # where each tuple is (X_int, X_cat, y)
        transposed_data = list(zip(*list_of_tuples))

        X_int = torch.log(torch.tensor(transposed_data[0], dtype=torch.float) + 1)
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

        return X_int, torch.stack(lS_o), torch.stack(lS_i), T
