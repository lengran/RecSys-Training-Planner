from dlrm_data_pytorch_cache import PlanedSampler as CriteoSampler
from dlrm_data_pytorch_cache_avazu import PlanedSampler as AvazuSampler
from dlrm_data_pytorch import CriteoDataset
from dlrm_data_pytorch_avazu import AvazuDataset

if __name__ == "__main__":
    dataset = "avazu"
    print("dataset: " + dataset)

    if dataset == "criteo":
        # parameters are copied from run_dlrm_baseline_cpu_gpu_cache.sh
        train_data = CriteoDataset(
                    "kaggle",
                    -1,
                    0.0,
                    "total",
                    "train",
                    "./input/kaggle/train.txt",
                    "./input/kaggle/kaggleAdDisplayChallenge_processed.npz",
                    False,
                    False
                )
        print("Dataset initilized.")

        # Change this parameters according to args in run_dlrm_baseline_cpu_gpu_cache.sh
        sampler = CriteoSampler(False, "./input/training_plan/", batch_size=1024, shuffle=True, drop_last=False, dataset=train_data, suffix=None)
        sampler.generate_batches()
    elif dataset == "avazu":
        # parameters are copied from run_dlrm_baseline_cpu_gpu_cache.sh
        train_data = AvazuDataset(
                    "avazu",
                    -1,
                    0.0,
                    "total",
                    "train",
                    "./input/avazu/train.txt",
                    "./input/avazu/train_processed.npz",
                    False,
                    False
                )
        print("Dataset initilized.")
    
        # Change this parameters according to args in run_dlrm_baseline_cpu_gpu_cache.sh
        sampler = AvazuSampler(False, "./input/avazu/training_plan/", batch_size=1024, shuffle=True, drop_last=False, dataset=train_data, suffix=None)
        sampler.generate_batches()
    
    print("Batch describer generated.")