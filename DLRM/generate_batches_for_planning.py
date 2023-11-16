from dlrm_data_pytorch_cache import PlanedSampler
from dlrm_data_pytorch import CriteoDataset

if __name__ == "__main__":
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

    # Change this parameters according to args in run_dlrm_baseline_cpu_gpu_cache.sh
    sampler = PlanedSampler(False, "./input/training_plan/", batch_size=1024, shuffle=True, drop_last=False, dataset=train_data)
    sampler.generate_batches()