python dlrm_baseline_cpu_gpu_cache_avazu.py --arch-sparse-feature-size=16 \
								--arch-mlp-bot="2-512-256-64-16" \
								--arch-mlp-top="512-256-1" \
								--data-generation=avazu \
								--data-set=avazu \
								--raw-data-file=./input/avazu/train.txt \
								--processed-data-file=./input/avazu/train_processed.npz \
								--loss-function=bce \
								--round-targets=True \
								--mini-batch-size=1024 \
								--print-freq=4096 \
								--print-time \
								--cache-ratio=0.05 \
								--training-plan-dir=./input/avazu/training_plan/ #\
								# --enable-profiling \
								# --num-batches=100
