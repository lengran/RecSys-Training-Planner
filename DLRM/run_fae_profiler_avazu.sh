#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# --hot-emb-gpu-mem=268435456

dlrm_fae_profiler_py="python ./dlrm_input_profiler_avazu.py "

$dlrm_fae_profiler_py  --arch-sparse-feature-size=16 \
						--arch-mlp-top="512-256-1" \
						--arch-mlp-bot="2-512-256-64-16" \
						--data-generation=avazu \
						--data-set=avazu \
						--raw-data-file=./input/avazu/train.txt \
						--processed-data-file=./input/avazu/train_processed.npz \
						--mini-batch-size=1024 \
						--hot-emb-gpu-mem=159610150 \
						--ip-sampling-rate=5
