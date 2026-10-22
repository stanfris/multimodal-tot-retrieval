#!/usr/bin/env bash
# ABCI 3.0 toolkit matching the locked PyTorch CUDA 13.0 runtime.
source /etc/profile.d/modules.sh
module unload cuda
module load cuda/13.0/13.0.1
