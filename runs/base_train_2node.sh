#!/bin/bash

# Launch base_train on two PPU nodes with four devices per node.
# Run this script once on each node at the same time:
#   NODE_RANK=0 on 192.168.12.110
#   NODE_RANK=1 on 192.168.12.111
#
# The checkpoint and dataset directory must be a shared path visible from both
# nodes. The script is intentionally invoked with `bash`, so it does not need
# executable permission changes.

set -euo pipefail

: "${NODE_RANK:?Set NODE_RANK to 0 on 192.168.12.110 or 1 on 192.168.12.111}"
: "${NANOCHAT_BASE_DIR:?Set NANOCHAT_BASE_DIR to the same shared path on both nodes}"

MASTER_ADDR="${MASTER_ADDR:-192.168.12.110}"
MASTER_PORT="${MASTER_PORT:-29500}"
NCCL_IB_HCA="${NCCL_IB_HCA:-vsolar_0}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-bfloat16}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_HCA
export NCCL_IB_GID_INDEX
export NCCL_ALLOC_BASE_SIZE="${NCCL_ALLOC_BASE_SIZE:-67108864}"
export NCCL_HOST_TO_DEV_TRANS_SIZE="${NCCL_HOST_TO_DEV_TRANS_SIZE:-67108864}"
export NCCL_DEV_TO_HOST_TRANS_SIZE="${NCCL_DEV_TO_HOST_TRANS_SIZE:-67108864}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"

# Set NCCL_SOCKET_IFNAME explicitly only when vsolar0 is also a Linux netdev.
if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
    export NCCL_SOCKET_IFNAME
fi

torchrun \
    --nnodes=2 \
    --nproc-per-node=4 \
    --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    -m scripts.base_train -- "$@"