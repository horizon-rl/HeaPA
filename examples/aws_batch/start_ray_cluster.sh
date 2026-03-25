#!/bin/bash

export VERL_NUM_NODES=$1
echo $VERL_NUM_NODES

RAY_PORT_NUMBER=6379
WORKER_SLEEP_TIME=300 # 5 mins

# Function to check if the head node is up
wait_for_head_node() {
    local head_ip=$1
    local timeout=$2
    local interval=10
    local elapsed=0

    echo "Waiting for head node at $head_ip:$RAY_PORT_NUMBER to be up..."

    while ! timeout 10 bash -c "echo > /dev/tcp/$head_ip/$RAY_PORT_NUMBER"; do
        sleep $interval
        elapsed=$((elapsed + interval))
        if [[ $elapsed -ge $timeout ]]; then
            echo "Timeout(${timeout}) waiting for head node to be ready. Exiting."
            exit 1
        fi
    done

    echo "Head node is ready."
}

# Function to check if the number of connected nodes equals the expected number
wait_for_nodes() {
    local expected_nodes=$1
    local timeout=$2
    local interval=30
    local elapsed=0

    echo "Waiting for $expected_nodes nodes to join the cluster..."

    while true; do
        connected_nodes=$(ray status | awk '/Active:/,/Pending:/' | grep '^ 1' | wc -l)
        echo "There are $connected_nodes nodes connected to the cluster"

        if [[ $connected_nodes == $expected_nodes ]]; then
            echo "Expected $connected_nodes nodes are connected to the cluster. Ray cluster is ready"
            echo "Fetching Ray status."
            ray status
            break
        fi

        # if [[ $connected_nodes > $expected_nodes ]]; then
        #     echo "$connected_nodes nodes are connected to the cluster, more than expected $expected_nodes. Exiting."
        #     exit 1
        # fi

        sleep $interval
        elapsed=$((elapsed + interval))
        if [[ $elapsed -ge $timeout ]]; then
            echo "Timeout(${timeout}) waiting for nodes to join the cluster. Exiting."
            exit 1
        fi
    done
}

# Environment setup
export MLFLOW_TRACKING_URI="https://prod.us-west-2.mlflow.nile.amazon.dev"
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true
export RAY_DEDUP_LOGS=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_COMPILE_DISABLE=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-"eth0"}

# Check if the current node is the main node
if [ "${AWS_BATCH_JOB_MAIN_NODE_INDEX}" -eq "${AWS_BATCH_JOB_NODE_INDEX}" ]; then
    # This is the main node
    echo "Starting Ray head node in batch node ${AWS_BATCH_JOB_NODE_INDEX}"

    # Start the Ray head node
    ray start --head --port=${RAY_PORT_NUMBER}

    # Wait for the worker nodes to start
    wait_for_nodes $VERL_NUM_NODES 1800

    echo "Ray Cluster Started ..."
    # exit 0
    # sleep $WORKER_SLEEP_TIME   
else
    # This is a worker node
    echo "Starting Ray worker node in batch node ${AWS_BATCH_JOB_NODE_INDEX}"

    if [[ -z "${AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS}" ]]; then
        echo "Main node IP not found, exiting."
        exit 1
    fi

    # Start the Ray worker node
    ray start --address="${AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS}:${RAY_PORT_NUMBER}"

    # Worker node will sleep forever. It will be immediately terminated when head node exited.
    while true
    do
        # Print log as heartbeat
        echo "Worker node heartbeat every ${WORKER_SLEEP_TIME} seconds".
        sleep $WORKER_SLEEP_TIME
    done
fi