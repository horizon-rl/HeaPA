set -eu

############# DO NOT CHANGE ANYTHING BELOW UNLESS YOU KNOW WHAT YOU ARE DOING

### MLFLOW_TRACKING_URI ###
MLFLOW_TRACKING_URI=https://prod.$REGION.internal.mlflow.nile.amazon.dev

### P4D.24XLARGE and P4DE.24XLARGE specs ###
# NUM_GPUS_PER_NODE=8
# NUM_CPUS_PER_NODE=96
# MEMORY_PER_NODE=1143265
### P5E.48XLARGE specs ###
NUM_GPUS_PER_NODE=8
NUM_CPUS_PER_NODE=192
MEMORY_PER_NODE=2013184

DOCKER_IMAGE_TAG=nemo-nile-runner
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 684288478426.dkr.ecr.us-west-2.amazonaws.com
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 684288478426.dkr.ecr.us-east-1.amazonaws.com
# Build docker image for nemo-nile-runner
echo "Start building docker image for $DOCKER_IMAGE_TAG:${RANDOM_TAG}"

docker build --tag=$DOCKER_IMAGE_TAG:${RANDOM_TAG} -f ./examples/aws_batch/Dockerfile.verl-runner .
# docker build --tag=$DOCKER_IMAGE_TAG:${RANDOM_TAG} -f ../aws_batch/Dockerfile.verl-runner ../aws_batch
echo "Built docker image $DOCKER_IMAGE_TAG:${RANDOM_TAG}"

# # Push docker image to ECR
echo "Pushing docker image $DOCKER_IMAGE_TAG:${RANDOM_TAG} to ECR"

ECR_DOCKER_IMAGE=684288478426.dkr.ecr.${REGION}.amazonaws.com/${DOCKER_IMAGE_TAG}:${RANDOM_TAG}
docker tag ${DOCKER_IMAGE_TAG}:${RANDOM_TAG} $ECR_DOCKER_IMAGE
docker push $ECR_DOCKER_IMAGE


echo "Registering job definition"
random_job_suffix=$(printf '%s' $(echo "$RANDOM" | md5sum) | cut -c 1-10)
JOB_DEFINITION_NAME=${DOCKER_IMAGE_TAG}-${JOB_NAME}
JOB_TYPE=multinode


HEALTH_CHECK=''
if [[ "$ENABLE_EFA_HEALTHCHECK" == true ]]; then
  HEALTH_CHECK="python ./src/hardware_healthcheck.py &&"
fi


build_container_properties() {
CONTAINER_PROPERTIES='{
    "image": "'$ECR_DOCKER_IMAGE'",
    "command": '$RUNCMD_LIST_STR',
    "privileged": true,
    "resourceRequirements": [
        {
            "value": "'${NUM_GPUS_PER_NODE}'",
            "type": "GPU"
        },
        {
            "value": "'${NUM_CPUS_PER_NODE}'",
            "type": "VCPU"
        },
        {
            "value": "'${MEMORY_PER_NODE}'",
            "type": "MEMORY"
        }
    ],
    "ulimits": [
        {
            "name": "memlock",
            "softLimit": -1,
            "hardLimit": -1
        },
        {
            "name": "stack",
            "softLimit": 67108864,
            "hardLimit": 67108864
        }
    ],
    "volumes": [
      {
        "host": {
          "sourcePath": "/fsx-alignment"
        },
        "name": "local-fsx-alignment-volume"
      },
      {
        "host": {
          "sourcePath": "/fsx-Training"
        },
        "name": "local-fsx-Training-volume"
      },
      {
        "host": {
          "sourcePath": "/tmp"
        },
        "name": "local-tmp-volume"
      },
      {
        "host": {
          "sourcePath": "/home"
        },
        "name": "local-home-volume"
      }
    ],
    "environment": [
        {
          "name": "ML_FLOW_EXPERIMENT_NAME",
          "value": "'${USER}'-nemo-aws-batch"
        },
        {
          "name": "ML_FLOW_RUN_NAME",
          "value": "aws-batch-'${JOB_NAME}'"
        },
        {
          "name": "NVTE_UB_SOCKET_IFNAME",
          "value": "eth0"
        },
        {
          "name": "FI_PROVIDER",
          "value": "efa"
        },
        {
          "name": "NODE_GROUP_NUM_NODES",
          "value": "'${NODE_GROUP_NUM_NODES}'"
        },
        {
          "name": "RANDOM_ID",
          "value": "'${RANDOM_ID}'"
        }
    ],
    "mountPoints": [
        {
            "sourceVolume": "local-fsx-alignment-volume",
            "containerPath": "/fsx-alignment",
            "readOnly": false
        },
        {
            "sourceVolume": "local-fsx-Training-volume",
            "containerPath": "/fsx-Training",
            "readOnly": false
        },
        {
            "sourceVolume": "local-tmp-volume",
            "containerPath": "/tmp",
            "readOnly": false
        },
        {
            "sourceVolume": "local-home-volume",
            "containerPath": "/home",
            "readOnly": false
        }
    ],
    "linuxParameters": {
        "devices": [
            {
            "hostPath": "/dev/infiniband/uverbs0",
            "containerPath": "/dev/infiniband/uverbs0",
            "permissions": [
                "READ",
                "WRITE",
                "MKNOD"
            ]
            },
            {
            "hostPath": "/dev/infiniband/uverbs1",
            "containerPath": "/dev/infiniband/uverbs1",
            "permissions": [
                "READ",
                "WRITE",
                "MKNOD"
            ]
            },
            {
            "hostPath": "/dev/infiniband/uverbs2",
            "containerPath": "/dev/infiniband/uverbs2",
            "permissions": [
                "READ",
                "WRITE",
                "MKNOD"
            ]
            },
            {
            "hostPath": "/dev/infiniband/uverbs3",
            "containerPath": "/dev/infiniband/uverbs3",
            "permissions": [
                "READ",
                "WRITE",
                "MKNOD"
            ]
            },
            {
              "hostPath": "/dev/infiniband/uverbs4",
              "containerPath": "/dev/infiniband/uverbs4",
              "permissions": [
                "READ",
                "WRITE",
                "MKNOD"
              ]
            },
            {
              "hostPath": "/dev/infiniband/uverbs5",
              "containerPath": "/dev/infiniband/uverbs5",
              "permissions": [
                "READ",
                "WRITE",
                "MKNOD"
              ]
            },
            {
              "hostPath": "/dev/infiniband/uverbs6",
              "containerPath": "/dev/infiniband/uverbs6",
              "permissions": [
                "READ",
                "WRITE",
                "MKNOD"
              ]
            },
            {
              "hostPath": "/dev/infiniband/uverbs7",
              "containerPath": "/dev/infiniband/uverbs7",
              "permissions": [
                "READ",
                "WRITE",
                "MKNOD"
              ]
            },
            {
              "hostPath": "/dev/infiniband/uverbs8",
              "containerPath": "/dev/infiniband/uverbs8",
              "permissions": [
                "READ",
                "WRITE",
                "MKNOD"
              ]
            },
            {
              "hostPath": "/dev/infiniband/uverbs9",
              "containerPath": "/dev/infiniband/uverbs9",
              "permissions": [
                "READ",
                "WRITE",
                "MKNOD"
              ]
            },
            {
              "hostPath": "/dev/infiniband/uverbs10",
              "containerPath": "/dev/infiniband/uverbs10",
              "permissions": [
                "READ",
                "WRITE",
                "MKNOD"
              ]
            },
            {
              "hostPath": "/dev/infiniband/uverbs11",
              "containerPath": "/dev/infiniband/uverbs11",
              "permissions": [
                "READ",
                "WRITE",
                "MKNOD"
              ]
            },
            {
              "hostPath": "/dev/infiniband/uverbs12",
              "containerPath": "/dev/infiniband/uverbs12",
              "permissions": [
                "READ",
                "WRITE",
                "MKNOD"
              ]
            },
            {
              "hostPath": "/dev/infiniband/uverbs13",
              "containerPath": "/dev/infiniband/uverbs13",
              "permissions": [
                "READ",
                "WRITE",
                "MKNOD"
              ]
            },
            {
              "hostPath": "/dev/infiniband/uverbs14",
              "containerPath": "/dev/infiniband/uverbs14",
              "permissions": [
                "READ",
                "WRITE",
                "MKNOD"
              ]
            },
            {
              "hostPath": "/dev/infiniband/uverbs15",
              "containerPath": "/dev/infiniband/uverbs15",
              "permissions": [
                "READ",
                "WRITE",
                "MKNOD"
              ]
            }
        ],
        "sharedMemorySize": 1143265,
        "tmpfs": [{
            "containerPath": "/tmpfs",
            "size": 150000
        }]
    },
    "logConfiguration": {
        "logDriver": "awslogs",
        "options": {},
        "secretOptions": []
    }
  }'
}

# ========== set veRL container properties ==========
NODE_GROUP_NUM_NODES=${VERL_NUM_NODES}
RUN_ARGS='run.user='${USER}'-nemo-aws-batch  run.name=aws-batch-'${JOB_NAME}
RUNCMD_LIST=("bash" "-c" "${HEALTH_CHECK} ${RUNCMD}")
RUNCMD_LIST_STR=$(printf '%s\n' "${RUNCMD_LIST[@]}" | jq -R . | jq -s .)
echo "DOCKER RUN CMD: $RUNCMD_LIST_STR"
build_container_properties
VERL_CONTAINER_PROPERTIES=${CONTAINER_PROPERTIES}
echo ${VERL_CONTAINER_PROPERTIES}

# ========== set GRS container properties when $GRS_NUM_NODES > 1 ==========
if (( GRS_NUM_NODES > 0 )); then
  NODE_GROUP_NUM_NODES=${GRS_NUM_NODES}
  ECR_DOCKER_IMAGE=$GRS_ECR_IMAGE
  RUNCMD_LIST=("bash" "-c" "bash scripts/start_ray_cluster.sh $VERL_NUM_NODES $GRS_NUM_NODES $RANDOM_ID ; $GRS_RUN_CMD ; sleep 259200")
  RUNCMD_LIST_STR=$(printf '%s\n' "${RUNCMD_LIST[@]}" | jq -R . | jq -s .)
  echo "DOCKER RUN CMD: $RUNCMD_LIST_STR"
  build_container_properties
  GRS_CONTAINER_PROPERTIES=${CONTAINER_PROPERTIES}
  echo ${GRS_CONTAINER_PROPERTIES}
fi


TOTAL_NUM_NODES=$(( VERL_NUM_NODES + GRS_NUM_NODES ))
VERL_NODE_RANGE=$(( VERL_NUM_NODES - 1))
if (( GRS_NUM_NODES > 0 )); then
  GRS_NODE_RANGE=$(( TOTAL_NUM_NODES - 1))
  NODE_PROP_STR='{
      "numNodes":'${TOTAL_NUM_NODES}',
      "mainNode":0,
      "nodeRangeProperties": [
        {
          "targetNodes": "0:'${VERL_NODE_RANGE}'",
          "container": '${VERL_CONTAINER_PROPERTIES}'
        }, {
          "targetNodes": "'${VERL_NUM_NODES}':'${GRS_NODE_RANGE}'",
          "container": '${GRS_CONTAINER_PROPERTIES}'
        }
      ]
  }'
else
  NODE_PROP_STR='{
      "numNodes":'${TOTAL_NUM_NODES}',
      "mainNode":0,
      "nodeRangeProperties": [
        {
          "targetNodes": "0:'${VERL_NODE_RANGE}'",
          "container": '${VERL_CONTAINER_PROPERTIES}'
        }
      ]
  }'
fi

NODE_TAG='{"torchx.pytorch.org/user":"'${USER}'"}'

echo $NODE_TAG

submit_scheduler_aws_batch() {

  echo "submit job to aws batch..."
  NODE_PROP_STR=$(echo "$NODE_PROP_STR" | tr -d '\n')
  response_reg_job_def=$(aws batch register-job-definition \
      --job-definition-name $JOB_DEFINITION_NAME \
      --type $JOB_TYPE \
      --scheduling-priority $JOB_PRIORITY \
      --node-properties "$NODE_PROP_STR" \
      --region $REGION \
      --output table \
      --tags "$NODE_TAG")

  echo "response_reg_job_def: $response_reg_job_def"
  echo "Registered job definition $JOB_DEFINITION_NAME"

  # waiting job definition register become active.
  sleep 10

  echo "Submitting job"
  aws batch submit-job --job-name $JOB_NAME --job-queue $JOB_QUEUE  --job-definition $JOB_DEFINITION_NAME --share-identifier $SHARE_IDENTIFIER --region $REGION --tags $NODE_TAG
  echo "Submitted job $JOB_NAME"
}

submit_scheduler_greenland() {
    echo "Submit job to greenland..."

    GREENLAND_PAYLOAD='
    {
        "JobName": "'${JOB_NAME}'",
        "InitiativeId": "'${INITIATIVE_ID}'",
        "InstanceCount": '${TOTAL_NUM_NODES}',
        "InstanceType": "'${INSTANCE_TYPE}'",
        "IsProduction": '${IS_PRODUCTION}',
        "JobType": "OBX",
        "region": "'${REGION}'",
        "Role": "'${ROLE}'",
        "BatchJobDefinitionParameters": {
            "jobDefinitionName": "'${JOB_NAME}-definition'",
            "type": "'${JOB_TYPE}'",
            "nodeProperties": '${NODE_PROP_STR}',
            "retryStrategy": {
                "attempts": 1,
                "evaluateOnExit": [
                    {
                        "onExitCode": "0",
                        "action": "EXIT"
                    }
                ]
            },
            "tags": '${NODE_TAG}'
        }
    }
    '

    export $(printf "AWS_ACCESS_KEY_ID=%s AWS_SECRET_ACCESS_KEY=%s AWS_SESSION_TOKEN=%s" $(aws sts assume-role \
        --role-arn $ROLE \
        --role-session-name GreenlandAPISession \
        --query "Credentials.[AccessKeyId,SecretAccessKey,SessionToken]" \
        --region us-east-1 \
        --output text))

    jq -c '.' <<< $GREENLAND_PAYLOAD

    awscurl --service execute-api -X POST -d "$(jq -c '.' <<< $GREENLAND_PAYLOAD)" https://prod.us-east-1.greenland.alloy.amazon.dev/cs/v2/submitjob --region us-east-1

    unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
    echo "Submitted job $JOB_NAME"
}


case $SCHEDULER in
  greenland)
    submit_scheduler_greenland
    ;;
  aws_batch)
    submit_scheduler_aws_batch
    ;;
  *)
    echo "Error: Invalid scheduler name '$SCHEDULER'"
    exit 1
    ;;
esac