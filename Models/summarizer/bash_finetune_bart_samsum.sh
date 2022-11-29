#!/bin/bash
#$ -cwd
#$ -l rt_G.small=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o logs/$JOB_NAME.o$JOB_ID

# ======== Modules ========
source /etc/profile.d/modules.sh
module load cuda/10.0/10.0.130.1

# For using GCC7.4 and MPI build by it
# pushd /groups2/gca50014/modules_y1r_gc/cuda100-ompi21-gcc74
# source ./gcc_source.sh
# source ./mpi_source.sh
# source ./nccl_source.sh
# popd

# ======== Virtualenv ======
export PYTHONPATH=.
# export PATH=/home/acc12119do/dl/virtualenv_py382/bin:$PATH

# ======== Configurations For ABCI Cluster ========
# N_PERNODE=4
# N_GPU=$(( $NHOSTS * $N_PERNODE ))

NUM_NODES=${NHOSTS}
NUM_GPUS_PER_NODE=1
NUM_GPUS_PER_SOCKET=$(expr ${NUM_GPUS_PER_NODE} / 2)
NUM_PROCS=$(expr ${NUM_NODES} \* ${NUM_GPUS_PER_NODE})

echo "NUM_NODES : $NUM_NODES"
echo "NUM_GPUS_PER_NODE : $NUM_GPUS_PER_NODE"
echo "NUM_PROCS : $NUM_PROCS"
echo "NUM_GPUS_PER_SOCKET : $NUM_GPUS_PER_SOCKET"

CLUSTER="abci"

# ======== Configurations For Pytorch ========

FINETUNE_CMD="bash finetune_bart_samsum.sh"

# ======== Execute Command ========

CMD_EXECUTE="$FINETUNE_CMD"

source ~/.bash_profile

echo "[CMD_EXECUTE] :  pyenv local 3.6.12"
echo ""
eval pyenv local 3.6.12

# echo "[CMD_EXECUTE] :  pipenv shell"
# echo ""
# eval pipenv shell

echo "Job started on $(date)"
echo "................................"

echo "[CMD_EXECUTE] :  $CMD_EXECUTE"
echo ""
eval $CMD_EXECUTE

echo "Job done on $(date)"
