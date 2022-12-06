#!/bin/bash

# Parameters
#SBATCH --job-name=diffusion
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ARRAY_TASKS,FAIL,TIME_LIMIT
#SBATCH --mail-user=sacha.morin@mila.quebec
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --partition=unkillable
#SBATCH -o /network/scratch/s/sacha.morin/diffuser/diffuser_logs/slurm/slurm-%j.out  # Write the log on scratch


module load singularity

CLUSTER_PATH=/network/scratch/s/sacha.morin/diffuser

export ARGS="$@"
echo "Args: ${ARGS}"

singularity exec -B /localscratch -B $CLUSTER_PATH -B /Tmp --nv --writable-tmpfs diffuser.sif \
        /bin/bash -c "pip install -e .; ${ARGS}"

# srun --unbuffered singularity run --nv -H $HOME:/home -B /localscratch -B $CLUSTER_PATH -B /Tmp $CLUSTER_PATH/deployment/plan2vec.sif -u -m submitit.core._submit /network/scratch/s/sacha.morin/o4a/deployment/logs/experiments/multiruns/local_metric/2022-12-01_14-40-41/.submitit/%j

# singularity run --nv -H $HOME:/home -B /localscratch -B $CLUSTER_PATH -B /Tmp $CLUSTER_PATH/deployment/plan2vec.sif
