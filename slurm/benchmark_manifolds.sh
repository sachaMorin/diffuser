#!/usr/bin/env bash

#for SIZE in small medium large;
for SIZE in large;
do
#  for DATASET in SO3GS T2 SO3 S2;
  for DATASET in SO3;
  do
    job=diff-${env}-${size}
    sbatch -J $job slurm/sbatch.sh \
      python scripts/benchmark_manifold.py \
        --dataset $DATASET-$SIZE-v1
  done
done
