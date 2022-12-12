#!/usr/bin/env bash

for SIZE in small medium large;
do
  for DATASET in SO3GS T2 SO3 S2;
  do
    job=diff-${env}-${size}
    sbatch -J $job slurm/sbatch_small.sh \
      python scripts/benchmark_manifold.py \
        --dataset $DATASET-$SIZE-v1
  done
done
