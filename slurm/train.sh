#for size in small medium large;
for size in large;
do
  for env in SO3;
#  for env in S2 T2 SO3 SO3GS;
  do
    for seed in 1 12 123;
    do
      for mode in no_projection start manifold_diffusion;
      do
        job=diff-${env}-${size}-${mode}-${seed}
        echo $job
        sbatch -J $job slurm/sbatch.sh \
          python scripts/train.py \
            --dataset $env-$size-v1 \
            --seed $seed \
            --manifold_diffuser_mode $mode
      done
    done
  done
done
