for env in S2 T2 SO3 SO3GS;
do
	for seed in 1;
	do
    for mode in start no_projection;
    do
      job=diff-${env}-${mode}-${seed}
      echo $job
      sbatch -J $job slurm/sbatch.sh \
        python scripts/train.py \
          --dataset $env-v1 \
          --seed $seed \
          --manifold_diffuser_mode $mode
	  done
	done
done
