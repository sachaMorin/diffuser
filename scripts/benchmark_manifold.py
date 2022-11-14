import copy
import os
import pdb
import torch
import numpy as np
import copy
import pandas as pd

# import diffuser.sampling as sampling
import diffuser.utils as utils
import matplotlib.pyplot as plt
from diffuser.models.diffusion import default_sample_fn
from diffuser.utils.arrays import batch_to_device
from diffuser.utils.colab import run_diffusion, show_diffusion


# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'S2-v1'
    config: str = 'config.locomotion'


args = Parser().parse_args('plan')

# args.diffusion_loadpath = 'diffusion/defaults_H12_T20_Pspherical_S42'


# -----------------------------------------------------------------------------#
# ---------------------------------- loading ----------------------------------#
# -----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)

## ensure that the diffusion model and value function are compatible with each other
diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1000, num_workers=1, shuffle=True, pin_memory=True
)

# Make sure dataset and model use the same normalizer
dataset.normalizer = diffusion.normalizer
dataset.normalize()

with torch.inference_mode():
    result = dict(shortest=list(), expert=list(), diffuser=list())
    for batch in dataloader:
        batch = batch_to_device(batch)
        trajectories, conds = batch
        trajectories_pred = diffusion(conds).trajectories
        import pdb; pdb.set_trace()

        # Unormalize trajectories
        trajectories = diffusion.normalizer.unnormalize(trajectories[:, :, dataset.action_dim:], "observations")
        trajectories_pred = diffusion.normalizer.unnormalize(trajectories_pred[:, :, dataset.action_dim:],
                                                             "observations")

        # Get distances
        expert_dist = dataset.env.seq_geodesic_distance(trajectories)
        diffuser_dist = dataset.env.seq_geodesic_distance(trajectories_pred)

        start_goal = trajectories[:, [0, -1], :]

        shortest_dist = dataset.env.seq_geodesic_distance(start_goal)


        result['expert'] += expert_dist.tolist()
        result['shortest'] += shortest_dist.tolist()
        result['diffuser'] += diffuser_dist.tolist()

df = pd.DataFrame(result)
anomalies = (df['diffuser'] < df['shortest']).sum()
if anomalies:
    raise Exception(f"Found {anomalies} trajectories where diffuser < shortest path.")

print(df)
print(df.mean(axis=0))
