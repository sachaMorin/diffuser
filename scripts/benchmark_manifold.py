import warnings
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
train_dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1000, num_workers=1, shuffle=True, pin_memory=True
)

# Make sure dataset and model use the same normalizer
train_dataset.normalizer = diffusion.normalizer
train_dataset.normalize()


# Validation dataset
dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=train_dataset.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=train_dataset.use_padding,
    max_path_length=train_dataset.max_path_length,
    seed=123,  # Validation seed
)

val_dataset = dataset_config()

# Make sure dataset and model use the same normalizer
val_dataset.normalizer = diffusion.normalizer
val_dataset.normalize()

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1000, num_workers=1, shuffle=True, pin_memory=True
)


def predict(loader):
    with torch.inference_mode():
        result = dict(shortest=list(), expert=list(), diffuser=list())
        for batch in loader:
            batch = batch_to_device(batch)
            trajectories, conds = batch
            trajectories_pred = diffusion(conds).trajectories

            # Make sure trajectories are projected onto the manifold
            # trajectories_pred = diffusion.project(trajectories_pred)

            # Unormalize trajectories
            trajectories = diffusion.normalizer.unnormalize(trajectories[:, :, train_dataset.action_dim:],
                                                            "observations")
            trajectories_pred = diffusion.normalizer.unnormalize(trajectories_pred[:, :, train_dataset.action_dim:],
                                                                 "observations")
            # Make sure trajectories are on the manifold
            _, trajectories_pred = train_dataset.env.projection(None, trajectories_pred)

            # Get distances
            expert_dist = train_dataset.env.seq_geodesic_distance(trajectories)
            diffuser_dist = train_dataset.env.seq_geodesic_distance(trajectories_pred)

            start_goal = trajectories[:, [0, -1], :]

            shortest_dist = train_dataset.env.seq_geodesic_distance(start_goal)

            result['expert'] += expert_dist.tolist()
            result['shortest'] += shortest_dist.tolist()
            result['diffuser'] += diffuser_dist.tolist()

    df = pd.DataFrame(result)
    anomalies_diffuser = df['diffuser'] < df['shortest']
    anomalies_expert = df['expert'] < df['shortest']
    if anomalies_diffuser.any():
        print(f"Found {anomalies_diffuser.sum()} trajectories where diffuser < shortest path.")
        import pdb; pdb.set_trace()
    if anomalies_expert.any():
        warnings.warn(f"Found {anomalies_expert.sum()} trajectories where expert < shortest path.")

    return df


# Aggregate and print results
df_train = predict(dataloader)
df_train['split'] = 'train'
df_val = predict(val_dataloader)
df_val['split'] = 'val'
df = pd.concat([df_train, df_val], axis=0)
print(f"Average over {df.shape[0]} trajectories")
print(df.groupby('split').mean())
