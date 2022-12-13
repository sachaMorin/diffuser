import warnings
import copy
import os
import pdb
import torch
import numpy as np
import copy
import pandas as pd
import itertools

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

dfs = []

for proj, seed in itertools.product(["manifold_diffusion", "start", "no_projection"], [1, 12, 123]):
    args.diffusion_loadpath = f'diffusion/defaults_H12_T20_P{proj}_S{seed}'

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
        train_dataset, batch_size=1000, num_workers=1, shuffle=False, pin_memory=True
    )

    # Make sure dataset and model use the same normalizer
    train_dataset.normalizer = diffusion.normalizer
    train_dataset.normalize()

    # Validation dataset
    # Always use the large version for validation
    dataset = args.dataset
    for s in ["small", "medium"]:
        if s in dataset:
            dataset = dataset.replace(s, "large")
    dataset_config = utils.Config(
        args.loader,
        savepath=(args.savepath, 'dataset_config.pkl'),
        env=dataset,
        horizon=train_dataset.horizon,
        normalizer=args.normalizer,
        preprocess_fns=args.preprocess_fns,
        use_padding=train_dataset.use_padding,
        max_path_length=train_dataset.max_path_length,
        seed=321,  # Validation seed
    )
    val_dataset = dataset_config()

    # Make sure dataset and model use the same normalizer
    val_dataset.normalizer = diffusion.normalizer
    val_dataset.normalize()

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1000, num_workers=1, shuffle=False, pin_memory=True
    )


    def predict(loader):
        with torch.inference_mode():
            result = dict(expert_dist=list(), expert_velocity=list(), diffuser_dist=list(), diffuser_velocity=list())
            for batch in loader:
                batch = batch_to_device(batch)
                trajectories, conds = batch
                trajectories_pred = diffusion(conds).trajectories

                trajectories = diffusion.normalizer.unnormalize(trajectories[:, :, train_dataset.action_dim:],
                                                                "observations")
                trajectories_pred = diffusion.normalizer.unnormalize(trajectories_pred[:, :, train_dataset.action_dim:],
                                                                     "observations")
                # show_diffusion(renderer,
                #                torch.repeat_interleave(trajectories_pred[0].unsqueeze(0), 12, dim=0).cpu().numpy(),
                #                savebase='.',
                #                filename='denoising_b.mp4',
                #                n_repeat=10,
                #                fps=5,
                #                )

                # Make sure trajectories are on the manifold
                # t = trajectories_pred[619]
                # show_diffusion(diffusion_experiment.renderer,
                #                torch.stack([t for _ in range(10)]).cpu().numpy(),
                #                savebase='.',
                #                filename='denoising_pred.mp4',
                #                n_repeat=10,
                #                fps=5,
                #                )
                # t = trajectories[619]
                # show_diffusion(diffusion_experiment.renderer,
                #                torch.stack([t for _ in range(10)]).cpu().numpy(),
                #                savebase='.',
                #                filename='denoising.mp4',
                #                n_repeat=10,
                #                fps=5,
                #                )

                if proj == "no_projection":
                    print("Projecting on Manifold")
                    _, trajectories_pred = train_dataset.env.projection(None, trajectories_pred)


                # Get distances
                expert_dist, expert_velocity = train_dataset.env.score(trajectories)
                diffuser_dist, diffuser_velocity = train_dataset.env.score(trajectories_pred)

                result['expert_dist'] += expert_dist.tolist()
                result['expert_velocity'] += expert_velocity.tolist()
                result['diffuser_dist'] += diffuser_dist.tolist()
                result['diffuser_velocity'] += diffuser_velocity.tolist()

        df = pd.DataFrame(result)
        anomalies_diffuser = (df['diffuser_dist'] - df['expert_dist']) < -1e-2
        if anomalies_diffuser.any():
            print(df['diffuser_dist'][anomalies_diffuser] - df['expert_dist'][anomalies_diffuser])
            warnings.warn(f"Found {anomalies_diffuser.sum()} trajectories where diffuser < expert path.")

        return df


    # Aggregate and print results
    df_train = predict(dataloader)
    df_train['split'] = 'train'
    df_train['proj'] = str(proj)
    df_train['seed'] = seed
    df_val = predict(val_dataloader)
    df_val['split'] = 'val'
    df_val['proj'] = str(proj)
    df_val['seed'] = seed
    dfs.append(df_train)
    dfs.append(df_val)

# Normalize and save dataframe
df = pd.concat(dfs, axis=0)

# Normalize columns
# divide_by = df['expert'].copy()
# for c in ['expert', 'diffuser']:
#     df[c] = df[c] / divide_by
#
#     # Some pred
#     df[c] = df[c].clip(lower=1.00)

df.to_csv(os.path.join("..", "diffuser_logs", "results", f"results_{args.dataset}.csv"))
