import os
import pdb
import torch
import numpy as np


# import diffuser.sampling as sampling
import diffuser.utils as utils
import matplotlib.pyplot as plt
from diffuser.models.diffusion import default_sample_fn
from diffuser.utils.colab import run_diffusion, show_diffusion


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'S1-v1'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan')

args.diffusion_loadpath = 'diffusion/defaults_H12_T20'


#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)

## ensure that the diffusion model and value function are compatible with each other
diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy_config = utils.Config(
    args.policy,
    guide=None,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    sample_fn=default_sample_fn,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)
policy = policy_config()

mid = np.sqrt(2)/2

# Visualize the denoising process
cond = {0: torch.tensor([1.0, 0.0]), -1: torch.tensor([-mid, -mid])}

_, samples = policy(cond, batch_size=1, verbose=args.verbose, return_chain=True)

chains = samples.chains[0]

save_base = os.path.join(args.loadbase, args.dataset, args.diffusion_loadpath)
show_diffusion(renderer,
               chains,
               savebase=save_base,
               filename='denoising.mp4',
               n_repeat=10,
               fps=5,
               )

# Visualize the diffusion process
# Make sure you have a full trajectory otherwise you may end up with the goal outside the manifold due to padding
traj = torch.from_numpy(dataset[10].trajectories) # Get first trajectory
traj = traj.to(diffusion.betas.device).unsqueeze(0)
t = torch.Tensor([diffusion.n_timesteps]).long().to(traj.device)
trajs = diffusion.q_sample(traj, t=t, return_chain=True).cpu().numpy()
trajs = dataset.normalizer.unnormalize(trajs[:, :, dataset.action_dim:], "observations")
trajs = trajs[:, :-1, :]  # Trim trajectory here to remove padding

# Fix start and goal
trajs[:, 0, :] = trajs[0, 0, :]
trajs[:, -1, :] = trajs[0, -1, :]

show_diffusion(renderer,
               trajs,
               savebase=save_base,
               filename='diffusion.mp4',
               n_repeat=10,
               fps=5,
               )

