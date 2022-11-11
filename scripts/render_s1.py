"""Render a few S1 trajectories."""
import pdb
import torch
import numpy as np


# import diffuser.sampling as sampling
import diffuser.utils as utils
import matplotlib.pyplot as plt
from diffuser.models.diffusion import default_sample_fn


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


#  Specify manual start/stop goals first
start = torch.tensor([
    [1.0, 0.0],
    [1.0, 0.0],
    [1.0, 0.0],
    # [1.0, 0.0],
    # [1.0, 0.0],
    # [1.0, 0.0],
    # [1.0, 0.0],
])

stop = torch.tensor([
    [np.sqrt(2) / 2, np.sqrt(2) / 2],
    [0.0, 1.0],
    [0.0, -1.0],
    # [-1.0, 0.0],
    # [-1.0, 0.0],
    # [-1.0, 0.0],
    # [-1.0, 0.0],
])

# Add random start stop goals
new_starts = torch.randn((6, 2))
new_starts /= torch.linalg.norm(new_starts, dim=1).reshape((-1, 1))
start = torch.cat((start, new_starts))

new_goals = torch.randn((6, 2))
new_goals /= torch.linalg.norm(new_goals, dim=1).reshape((-1, 1))
stop = torch.cat((stop, new_goals))



# Render trajectories
for s, g in zip(start, stop):
    conditions = {0: s, -1: g}
    _, samples = policy(conditions, batch_size=1, verbose=args.verbose)
    trajectory = utils.to_np(samples.observations)[0]
    im = renderer.render(trajectory)
    plt.imshow(im)
    plt.show()