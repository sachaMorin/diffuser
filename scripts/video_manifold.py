import copy
import os
import pdb
import torch
import numpy as np
import copy


# import diffuser.sampling as sampling
import diffuser.utils as utils
import matplotlib.pyplot as plt
from diffuser.models.diffusion import default_sample_fn
from diffuser.utils.colab import run_diffusion, show_diffusion


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'T2-v1'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan')

args.diffusion_loadpath = 'diffusion/defaults_H12_T20_Pmanifold_diffusion_S1'


#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch,
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
    normalizer=diffusion.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    sample_fn=default_sample_fn,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)

policy = policy_config()


# Visualize the denoising process
# Torus
if 'T2' in args.dataset:
    # cond = {0: torch.tensor([1.0, 0.0, 1.0, 0.0]), -1: torch.tensor([0.0, 1.0, 1.0, 0.0])}
    cond = {0: torch.tensor([1.0, 0.0, 1.0, 0.0]), -1: -torch.tensor([0.0, 1.0, 0.0, 1.0])}
elif 'S2' in args.dataset:
    cond = {0: torch.tensor([1.0, 0.0, 0.0]), -1: torch.tensor([0.0, -1.0, 0.0])}
elif 'SO3' in args.dataset:
    cond = {
        # 0: torch.tensor([ 0.4684, -0.0467,  0.8672,  0.2154, -0.1691,  0.9754]),  # Test hard traj
        # -1: torch.tensor([ 0.4931,  0.0054,  0.8698, -0.0248,  0.0189,  0.9997]),  # Test hard traj
        # 0: torch.tensor([-0.9276,  0.3369,  0.1208,  0.6793,  0.3535,  0.6520]),  # Test hard traj 2
        # -1: torch.tensor([-0.2745, -0.7507,  0.4615, -0.6511, -0.8436, -0.1119]),  # Test hard traj 2
        # 0: torch.tensor([ 0.4684, -0.0467,  0.8672,  0.2154, -0.1691,  0.9754]),  # Test hard traj 3
        # -1: torch.tensor([ 0.8099, -0.1148, -0.5144, -0.6101,  0.2817, -0.7839]),  # Test hard traj 3
        0: torch.tensor([1.0, 0.0, 0.0, 0.9659258262890682, 0.0, 0.25881904510252074]),  # Nice vid
        -1: torch.tensor([-1.00000000e+00, -1.22464680e-16,  2.46519033e-32, -2.22044605e-16,1.22464680e-16, -1.00000000e+00]),  # Nice vid
        # -1: torch.tensor([1.0, 0.0, 0.0, 2.220446049250313e-16, 0.0, 1.0]),  # Paralelle
        # 0: torch.tensor([-0.2452,  0.6542,  0.5221, -0.5328,  0.8169,  0.5369]),
        # -1: torch.tensor([ 0.1286,  0.5054,  0.8372, -0.5165,  0.5316,  0.6912]),
    }
else:
    raise ValueError('Wrong dataset name.')
cond_copy = copy.deepcopy(cond)

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
import random
random.seed(seed)

_, samples = policy(cond, batch_size=1, verbose=args.verbose, return_chain=True)

chains = samples.chains[0]
# _, chains = dataset.env.projection(None, torch.from_numpy(chains))
# chains = chains.numpy()
# Project last step to manifold if model is not using manifold projections

chains_torch = torch.from_numpy(chains)

# Make sure the last trajectory is on the manifold
_, chains_torch[-1] = dataset.env.projection(None, chains_torch[-1].unsqueeze(0))
# Reapply conditioning
chains_torch[-1, 0] = cond[0]
chains_torch[-1, -1] = cond[-1]

diffuser_dist, _ = dataset.env.score(chains_torch[-1].unsqueeze(0))

chains = chains_torch.numpy()

save_base = os.path.join(args.loadbase, args.dataset, args.diffusion_loadpath)
show_diffusion(renderer,
               chains,
               savebase=save_base,
               filename='denoising.mp4',
               n_repeat=10,
               fps=5,
               )

# Visualize the diffusion process
traj = dataset.env.interpolate(cond_copy[0].cpu().numpy(), cond_copy[-1].cpu().numpy())
traj = np.expand_dims(traj, 0)
expert_dist, _ = dataset.env.score(torch.from_numpy(traj))

# Add dummy actions
dummy_actions = np.zeros((traj.shape[0], traj.shape[1], diffusion.action_dim))
traj = np.concatenate((dummy_actions, traj), axis=-1)
traj[:, :, diffusion.action_dim:] =  diffusion.normalizer.normalize(traj[:, :, diffusion.action_dim:], "observations")
traj[:, :, :diffusion.action_dim] =  diffusion.normalizer.normalize(traj[:, :, :diffusion.action_dim], "actions")
traj = torch.from_numpy(traj) # Get first trajectory
traj = traj.to(diffusion.betas.device)
t = torch.Tensor([diffusion.n_timesteps]).long().to(traj.device)
trajs = diffusion.q_sample(traj, t=t, return_chain=True).cpu().numpy()
trajs = diffusion.normalizer.unnormalize(trajs[:, :, diffusion.action_dim:], "observations")
# trajs = trajs[:, :-1, :]  # Trim trajectory here to remove padding


# Fix start and goal
trajs[:, 0, :] = trajs[0, 0, :]
trajs[:, -1, :] = trajs[0, -1, :]


show_diffusion(renderer,
               trajs,
               savebase=save_base,
               filename='diffusion.mp4',
               n_repeat=10,
               fps=5,
               save_im=False,
               )

print(f"Diffuser dist : {diffuser_dist.item():.4f}")
print(f"Expert   dist : {expert_dist.item():.4f}")