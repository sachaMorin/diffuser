import pdb
import torch

# import diffuser.sampling as sampling
import diffuser.utils as utils
import matplotlib.pyplot as plt


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'S1-v1'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan')


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


for _ in range(10):
    start = torch.tensor([[1.0, 0.0]])
    stop = torch.tensor([[0.0, 1.0]])

    conditions = {0: start, -1: stop}
    samples = diffusion(conditions)
    trajectories = utils.to_np(samples.trajectories)

    # First column is action. Get coordinates
    trajectory = trajectories[0, :, 1:]

    im = renderer.render(trajectory)

    plt.imshow(im)
    plt.show()