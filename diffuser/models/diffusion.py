from collections import namedtuple
import numpy as np
import torch
from torch import nn

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

Sample = namedtuple('Sample', 'trajectories values chains')


@torch.no_grad()
def default_sample_fn(model, x, cond, t, **kwargs):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
                 loss_type='l1', clip_denoised=False, predict_epsilon=True,
                 action_weight=1.0, loss_discount=1.0, loss_weights=None, normalizer=None,
                 mask_action=False, projection_operator=None, manifold_diffuser_mode="no_projection"):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.normalizer = normalizer

        # Map actions to 0
        # Useful to only optimize trajectories and ignore actions
        self.mask_action = mask_action

        # Manifold Projection Settings
        self.manifold_diffuser_mode = manifold_diffuser_mode
        self.projection_operator = projection_operator
        if manifold_diffuser_mode == "start":
            # Only apply manifold projection to reconstructions of x_0
            # Diffusion is otherwise standard
            self.project_x_0 = True
            self.project_x_t = False
            self.project_diffusion = False
        elif manifold_diffuser_mode == "start_and_noise":
            # Apply manifold projection to reconstructions of x_0
            # Apply manifold projection to x_t after free diffusion in Euclidean space
            self.project_x_0 = True
            self.project_x_t = True
            self.project_diffusion = False
        elif manifold_diffuser_mode == "full":
            # Apply manifold projection throughout the diffusion process
            # This will interleave noise injection and projections
            # Can be quite slow if projections are costly
            self.project_x_0 = True
            self.project_x_t = True
            self.project_diffusion = True
        elif manifold_diffuser_mode == "no_projection":
            # No projection. This is standard diffuser
            self.project_x_0 = False
            self.project_x_t = False
            self.project_diffusion = False
        else:
            raise ValueError("Invalid value for manifold_diffuser_mode.")

        # Coefficients
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('sqrt_alphas', torch.sqrt(alphas))
        self.register_buffer('sqrt_one_minus_alphas', torch.sqrt(1 - alphas))
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

        # Make sure normalizer can also handle torch tensors
        self.normalizer.torchify(self.betas.device)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        # x_0
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.project_x_0:
            x_recon = self.projection(x_recon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        # x_{t-1}
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn,
                      **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        # Projection onto the manifold during sampling
        if self.project_diffusion or self.project_x_t:
            x = self.projection(x)

        x = apply_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, cond, t, **sample_kwargs)

            # Projection onto the manifold during sampling
            if self.project_diffusion or self.project_x_t:
                x = self.projection(x)
            if self.mask_action:
                x[:, :, :self.action_dim] = 0.0

            x = apply_conditioning(x, cond, self.action_dim)


            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)

        if self.project_x_0 and not self.project_diffusion:
            # If project_diffusion, x is already on the manifold
            x = self.projection(x)

        x = apply_conditioning(x, cond, self.action_dim)
        progress.stamp()

        # x, values = sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, **sample_kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None, return_chain=False):
        chain = None
        if noise is None:
            noise = torch.randn_like(x_start)

        if not self.project_diffusion and not return_chain:
            # Original Diffusion with jump gaussians
            sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
            )
            if self.mask_action:
                sample[:, :, :self.action_dim] = 0.0

            # Project noisy sample on manifold
            if self.project_x_t:
                sample = self.projection(sample)
        else:
            # Iterative Diffusion
            # If we return chain, we still use iterative diffusion. Useful for visualization
            sample = x_start

            if return_chain:
                chain = [sample]

            # Iterate over diffusion steps
            for t_i in range(t.max().item()):
                # Mask alphas where diffusion is done
                diffusion_is_done = t_i > t
                sqrt_alphas = self.sqrt_alphas[t_i] * torch.ones_like(t)
                sqrt_alphas[diffusion_is_done] = 1. # Multiplication Identity
                sqrt_one_minus_alphas = self.sqrt_one_minus_alphas[t_i] * torch.ones_like(t)
                sqrt_one_minus_alphas[diffusion_is_done] = 0. # Addition Identity

                # Reshape coefficients for proper broadcast
                sqrt_alphas = sqrt_alphas.reshape((-1, 1, 1))
                sqrt_one_minus_alphas = sqrt_one_minus_alphas.reshape((-1, 1, 1))

                # Add noise
                noise = torch.randn_like(sample)
                sample = sqrt_alphas * sample + sqrt_one_minus_alphas * noise

                # Project to manifold during diffusion
                if self.project_diffusion:
                    sample = self.projection(sample)
                if self.mask_action:
                    sample[:, :, :self.action_dim] = 0.0

                # Save chain
                if return_chain:
                    chain.append(sample.clone())

            if self.project_x_t and not self.project_diffusion:
                # If self.project_diffusion, x_t is already on the manifold
                sample = self.projection(sample)
                if return_chain:
                    chain[-1] = self.projection(chain[-1])

            if return_chain:
                sample = torch.cat(chain)

        return sample

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        if self.project_x_0:
            x_recon = self.projection(x_recon)

        if self.mask_action:
            x_recon[:, :, :self.action_dim] = 0.0

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
            raise NotImplementedError("Current manifold implementation does not account for epsilon. Do not use.")
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, *args):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, *args, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)

    def projection(self, x):
        if self.manifold_diffuser_mode == "no_projection":
            raise Exception("self.manifold_diffuser_mode is set to no_projection. I should not be called.")

        # Unormalize state
        obs = x[:, :, self.action_dim:]
        obs = self.normalizer.unnormalize(obs, "observations")

        # Unormalize actions
        actions = x[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, "actions")

        # Manifold projection
        actions, obs = self.projection_operator(actions, obs)

        # Renormalize state
        obs = self.normalizer.normalize(obs, "observations")
        x[:, :, self.action_dim:] = obs

        # Renormalize actions
        actions = self.normalizer.normalize(actions, "actions")
        x[:, :, :self.action_dim] = actions

        return x


class ValueDiffusion(GaussianDiffusion):

    def p_losses(self, x_start, cond, target, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        pred = self.model(x_noisy, cond, t)

        loss, info = self.loss_fn(pred, target)
        return loss, info

    def forward(self, x, cond, t):
        return self.model(x, cond, t)
