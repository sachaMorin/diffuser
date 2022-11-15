import torch
from abc import ABC

import gym
import numpy as np

from gym import spaces
import matplotlib.pyplot as plt

from diffuser.environments.utils import surface_plot, triu_plot, ManifoldPlanner


class ManifoldEnv(gym.Env):
    def __init__(self, low_obs, high_obs, low_action, high_action, seed=42, horizon=12):
        self.name = None
        self.horizon = horizon
        self.random_state = seed
        self.rng = np.random.RandomState(seed)

        # Action space
        self.action_space = spaces.Box(low=np.array(low_action),
                                       high=np.array(high_action), dtype=np.float32)
        self.action_space.seed(seed)

        # Observation space
        self.observation_space = spaces.Box(low=np.array(low_obs),
                                            high=np.array(high_obs), dtype=np.float32)
        self.observation_space.seed(seed)

        # Memory
        self.state_intrinsic = None
        self.state_embedding = None
        self.goal_intrisic = None
        self.goal_embedding = None
        self.state_buffer = []
        self._max_episode_steps = 100
        self.t = 0

        # Manifold planner
        self.planner = self.get_planner(self.random_state)

    def seed(self, seed=None):
        super().seed(seed)
        self.rng = np.random.RandomState(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.random_state = seed
        self.planner = self.get_planner(self.random_state)

    def get_planner(self, random_state):
        return ManifoldPlanner(self, random_seed=random_state)

    def random_step(self):
        action = self.action_space.sample()
        self.step(action)

    def _update_state_intrinsic(self, action):
        raise NotImplementedError()

    def step(self, action):
        action = np.array(action)
        self._update_state_intrinsic(action)
        self.state_embedding = self.intrisic_to_embedding(self.state_intrinsic)

        terminated = self._terminal()
        reward = -1.0 if not terminated else 0.0

        # Save state
        self.state_buffer.append(self.state_embedding)

        self.t += 1

        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        return self.state_embedding

    def _terminal(self):
        return np.linalg.norm(self.goal_embedding - self.state_embedding) < 1e-2 or self.t == self._max_episode_steps

    def intrisic_to_embedding(self, intrinsic):
        raise NotImplementedError()

    def embedding_to_3D(self, embedding):
        # By default, assume embedding is 3D and used for plotting, otherwise override
        return embedding

    def intrinsic_to_3D(self, intrinsic):
        return self.embedding_to_3D(self.intrisic_to_embedding(intrinsic))

    def reset(self):
        self.t = 0
        self.state_intrinsic = self.observation_space.sample()
        self.state_embedding = self.intrisic_to_embedding(self.state_intrinsic)
        self.goal_intrisic = self.observation_space.sample()
        self.goal_embedding = self.intrisic_to_embedding(self.goal_intrisic)
        self.state_buffer = [self.state_embedding]

        return self._get_obs()

    def sample(self, n_samples):
        """Return sampled embeddings."""
        raise NotImplementedError()

    def get_dataset(self, n_samples=1000):
        dataset = dict(observations=[], actions=[], rewards=[], terminals=[])
        for i in range(n_samples):
            traj = []

            # Sample random path
            while len(traj) < self.horizon:
                traj = self.planner.path()

            # Lower the resolution down to horizon
            if traj.shape[0] > 2 * self.horizon:
                step = traj.shape[0] // self.horizon
                traj = traj[::step]

            # Then just pick first self.horizon observations
            traj = traj[:self.horizon]

            # Fillers
            actions = np.zeros((traj.shape[0], 2))
            terminal = np.zeros(traj.shape[0])
            terminal[-1] = True
            reward = -1 * np.ones(traj.shape[0])
            reward[-1] = 0.0
            dataset['observations'].append(traj)
            dataset['actions'].append(actions)
            dataset['rewards'].append(reward)
            dataset['terminals'].append(terminal)

        # Concat observations
        for k, v in dataset.items():
            dataset[k] = np.concatenate(v)

        return dataset

    def get_intrisic_mesh(self):
        raise NotImplementedError()

    def render(self, observations=None, mode='human'):
        # Use provided observations or own state buffer
        if observations is None:
            traj = np.stack(self.state_buffer + [self.goal_embedding])
        else:
            traj = observations

        # Plot manifold mesh
        mesh = self.get_intrisic_mesh()
        fig, ax = triu_plot(self.intrinsic_to_3D(mesh), mesh)

        # Plot trajectory
        traj = self.embedding_to_3D(traj)
        surface_plot(traj, fig=fig, ax=ax)
        ax.scatter(*traj[-1], c='r', s=100)
        ax.scatter(*traj[0], c='m', s=100)

        # Remove border
        ax.margins(0)

        fig.canvas.draw()
        im = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        im = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()

        return im

    def projection(self, actions, obs):
        """Project x onto the manifold."""
        raise NotImplementedError()

    def seq_geodesic_distance(self, x):
        """Compute geodesic distance of sequences."""
        raise NotImplementedError()


class T2(ManifoldEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, R=3, r=1, max_angle_degree=15, **kwargs):
        self.R = R
        self.r = r
        self.max_angle_radians = max_angle_degree * np.pi / 180

        # Boundaries
        high_action = np.array([self.max_angle_radians, self.max_angle_radians], dtype=np.float32)
        low_action = -high_action
        high_obs = np.array([2 * np.pi, 2 * np.pi], np.float32)
        low_obs = np.array([0.0, 0.0], np.float32)

        super().__init__(low_obs=low_obs, high_obs=high_obs, low_action=low_action, high_action=high_action, **kwargs)

        self.name = 't2'

    def _update_state_intrinsic(self, action):
        self.state_intrinsic = (self.state_intrinsic + action) % (2 * np.pi)

    def intrisic_to_embedding(self, intrinsic):
        emb = np.repeat(intrinsic, 2, axis=-1)
        emb[..., [0, 2]] = np.cos(emb[..., [0, 2]])
        emb[..., [1, 3]] = np.sin(emb[..., [1, 3]])
        return emb

    def embedding_to_3D(self, embedding):
        cos_phi, sin_phi, cos_theta, sin_theta = embedding.T
        emb = np.zeros((embedding.shape[0], 3))
        emb[:, 0] = (self.R + self.r * cos_theta) * cos_phi
        emb[:, 1] = (self.R + self.r * cos_theta) * sin_phi
        emb[:, 2] = self.r * sin_theta
        return emb

    def sample(self, n_samples):
        result = []
        n = 0

        # Rejection sampling to achieve uniform sampling on the Torus
        while n < n_samples:
            phi, theta, w = self.rng.uniform(size=(3, 1000))
            phi *= 2 * np.pi
            theta *= 2 * np.pi

            c = self.R + self.r * np.cos(theta)

            # Rejection sampling
            accepted = w < c / (self.R + self.r)
            n += accepted.sum()
            result.append(np.stack([theta, phi]).T)

        angles = np.concatenate(result)[:n_samples]

        return self.intrisic_to_embedding(angles)

    def get_intrisic_mesh(self):
        u = np.linspace(0, 2.0 * np.pi, endpoint=True, num=50)
        v = np.linspace(0, 2.0 * np.pi, endpoint=True, num=20)
        theta, phi = np.meshgrid(u, v)
        theta, phi = theta.flatten(), phi.flatten()
        angles = np.stack([theta, phi]).T
        return angles

    # def projection(self, actions, obs):
    #     b, t, d = obs.shape
    #     obs = obs.reshape((b, t, 2, d//2))
    #     # Ignore perfect 0s, they usually indicate padding
    #     not_zero = ~(obs == 0).all(dim=-1)
    #
    #     obs[not_zero] /= torch.linalg.norm(obs[not_zero], dim=-1, keepdim=True)
    #     obs = obs.reshape((b, t, d))


class S2(ManifoldEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_angle_degree=15, **kwargs):
        self.max_angle_radians = max_angle_degree * np.pi / 180

        # Boundaries
        high_action = np.array([self.max_angle_radians, self.max_angle_radians], dtype=np.float32)
        low_action = -high_action
        self.high_obs = np.array([2 * np.pi, np.pi], np.float32)
        low_obs = np.array([0.0, 0.0], np.float32)

        super().__init__(low_obs=low_obs, high_obs=self.high_obs, low_action=low_action, high_action=high_action,
                         **kwargs)

        self.name = 't2'

    def _update_state_intrinsic(self, action):
        self.state_intrinsic = (self.state_intrinsic + action) % self.high_obs

    def intrisic_to_embedding(self, intrinsic):
        if intrinsic.ndim == 1:
            intrinsic = intrinsic.reshape((1, -1))

        phi, theta = intrinsic.T
        emb = np.zeros((intrinsic.shape[0], 3))
        emb[:, 0] = np.sin(theta) * np.cos(phi)
        emb[:, 1] = np.sin(theta) * np.sin(phi)
        emb[:, 2] = np.cos(theta)
        return np.squeeze(emb)

    def sample(self, n_samples):
        samples = self.rng.normal(size=(n_samples, 3))
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        return samples

    def get_intrisic_mesh(self):
        u = np.linspace(0, 2.0 * np.pi, endpoint=True, num=50)
        v = np.linspace(0, np.pi, endpoint=True, num=50)
        theta, phi = np.meshgrid(u, v)
        theta, phi = theta.flatten(), phi.flatten()
        angles = np.stack([theta, phi]).T
        return angles

    def projection(self, actions, obs):
        if torch.is_tensor(obs):
            norm = torch.linalg.norm(obs, dim=-1, keepdims=True)
        else:
            norm = np.linalg.norm(obs, axis=-1, keepdims=True)

        norm[norm == 0.0] = 1.00

        return actions, obs / norm

    def seq_geodesic_distance(self, x):
        # See https://stackoverflow.com/questions/52210911/great-circle-distance-between-two-p-x-y-z-points-on-a-unit-sphere#:~:text=the%20distance%20on%20the%20great,%3D%202*phi*R%20.
        # Given a b sequences of t spherical coordinates (b x t x 3 Tensor)
        # Return geodesic distance of the trajectory (b Tensor)
        x = x.double()
        from_ = x[:, :-1, :]
        to = x[:, 1:, :]
        chordal_dist = torch.linalg.norm(to - from_, dim=-1)
        half_angle = torch.arcsin(chordal_dist / 2)

        return (2 * half_angle).sum(dim=-1)
