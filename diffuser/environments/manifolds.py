import torch
from abc import ABC

import gym
import numpy as np

from gym import spaces
import matplotlib.pyplot as plt

from diffuser.environments.utils import surface_plot, triu_plot

import contracts
from geometry.manifolds.sphere import Sphere
from geometry.manifolds.torus import Torus

# Disable PyGeometry contracts for performance + avoid errors in Sphere due to
# belong_ts
contracts.disable_all()

# Define dataset sizes
SMALL = 100
MEDIUM = 1000
LARGE = 5000


class ManifoldEnv(gym.Env):
    n_samples = 200

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
        self.goal_intrinsic = None
        self.goal_embedding = None
        self.state_buffer = []
        self._max_episode_steps = 100
        self.t = 0

    def seed(self, seed=None):
        super().seed(seed)
        self.rng = np.random.RandomState(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.random_state = seed

    def random_step(self):
        action = self.action_space.sample()
        self.step(action)

    def _update_state_intrinsic(self, action):
        raise NotImplementedError()

    def step(self, action):
        action = np.array(action)
        self._update_state_intrinsic(action)
        self.state_embedding = self.intrinsic_to_embedding(self.state_intrinsic)

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

    def intrinsic_to_embedding(self, intrinsic):
        raise NotImplementedError()

    def embedding_to_3D(self, embedding):
        # By default, assume embedding is 3D and used for plotting, otherwise override
        return embedding

    def intrinsic_to_3D(self, intrinsic):
        return self.embedding_to_3D(self.intrinsic_to_embedding(intrinsic))

    def reset(self):
        self.t = 0
        self.state_intrinsic = self.observation_space.sample()
        self.state_embedding = self.intrinsic_to_embedding(self.state_intrinsic)
        self.goal_intrinsic = self.observation_space.sample()
        self.goal_embedding = self.intrinsic_to_embedding(self.goal_intrinsic)
        self.state_buffer = [self.state_embedding]

        return self._get_obs()

    def sample(self, n_samples):
        """Return sampled embeddings."""
        raise NotImplementedError()

    def get_dataset(self):
        dataset = dict(observations=[], actions=[], rewards=[], terminals=[])
        for i in range(self.n_samples):
            # traj = []

            # Sample random path
            # while len(traj) < self.horizon:
            #     traj = self.planner.path()
            start, goal = self.sample(2)
            traj = self.interpolate(start, goal)

            # Lower the resolution down to horizon
            # if traj.shape[0] > 2 * self.horizon:
            #     step = traj.shape[0] // self.horizon
            #     traj = traj[::step]
            #
            # # Then just pick first self.horizon observations
            # traj = traj[:self.horizon]

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

    def interpolate(self, start, goal, times=None):
        # If times is none, it should default to self.get_horizon_times
        raise NotImplementedError()

    def get_horizon_times(self):
        return np.linspace(0, 1, num=self.horizon, endpoint=True)

    def interpolate_batch(self, x_0, x_t, times):
        # Naive numpy iterative interpolation for now
        device = x_0.device
        x_0, x_t = x_0.cpu().double().numpy(), x_t.cpu().double().numpy()
        times = times.cpu().numpy()

        batch_size = x_0.shape[0]
        result = list()
        for i in range(batch_size):
            result.append(self.interpolate(x_0[i], x_t[i], np.array([times[i]])))

        result = np.concatenate(result, axis=0)

        return torch.from_numpy(result).to(device)

    def get_intrinsic_mesh(self):
        raise NotImplementedError()

    def expand_traj(self, traj, steps=100):
        """Interpolate in Euclidean space to increase trajectory resolution."""
        n, d = traj.shape
        deltas = traj[1:] - traj[:-1]
        deltas = np.repeat(deltas, steps, axis=0)
        traj = np.repeat(traj[:-1], steps, axis=0)
        coefs = np.tile(np.linspace(0, 1, num=steps), n - 1)

        return traj + deltas * coefs.reshape((-1, 1))

    def render(self, observations=None, mode='human'):
        # Use provided observations or own state buffer
        if observations is None:
            traj = np.stack(self.state_buffer + [self.goal_embedding])
        else:
            traj = observations

        # Plot manifold mesh
        mesh = self.get_intrinsic_mesh()
        fig, ax = triu_plot(self.intrinsic_to_3D(mesh), mesh)

        # Plot trajectory
        _, traj = self.projection(None, traj)
        traj = self.expand_traj(traj)
        _, traj = self.projection(None, traj)
        traj = self.embedding_to_3D(traj)
        surface_plot(traj, fig=fig, ax=ax, cmap="plasma")
        ax.scatter(*traj[0], c='#0D0887', s=100, linewidths=1, edgecolors='k')
        ax.scatter(*traj[-1], c='#F0F921', s=100, linewidths=1, edgecolors='k')

        # Remove border
        ax.margins(0)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_zticklabels([])

        # ax.plot3D(*x.T, 'magma', c=np.arange(x.shape[0]), linewidth=2)

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

    def sample_torch(self, n_samples, device='cpu'):
        samples = self.sample(n_samples)
        return torch.from_numpy(samples).to(device)

    def score(self, R):
        """Score trajectories.

        We aim for short, constant velocity trajectories.
        R is of size [batch_size, T, n_dim].

        Given a start, a goal, T steps and an optimal dist d^* between the start and
        goal, each step in the trajectory should be of length d^*/T, no more, no less.
        Note that by design, all trajectories already start and end at the expected coordinates."""
        batch_size, T, n_dim = R.shape
        start_goal = R[:, [0, -1], :]
        dist = self.seq_geodesic_distance(R)
        optimal_dist = self.seq_geodesic_distance(start_goal)
        optimal_step_size = optimal_dist / T

        # Return length of trajectories on the manifold
        # And how well the velocity constraint is satisfied
        return dist.sum(dim=-1), ((dist - optimal_step_size) ** 2).mean(dim=-1)


class T2(ManifoldEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, R=3, r=1, max_angle_degree=15, **kwargs):
        self.R = R  # Only used for visualization
        self.r = r  # Only used for visualization
        self.max_angle_radians = max_angle_degree * np.pi / 180

        # Boundaries
        high_action = np.array([self.max_angle_radians, self.max_angle_radians], dtype=np.float32)
        low_action = -high_action
        high_obs = np.array([2 * np.pi, 2 * np.pi], np.float32)
        low_obs = np.array([0.0, 0.0], np.float32)

        super().__init__(low_obs=low_obs, high_obs=high_obs, low_action=low_action, high_action=high_action, **kwargs)

        self.name = 't2'
        self.manifold = Torus(2)

    def _update_state_intrinsic(self, action):
        self.state_intrinsic = (self.state_intrinsic + action) % (2 * np.pi)

    def intrinsic_to_embedding(self, intrinsic):
        # 2 angles to R^4
        emb = np.repeat(intrinsic, 2, axis=-1)
        emb[..., [0, 2]] = np.cos(emb[..., [0, 2]])
        emb[..., [1, 3]] = np.sin(emb[..., [1, 3]])
        return emb

    def embedding_to_intrinsic(self, embedding):
        # R^4 to 2 angles
        angle_1 = np.arctan2(embedding[:, 1], embedding[:, 0])
        angle_2 = np.arctan2(embedding[:, 3], embedding[:, 2])

        # Map back to [0, 2pi]
        return np.vstack((angle_1, angle_2)).T

    def embedding_to_3D(self, embedding):
        cos_phi, sin_phi, cos_theta, sin_theta = embedding.T
        emb = np.zeros((embedding.shape[0], 3))
        emb[:, 0] = (self.R + self.r * cos_theta) * cos_phi
        emb[:, 1] = (self.R + self.r * cos_theta) * sin_phi
        emb[:, 2] = self.r * sin_theta
        return emb

    def sample(self, n_samples):
        data = self.rng.normal(size=(1, n_samples, 4))
        _, data = self.projection(None, data)
        return data[0]

    def get_intrinsic_mesh(self):
        u = np.linspace(0, 2.0 * np.pi, endpoint=True, num=50)
        v = np.linspace(0, 2.0 * np.pi, endpoint=True, num=20)
        theta, phi = np.meshgrid(u, v)
        theta, phi = theta.flatten(), phi.flatten()
        angles = np.stack([theta, phi]).T
        return angles

    def projection(self, actions, obs):
        # Autobatch
        is_2D = obs.ndim == 2
        if is_2D:
            t, d = obs.shape
            obs = obs.reshape((1, t, d))

        b, t, d = obs.shape
        obs = obs.reshape((b, t, 2, d // 2))

        if torch.is_tensor(obs):
            # Torch
            obs = torch.nn.functional.normalize(obs, p=2, dim=-1)
        else:
            # Numpy 
            norm = np.linalg.norm(obs, axis=-1, keepdims=True)
            norm[norm == 0.0] = 1.0
            obs /= norm

        obs = obs.reshape((b, t, d))

        # Autobatch
        if is_2D:
            obs = obs[0]
        return actions, obs

    def seq_geodesic_distance(self, x):
        # See https://stackoverflow.com/questions/52210911/great-circle-distance-between-two-p-x-y-z-points-on-a-unit-sphere#:~:text=the%20distance%20on%20the%20great,%3D%202*phi*R%20.
        # Given a b sequences of t 2-polar coordinates (b x t x 4 Tensor)
        # Return geodesic distance of the trajectory (b Tensor)
        x = x.double()
        b, t, d = x.shape
        x = self.embedding_to_intrinsic(x.reshape((b * t, d)).cpu().numpy()).reshape((b, t, 2))
        from_ = x[:, :-1, :]
        to = x[:, 1:, :]
        result = torch.zeros(b, t - 1)

        for i in range(b):
            for j, (from_i, to_i) in enumerate(zip(from_[i], to[i])):
                dist = self.manifold.distance(from_i, to_i)
                result[i,  j] += dist
                # print(dist)

        return result

    def interpolate(self, start, goal, times=None):
        start, goal = self.embedding_to_intrinsic(np.vstack((start, goal)))

        # Then use same logic as S2
        angles = S2.interpolate(self, start, goal, times)

        return self.intrinsic_to_embedding(angles)


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
        self.manifold = Sphere(2)  # PyGeometric Object

    def _update_state_intrinsic(self, action):
        self.state_intrinsic = (self.state_intrinsic + action) % self.high_obs

    def intrinsic_to_embedding(self, intrinsic):
        if intrinsic.ndim == 1:
            intrinsic = intrinsic.reshape((1, -1))

        phi, theta = intrinsic.T
        emb = np.zeros((intrinsic.shape[0], 3))
        emb[:, 0] = np.sin(theta) * np.cos(phi)
        emb[:, 1] = np.sin(theta) * np.sin(phi)
        emb[:, 2] = np.cos(theta)
        return np.squeeze(emb)

    def interpolate(self, start, goal, times=None):
        if times is None:
            times = self.get_horizon_times()
            result = [start]

            # We add start and goal manually to save time and avoid numerical errors
            times = times[1:-1]  # We add start and goal manually to save time
            for t in times:
                result.append(self.manifold.geodesic(start, goal, t))
            result.append(goal)
        else:
            # Only iterate provided times
            result = []
            for t in times:
                result.append(self.manifold.geodesic(start, goal, t))

        return np.vstack(result)

    def sample(self, n_samples):
        samples = self.rng.normal(size=(n_samples, 3))
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        return samples

    def get_intrinsic_mesh(self):
        u = np.linspace(0, 2.0 * np.pi, endpoint=True, num=50)
        v = np.linspace(0, np.pi, endpoint=True, num=50)
        theta, phi = np.meshgrid(u, v)
        theta, phi = theta.flatten(), phi.flatten()
        angles = np.stack([theta, phi]).T
        return angles

    def projection(self, actions, obs):
        if torch.is_tensor(obs):
            return actions, torch.nn.functional.normalize(obs, p=2, dim=-1)
        else:
            #  Numpy array
            norm = np.linalg.norm(obs, axis=-1, keepdims=True)
            mask = norm == 0
            norm[mask] = 1

            obs = obs / norm

            return actions, obs

    def seq_geodesic_distance(self, x):
        # See https://stackoverflow.com/questions/52210911/great-circle-distance-between-two-p-x-y-z-points-on-a-unit-sphere#:~:text=the%20distance%20on%20the%20great,%3D%202*phi*R%20.
        # Given a b sequences of t spherical coordinates (b x t x 3 Tensor)
        # Return geodesic distance of the trajectory (b Tensor)
        x = x.double()
        from_ = x[:, :-1, :]
        to = x[:, 1:, :]
        chordal_dist = torch.linalg.norm(to - from_, dim=-1)
        half_angle = torch.arcsin(chordal_dist / 2)

        return 2 * half_angle


# Define datasets with different sizes
class T2small(T2):
    n_samples = SMALL


class T2medium(T2):
    n_samples = MEDIUM


class T2large(T2):
    n_samples = LARGE


class S2small(S2):
    n_samples = SMALL


class S2medium(S2):
    n_samples = MEDIUM


class S2large(S2):
    n_samples = LARGE


if __name__ == '__main__':
    env = S2()
    env_small = S2small()
    # env = T2()
    dataset = env_small.get_dataset()
    import pdb;

    pdb.set_trace()
    #
    # # Render some planner trajectories
    # for i in range(20):
    #     im = env.render(torch.from_numpy(dataset['observations'][i * 12: (i + 1) * 12]))
    #     plt.imshow(im)
    #     plt.show()
