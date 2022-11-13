import gym
import numpy as np

from gym import spaces
import matplotlib.pyplot as plt

from diffuser.environments.utils import surface_plot, triu_plot, ManifoldPlanner


class T2(gym.Env):
    metadata = {'render.modes': ['human']}
    MAX_ANGLE_DEGREE = 15

    def __init__(self, seed=42, horizon=12):
        self.rng = np.random.RandomState(seed)
        self.name = 't2'
        self.max_angle_radians = self.MAX_ANGLE_DEGREE * np.pi / 180
        high_action = np.array([self.max_angle_radians, self.max_angle_radians], dtype=np.float32)
        self.action_space = spaces.Box(low=-high_action,
                                       high=high_action, dtype=np.float32)
        self.action_space.seed(seed)

        high = np.array([2 * np.pi, 2 * np.pi], np.float32)
        low = np.array([0.0, 0.0], np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space.seed(seed)

        self.state_intrinsic = None
        self.state_embedding = None
        self.goal_intrisic = None
        self.goal_embedding = None
        self.state_buffer = []
        self._max_episode_steps = 100
        self.t = 0

        # Radius
        self.R = 3
        self.r = 1

        # Expert planner for generating dataset
        self.planner = ManifoldPlanner(self, horizon=horizon)

    def random_step(self):
        action = self.action_space.sample()
        self.step(action)

    def step(self, action):
        action = np.array(action)
        action = np.clip(action, a_min=0.0, a_max=2 * np.pi)
        self.state_intrinsic = (self.state_intrinsic + action) % (2 * np.pi)
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

    def get_dataset(self, render=False, n_samples=1000):
        dataset = dict(observations=[], actions=[], rewards=[], terminals=[])
        for _ in range(n_samples):
            traj = self.planner.path()
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

    def render(self, observations=None, mode='human'):
        # Use provided observations or own state buffer
        if observations is None:
            traj = np.stack(self.state_buffer + [self.goal_embedding])
        else:
            traj = observations

        # Plot manifold mesh
        u = np.linspace(0, 2.0 * np.pi, endpoint=True, num=50)
        v = np.linspace(0, 2.0 * np.pi, endpoint=True, num=20)
        theta, phi = np.meshgrid(u, v)
        theta, phi = theta.flatten(), phi.flatten()
        angles = np.stack([theta, phi]).T
        fig, ax = triu_plot(self.intrinsic_to_3D(angles), angles)

        # Plot trajectory
        traj = self.embedding_to_3D(traj)
        surface_plot(traj, fig=fig, ax=ax)
        ax.scatter(*traj[-1], c='r', s=100)
        ax.scatter(*traj[0], c='m', s=100)

        # Remove border
        # fig.tight_layout(pad=0)
        ax.margins(0)

        fig.canvas.draw()
        im = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        im = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()

        return im
