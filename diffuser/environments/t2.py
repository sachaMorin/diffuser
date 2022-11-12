import gym
import numpy as np

from gym import spaces
import matplotlib.pyplot as plt

from diffuser.environments.utils import set_axes_equal, surface_plot, triu_plot


def angle_to_polar_t2(theta):
    theta = np.repeat(theta, 2, axis=-1)
    theta[..., [0, 2]] = np.cos(theta[..., [0, 2]])
    theta[..., [1, 3]] = np.sin(theta[..., [1, 3]])
    return theta


def polar_to_3D(polar, R, r):
    cos_phi, sin_phi, cos_theta, sin_theta = polar.T
    emb = np.zeros((polar.shape[0], 3))
    emb[:, 0] = (R + r * cos_theta) * cos_phi
    emb[:, 1] = (R + r * cos_theta) * sin_phi
    emb[:, 2] = r * sin_theta
    return emb


def angle_to_3D(angle, R, r):
    return polar_to_3D(angle_to_polar_t2(angle), R, r)


class T2(gym.Env):
    metadata = {'render.modes': ['human']}
    MAX_ANGLE_DEGREE = 15

    def __init__(self, seed=42):
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

        self.state = None
        self.state_polar = None
        self.goal = None
        self.goal_polar = None
        self.state_buffer = []
        self._max_episode_steps = 100
        self.t = 0

        # Radius
        self.R = 3
        self.r = 1

    def random_step(self):
        action = self.action_space.sample()
        self.step(action)

    def step(self, action):
        action = np.array(action)
        action = np.clip(action, a_min=0.0, a_max=2 * np.pi)
        self.state = (self.state + action) % (2 * np.pi)
        self.state_polar = angle_to_polar_t2(self.state)

        terminated = self._terminal()
        reward = -1.0 if not terminated else 0.0

        # Save state
        self.state_buffer.append(self.state_polar)

        self.t += 1

        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        return self.state_polar

    def _terminal(self):
        return np.linalg.norm(self.goal_polar - self.state_polar) < 1e-2 or self.t == self._max_episode_steps

    def reset(self):
        self.t = 0
        self.state = self.observation_space.sample()
        self.state_polar = angle_to_polar_t2(self.state)
        self.goal = self.observation_space.sample()
        self.goal_polar = angle_to_polar_t2(self.goal)
        self.state_buffer = [self.state_polar]

        return self._get_obs()

    def sample(self, n_samples):
        result = []
        n = 0

        # Rejection sampling to achieve uniform sampling on the Torus
        while n < n_samples:
            phi, theta, w = np.random.uniform(size=(3, 5000))
            phi *= 2 * np.pi
            theta *= 2 * np.pi

            c = self.R + self.r * np.cos(theta)

            # Rejection sampling
            accepted = w < c / (self.R + self.r)
            n += accepted.sum()
            result.append(np.stack([theta, phi]).T)

        angles = np.concatenate(result)[:n_samples]

        return angle_to_polar_t2(angles), angles

    def get_dataset(self, render=False, n_samples=1000):
        dataset = dict(observations=[], actions=[], rewards=[], terminals=[])
        for _ in range(n_samples):
            terminal = False
            obs = self.reset()
            delta_angle = (self.goal - self.state) % (2 * np.pi) - np.pi
            sign = 1 if delta_angle < 0 else -1
            while not terminal:
                action = sign * abs(self.action_space.sample())
                next_obs, reward, terminal, _, _ = self.step(action)
                dataset['observations'].append(obs)
                dataset['actions'].append(action[0])
                dataset['rewards'].append(reward)
                dataset['terminals'].append(terminal)

                obs = next_obs

            if render:
                im = self.render()
                plt.imshow(im)
                plt.show()

        # Concat observations
        for k, v in dataset.items():
            dataset[k] = np.stack(v)

        return dataset

    def render(self, observations=None, mode='human'):
        # Use provided observations or own state buffer
        if observations is None:
            traj = np.stack(self.state_buffer + [self.goal_polar])
        else:
            traj = observations

        # Plot manifold mesh
        u = np.linspace(0, 2.0 * np.pi, endpoint=True, num=50)
        v = np.linspace(0, 2.0 * np.pi, endpoint=True, num=20)
        theta, phi = np.meshgrid(u, v)
        theta, phi = theta.flatten(), phi.flatten()
        angles = np.stack([theta, phi]).T
        fig, ax = triu_plot(angle_to_3D(angles, self.R, self.r), angles)

        # Plot trajectory
        traj = polar_to_3D(traj, self.R, self.r)
        surface_plot(traj[:-1], fig=fig, ax=ax)
        ax.scatter(*traj[-1], c='r', s=100)
        ax.scatter(*traj[0], c='m', s=100)


        # Remove border
        fig.tight_layout(pad=0)
        ax.margins(0)

        fig.canvas.draw()
        im = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        im = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()

        return im


env = T2(seed=123)
obs = env.reset()
obs = np.stack([np.linspace(-.75 * np.pi, 2 * -np.pi, num=100), 0 * np.ones(100)]).T
# for _ in range(0):
#     env.step([np.pi/8, np.pi/24])
im = env.render(angle_to_polar_t2(obs))
plt.imshow(im)
plt.show()

# env = S1(seed=42)
# env.get_dataset(render=True)
