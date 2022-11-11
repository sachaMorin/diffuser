import gym
import numpy as np

from gym import spaces
import matplotlib.pyplot as plt


def angle_to_polar_t2(theta):
    theta = np.repeat(theta, 2, axis=-1)
    theta[..., [0, 2]] = np.cos(theta[..., [0, 2]])
    theta[..., [1, 3]] = np.sin(theta[..., [1, 3]])
    return theta


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

    def step(self, action):
        action= np.array(action)
        action = np.clip(action, a_min=0.0, a_max=2 * np.pi)
        self.state = (self.state + action) % (2 * np.pi)
        self.state_polar = angle_to_polar_t2(self.state)

        terminated = self._terminal()
        reward = -1.0 if not terminated else 0.0

        # Save state
        self.state_buffer.append(self.state)

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
        self.state_buffer = [self.state]

        return self._get_obs()

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
        pass
        # c = plt.Circle((0, 0), 1.00, color='k', fill=False)
        # fig, ax = plt.subplots()
        # ax.set_xlim((-1.1, 1.1))
        # ax.set_ylim((-1.1, 1.1))
        # ax.add_patch(c)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # ax.axis('off')
        # ax.set_aspect('equal', adjustable='box')
        #
        # # Use provided observations or own state buffer
        # if observations is None:
        #     traj = np.concatenate(self.state_buffer)
        #     polar_coords = angle_to_polar(traj)
        #     goal = angle_to_polar(self.goal)[0]
        # else:
        #     polar_coords = observations
        #     goal = observations[-1]
        #
        # ax.plot(*polar_coords.T, c='b')
        # ax.scatter(*goal, c='r', s=100)
        # ax.scatter(*polar_coords[0], c='m', s=100)
        #
        # # Image from plot
        # ax.axis('off')
        # fig.tight_layout(pad=0)
        #
        # # To remove the huge white borders
        # ax.margins(0)
        #
        # fig.canvas.draw()
        # im = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # im = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #
        # plt.close()
        #
        # return im

env = T2(seed=123)
obs = env.reset()
print(env.state)
obs = env.step([np.pi, np.pi])
print(env.state)


# env = S1(seed=42)
# env.get_dataset(render=True)
