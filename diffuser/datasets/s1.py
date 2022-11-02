import gym
import numpy as np

from gym import spaces
import matplotlib.pyplot as plt


def angle_to_polar(theta):
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.vstack((cos, sin)).T


class S1(gym.GoalEnv):
    metadata = {'render.modes': ['human']}
    MAX_ANGLE_DEGREE = 15

    def __init__(self, seed=42):
        self.name = 's1'
        self.max_angle_radians = self.MAX_ANGLE_DEGREE * np.pi / 180
        high_action = np.array([self.max_angle_radians], dtype=np.float32)
        self.action_space = spaces.Box(low=-high_action, high=high_action, dtype=np.float32)
        self.action_space.seed(seed)
        high = np.array([2 * np.pi], np.float32)
        low = np.array([0.0], np.float32)
        self.observation_space = gym.spaces.Dict(
            observation=spaces.Box(low=low, high=high, dtype=np.float32),
            achieved_goal=spaces.Box(low=low, high=high, dtype=np.float32),
            desired_goal=spaces.Box(low=low, high=high, dtype=np.float32),
        )
        self.observation_space.seed(seed)
        self.state = None
        self.goal = None
        self.state_buffer = []
        self._max_episode_steps = 100
        self.t = 0

    def step(self, action):
        action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)
        self.state = (self.state + action) % (2 * np.pi)
        terminated = self._terminal()
        reward = -1.0 if not terminated else 0.0

        # Save state
        self.state_buffer.append(self.state)

        self.t += 1

        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        state = angle_to_polar(self.state)
        goal = angle_to_polar(self.goal)
        return np.hstack((state, goal))[0]

    def _terminal(self):
        diff = min((2 * np.pi) - abs(self.state - self.goal), abs(self.state - self.goal))[0]
        return diff < .3 or self.t == self._max_episode_steps

    def reset(self):
        self.t = 0
        self.state = self.observation_space['observation'].sample()
        self.goal = self.observation_space['observation'].sample()
        self.state_buffer = [self.state]
        super().reset()

        return self._get_obs()

    def get_dataset(self, render=False):
        dataset = dict(observations=[], actions=[], rewards=[], terminals=[])
        for _ in range(1000):
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
                self.render()

        # Concat observations
        for k, v in dataset.items():
            dataset[k] = np.stack(v)


        return dataset

    def render(self, mode='human'):
        c = plt.Circle((0, 0), 1.00, color='k', fill=False)
        fig, ax = plt.subplots()
        ax.set_xlim((-1.1, 1.1))
        ax.set_ylim((-1.1, 1.1))
        ax.add_patch(c)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axis('off')
        ax.set_aspect('equal', adjustable='box')
        traj = np.concatenate(self.state_buffer)
        polar_coords = angle_to_polar(traj)
        ax.scatter(*polar_coords.T, c='b', s=100)
        ax.scatter(*angle_to_polar(self.goal).T, c='r', s=100)
        ax.scatter(*polar_coords[0], c='m', s=100)
        fig.show()





env = S1(seed=42)
env.get_dataset(render=True)
# for _ in range(10):
#     terminal = False
#     env.reset()
#     while not terminal:
#         action = abs(env.action_space.sample())
#         obs, _, terminal, _, _ = env.step(action)
#         print(obs)
#
#     env.render()
