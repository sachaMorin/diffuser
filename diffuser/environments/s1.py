import gym
import numpy as np

from gym import spaces
import matplotlib.pyplot as plt


def angle_to_polar(theta):
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.vstack((cos, sin)).T


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


class S1(gym.Env):
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

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
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
        # goal = angle_to_polar(self.goal)
        return state[0]

    def _terminal(self):
        diff = min((2 * np.pi) - abs(self.state - self.goal), abs(self.state - self.goal))[0]
        return diff < .3 or self.t == self._max_episode_steps

    def reset(self):
        self.t = 0
        self.state = self.observation_space.sample()
        self.goal = self.observation_space.sample()
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

        # Use provided observations or own state buffer
        if observations is None:
            traj = np.concatenate(self.state_buffer)
            polar_coords = angle_to_polar(traj)
            goal = angle_to_polar(self.goal)[0]
        else:
            polar_coords = observations
            goal = observations[-1]

        ax.plot(*polar_coords.T, c='b')
        ax.scatter(*goal, c='r', s=100)
        ax.scatter(*polar_coords[0], c='m', s=100)

        # Image from plot
        ax.axis('off')
        fig.tight_layout(pad=0)

        # To remove the huge white borders
        ax.margins(0)

        fig.canvas.draw()
        im = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        im = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()

        return im

# env = S1(seed=42)
# env.get_dataset(render=True)
