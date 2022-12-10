import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

import einops
from diffuser.environments.manifolds import ManifoldEnv
from .manifolds import S2

from diffuser.environments.utils import surface_plot, triu_plot


class SO3(ManifoldEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_angle_degree=15, **kwargs):
        self.max_angle_radians = max_angle_degree * np.pi / 180

        # Boundaries
        # Represent state space and actions as euler angles
        # We won't really be using this
        high_action = np.array([self.max_angle_radians, self.max_angle_radians], dtype=np.float32)
        low_action = -high_action

        self.high_obs = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi], np.float32)
        low_obs = np.array([0.0, 0.0, 0.0], np.float32)

        super().__init__(low_obs=low_obs, high_obs=self.high_obs, low_action=low_action, high_action=high_action,
                         **kwargs)

        self.name = 'SO(3)'
        self.observation_dim = 6

    def _update_state_intrinsic(self, action):
        # Shouldn't be needed
        raise NotImplementedError()

    def sample(self, n_samples):
        # Only keep first 2 columns (R^6 embedding)
        samples = np.stack([Rotation.random(random_state=self.rng).as_matrix()[:, :2] for _ in range(n_samples)],
                           axis=0).reshape(
            (n_samples, -1))

        return samples

    def intrinsic_to_embedding(self, intrinsic):
        return S2.intrinsic_to_embedding(self, intrinsic)

    def get_intrinsic_mesh(self):
        return S2.get_intrinsic_mesh(self)

    def to_full_matrix(self, R):
        """Cross product to predict the third column of an R^6 rotation prediction.

        Return the reshaped matrix."""
        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R)

        # Auto-batching
        single_traj = R.ndim == 2
        if single_traj:
            R = R.unsqueeze(0)

        # Reshape to matrix form and add third column
        R = einops.rearrange(R, "n t (m1 m2) -> n t m1 m2", m1=3, m2=2)
        third = torch.cross(R[..., 0], R[..., 1], dim=-1)
        R = torch.cat([R, third.unsqueeze(-1)], dim=-1)

        # Auto-batching
        if single_traj:
            R = R[0]

        return R

    def projection(self, actions, obs):
        """Project ambient R^6 prediction to SO(3) manifolds.

        Operates on R^6 rotation embedding (first two columns).
        Works on trajectories."""
        # obs, ps = einops.pack([obs], "* t m")  # Auto-batching

        # Matrix format
        obs_mx = einops.rearrange(obs.clone(), "b t (m1 m2) -> b t m1 m2", m1=3, m2=2)

        # SVD
        U, S, Vt = torch.svd(obs_mx)

        # Orthogonalize
        obs_prime = U @ Vt

        # Flat format
        obs_prime = einops.rearrange(obs_prime, "n t m1 m2 -> n t (m1 m2)", m1=3, m2=2)

        # Flip check
        with torch.no_grad():
            norm_1 = torch.linalg.norm(obs - obs_prime, dim=-1)
            norm_2 = torch.linalg.norm(obs + obs_prime, dim=-1)
            mask = norm_2 < norm_1
            # print(f"Flipped {mask.sum()} matrices")
        obs_prime[mask] *= -1

        # Sanity check (comment this for performance)
        # mx = self.to_full_matrix(obs_prime)
        # det = mx.det()
        # import pdb; pdb.set_trace()
        # if not torch.allclose(torch.ones_like(det), det):
        #     raise Exception('We have weird determinants')

        # [obs_prime] = einops.unpack(obs_prime, ps, "* t m")  # Auto-batching

        return actions, obs_prime

    def seq_geodesic_distance(self, R):
        # Given b sequences of t rotations  (b x t x 9 Tensor)
        # Return geodesic distance of the trajectory (b Tensor)
        # See On the Continuity of Rotation Representations in Neural Networks, p.7
        R = self.to_full_matrix(R)
        from_ = R[:, :-1, :]
        to = R[:, 1:, :]

        M = from_ @ torch.inverse(to)
        traces = torch.einsum("ijkk->ij", M)
        cos = (traces - 1) / 2

        # cos may be lower than -1 or higher than 1 due to numerical issues
        # clip to appropriate range to avoid nans in arccos output
        cos = torch.clip(cos, min=-1., max=1.)

        dist = torch.arccos(cos)

        return dist.sum(dim=1)

    def interpolate(self, start, goal, times=None):
        # Need to map R^6 to Scipy rotations
        coords = np.vstack((start, goal))
        matrices = self.to_full_matrix(coords)
        rots = Rotation.from_matrix(matrices)

        slerp = Slerp([0, 1], rots)

        if times is None:
            times = np.linspace(0, 1, num=self.horizon, endpoint=True)

        mx_traj = slerp(times).as_matrix()

        # Map to R^6 embedding
        mx_traj_r6 = mx_traj[:, :, :2].reshape((len(times), 6))

        return mx_traj_r6

    def interpolate_batch(self, x_0, x_t, times):
        # Naive numpy iterative interpolation for now
        device = x_0.device
        x_0, x_t = x_0.cpu().numpy(), x_t.cpu().numpy()
        times = times.cpu().numpy()

        batch_size = x_0.shape[0]
        result = list()
        for i in range(batch_size):
            result.append(self.interpolate(x_0[i], x_t[i], np.array([times[i]])))

        result = np.concatenate(result, axis=0)

        return torch.from_numpy(result).to(device)

    def render(self, observations=None, mode='human'):
        """Render trajectory of 3D rotations.

        Input should be an R^6 rotation embedding.
        """
        # Plot manifold mesh
        mesh = self.get_intrinsic_mesh()
        fig, ax = triu_plot(self.intrinsic_to_3D(mesh), mesh)

        # Generate some 3D arrow
        shape = np.linspace([0., 0., 0.], [.1, 0., .0], endpoint=True, num=100)
        handle_1 = np.linspace([.1, .0, 0.], [.075, .025, 0.], endpoint=True, num=100)
        handle_2 = np.linspace([.1, 0., 0.], [.075, -.025, 0.], endpoint=True, num=100)
        shape = np.concatenate((shape, handle_1, handle_2), axis=0)
        shape *= 2  # Bigger arrow
        shape += np.array([[0., 0., 1.]])  # Put arrow in tangent space of unit sphere
        n_points = shape.shape[0]

        # Rotate arrow
        # rot = Rotation.from_euler("zx", [180, 15], degrees=True).as_matrix()
        # # shape = shape @ rot.T
        # # print(np.eye(3)[:, :2].reshape((6,)))
        # print(rot[:, :2].reshape((6,)))

        # Apply rotations to arrow
        observations = self.to_full_matrix(observations)
        # observations = torch.stack([torch.eye(3) for _ in range(observations.shape[0])], dim=0)  # To visualize untransformed arrow
        shape_rot = observations @ shape.T
        shape_rot = np.transpose(shape_rot, (0, 2, 1)).reshape((-1, 3))
        c = np.repeat(np.arange(observations.shape[0]), n_points)
        rot = Rotation.from_euler("zx", [25, 0], degrees=True).as_matrix()
        shape_rot = shape_rot @ rot.T
        fig, ax = surface_plot(shape_rot, fig=fig, ax=ax, c=c, cmap='rainbow')

        # Remove border
        ax.margins(0)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_zticklabels([])

        fig.canvas.draw()
        im = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        im = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()

        return im


class SO3GS(SO3):
    """Use Gram–Schmidt projection instead of SVD."""

    def projection(self, actions, obs):
        """Project ambient R^6 prediction to SO(3) manifolds.

        Operates on R^6 rotation embedding (first two columns).
        Works on batch of trajectories."""
        # obs, ps = einops.pack([obs], "* t m")  # Auto-batching

        # Matrix format
        obs = einops.rearrange(obs.clone(), "b t (m1 m2) -> b t m1 m2", m1=3, m2=2)

        # Unit norm first vector
        obs[..., 0] = torch.nn.functional.normalize(obs[..., 0].clone(), p=2, dim=-1)

        # Orthogonalize second vector
        dot_u1_u2 = (obs[..., 0].clone().unsqueeze(-2) @ obs[..., 1].clone().unsqueeze(-1)).squeeze(-1)
        obs[..., 1] = obs[..., 1].clone() - dot_u1_u2 * obs[..., 0].clone()

        # Unit norm second vector
        obs[..., 1] = torch.nn.functional.normalize(obs[..., 1].clone(), p=2, dim=-1)

        # Flat format
        obs = einops.rearrange(obs, "n t m1 m2 -> n t (m1 m2)", m1=3, m2=2)

        # Sanity check (comment this for performance)
        # mx = self.to_full_matrix(obs)
        # det = mx.det()
        # if not torch.allclose(torch.ones_like(det), det):
        #     raise Exception('We have weird determinants')

        # [obs_prime] = einops.unpack(obs_prime, ps, "* t m")  # Auto-batching

        return actions, obs


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # env = SO3(seed=42)
    #
    # R = torch.randn((5, 10, 6))[0]
    # _, R = SO3.projection(None, None, R)
    # for i in [R]:
    #     im = env.render(i)
    #     plt.imshow(im)
    #     plt.show()
    # print(R.shape)
    # exit()

    # your code
    import matplotlib.pyplot as plt
    import geotorch

    env = SO3GS(seed=42, n_samples_planner=5000)
    dataset = env.get_dataset(100)

    # Render some planner trajectories
    # for i in range(10):
    #     im = env.render(torch.from_numpy(dataset['observations'][i * 12: (i + 1) * 12]))
    #     plt.imshow(im)
    #     plt.show()

    # Score planner trajectories
    obs = dataset['observations'].reshape((100, 12, 6))
    obs_t = torch.from_numpy(obs)
    geo = env.seq_geodesic_distance(obs_t)
    obs_direct = obs_t[:, [0, -1], :]
    geo_direct = env.seq_geodesic_distance(obs_direct)

    # Check batch interpolations
    traj = obs_t[0]
    start, goal = traj[0], traj[-1]
    start = torch.tile(start, (12, 1))
    goal = torch.tile(goal, (12, 1))
    times = torch.linspace(0, 1, steps=12)
    inter = env.interpolate_batch(start, goal, times)
    assert torch.allclose(traj, inter)

    # print((geo / geo_direct).mean())

    # Some checks
    def is_steifel(obs):
        stiefel = geotorch.Stiefel((3, 2))
        for b in range(obs.shape[0]):
            for t in range(obs.shape[1]):
                assert stiefel.in_manifold(obs[b, t])


    def is_so3(obs):
        so3 = geotorch.SO((3, 3))
        for b in range(obs.shape[0]):
            for t in range(obs.shape[1]):
                assert so3.in_manifold(obs[b, t])


    # Make sure our data is actually on the manifold
    obs = obs_t.reshape((100, 12, 3, 2))
    is_steifel(obs)

    # Now make sure the R^6 embedding is on the Steifel manifold
    obs_randn = torch.randn((100, 12, 6))
    obs_randn_bk = obs_randn.clone()
    _, obs_svd = SO3.projection(None, None, obs_randn)
    is_steifel(obs_svd.reshape((100, 12, 3, 2)))
    _, obs_gs = SO3GS.projection(None, None, obs_randn)
    is_steifel(obs_gs.reshape((100, 12, 3, 2)))

    # Complete matrices and check if they are on SO(3)
    obs_svd_mx = SO3.to_full_matrix(None, obs_svd)
    obs_svd_gs = SO3GS.to_full_matrix(None, obs_gs)
    is_so3(obs_svd_mx)
    is_so3(obs_svd_gs)

    # New projections
    _, obs_svd_2 = SO3.projection(None, None, obs_randn)
    is_steifel(obs_svd_2.reshape((100, 12, 3, 2)))
    _, obs_gs_2 = SO3GS.projection(None, None, obs_randn)
    is_steifel(obs_gs_2.reshape((100, 12, 3, 2)))

    # Make sure reprojections and projections are identical
    assert torch.allclose(obs_gs, obs_gs_2)
    assert torch.allclose(obs_svd, obs_svd_2)

    # Make sure input noise didn't change
    assert torch.allclose(obs_randn_bk, obs_randn)

    # Now project an already projected version
    _, obs_svd_3 = SO3.projection(None, None, obs_svd)
    is_steifel(obs_svd_3.reshape((100, 12, 3, 2)))
    _, obs_gs_3 = SO3GS.projection(None, None, obs_gs)
    is_steifel(obs_gs_3.reshape((100, 12, 3, 2)))
    assert torch.allclose(obs_gs, obs_gs_3, atol=1e-5)
    assert torch.allclose(obs_svd, obs_svd_3, atol=1e-5)

    # Score random trajectories
    # obs = torch.randn((1000, 12, 6))
    # _, obs = env.projection(None, obs)
    # geo = env.seq_geodesic_distance(obs)
    # obs_direct = obs[:, [0, -1], :]
    # geo_direct = env.seq_geodesic_distance(obs_direct)
    # print((geo / geo_direct).mean())
