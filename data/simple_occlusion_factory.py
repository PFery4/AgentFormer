import torch
Tensor = torch.Tensor


class SimpleOcclusionFactory:

    def __init__(self, p: float = 1.0, n_timesteps: int = 8):

        assert 0. <= p <= 1.0
        assert n_timesteps > 2

        self.agent_selection_probability = p            # [-] probability that a given agent in a provided set of trajectories will have its trajectory corrupted
        self.T_obs = n_timesteps                        # [-] number of total timesteps for which to generate an occluded observation mask
        self.max_occlusion_length = self.T_obs - 2      # [-] timesteps

    def corrupt_trajectories(self, n_agents: int):
        # trajectories [N, T, 2]

        occluded_agent_mask = torch.rand(size=[n_agents]) <= self.agent_selection_probability
        obs_mask = torch.ones([n_agents, self.T_obs])
        occlusion_lengths = torch.randint(low=1, high=self.max_occlusion_length + 1, size=(occluded_agent_mask.sum(),))

        for idx, occlusion_length in zip(torch.nonzero(occluded_agent_mask), occlusion_lengths):
            start_idx = torch.randint(high=self.T_obs-occlusion_length, size=(1,))
            obs_mask[idx, start_idx:start_idx+occlusion_length] = 0.
        return obs_mask


if __name__ == '__main__':

    n_agents = 10

    corruptor = SimpleOcclusionFactory(p=0.6, n_timesteps=8)
    print(f"{corruptor.corrupt_trajectories(n_agents=n_agents)=}")
