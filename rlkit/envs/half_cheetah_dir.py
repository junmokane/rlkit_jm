import numpy as np

from .half_cheetah import HalfCheetahEnv
from . import register_env


@register_env('HalfCheetah-Fwd')
class HalfCheetahFwdEnv(HalfCheetahEnv):
    def __init__(self):
        self._goal_dir = 1
        super(HalfCheetahFwdEnv, self).__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = self._goal_dir * forward_vel
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost)
        return (observation, reward, done, infos)


@register_env('HalfCheetah-Bwd')
class HalfCheetahBwdEnv(HalfCheetahEnv):
    def __init__(self):
        self._goal_dir = -1
        super(HalfCheetahBwdEnv, self).__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = self._goal_dir * forward_vel
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost)
        return (observation, reward, done, infos)