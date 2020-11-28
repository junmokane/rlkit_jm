from rlkit.envs import ENVS  # done after envs.__init__.py executes
from rlkit.envs.wrappers import NormalizedBoxEnv

env = NormalizedBoxEnv(ENVS['HalfCheetah-Fwd']())
print(env.observation_space, env.action_space)

exit()
obs = env.reset()
print(obs.shape)

while True:
    env.render(mode='human')
    a = env.step(env.action_space.sample())
    print(a)