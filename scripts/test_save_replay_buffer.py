from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.save_replay_buffer import SaveReplayBuffer
from rlkit.envs import ENVS  # done after envs.__init__.py executes
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.policies.base import Policy
from rlkit.samplers.rollout_functions import rollout
from rlkit.samplers.data_collector.path_collector import MdpPathCollector
from rlkit.envs.read_hdf5 import get_dataset, qlearning_dataset


class RandomPolicy(Policy):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def get_action(self, observation, ):
        return self.env.action_space.sample(), {}


env = NormalizedBoxEnv(ENVS['HalfCheetah-Fwd']())
policy = RandomPolicy(env)

# test rollout function
single_ep = rollout(env, policy, max_path_length=1000)
print('key elements in single episode are :', single_ep.keys())

# test path collector
path_collector = MdpPathCollector(env, policy)
paths = path_collector.collect_new_paths(max_path_length=100,
                                         num_steps=1000,
                                         discard_incomplete_paths=False)
print('key elements in single episode are :', paths[0].keys())

# test replay buffer
replay_buffer = EnvReplayBuffer(int(1E6), env,)
replay_buffer.add_paths(paths)
print('the current size of replay buffer is', replay_buffer.num_steps_can_sample())

# Save the replay buffer
replay_buffer = SaveReplayBuffer(int(1E6), env,)  # this is just replay_buffer with save function
file_path = './data/halfcheetah_fwd.hdf5'
while True:
    paths = path_collector.collect_new_paths(max_path_length=100,
                                             num_steps=1111,
                                             discard_incomplete_paths=False)
    replay_buffer.add_paths(paths)
    size = replay_buffer.num_steps_can_sample()
    print('the current size of replay buffer is', size)
    if size > int(1E4):
        replay_buffer.save_buffer(file_path)
        break

# Read the saved replay buffer
data_dict = get_dataset(file_path)
# Run a few quick sanity checks
for key in ['observations', 'actions', 'rewards', 'terminals']:
    assert key in data_dict, 'Dataset is missing key %s' % key
print(data_dict['observations'].shape,
      data_dict['actions'].shape,
      data_dict['rewards'].shape,
      data_dict['terminals'].shape)
qlearning_data = qlearning_dataset(data_dict)
print(qlearning_data['observations'].shape,
      qlearning_data['actions'].shape,
      qlearning_data['next_observations'].shape,
      qlearning_data['rewards'].shape,
      qlearning_data['terminals'].shape)