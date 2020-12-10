from gym.envs.mujoco import HalfCheetahEnv as HalfCheetahEnv_
import gym
import d4rl
import torch
import torch.nn as nn
import numpy as np

def get_diffs(x, model, batch_size=256):
    model.eval()
    with torch.no_grad():
        batchified = x.split(batch_size)
        stacked = []
        for _x in batchified:
            model.eval()
            diffs = []
            _x = _x.to(next(model.parameters()).device).float()
            x_tilde = model(_x)
            diffs.append((x_tilde - _x).cpu())

            for layer in model.enc_layer_list:
                _x = layer(_x)
                x_tilde = layer(x_tilde)
                diffs.append((x_tilde - _x).cpu())

            stacked.append(diffs)

        stacked = list(zip(*stacked))
        diffs = [torch.cat(s, dim=0).numpy() for s in stacked]

    return diffs


class RaPP(nn.Module):
    def __init__(self, in_dim):
        super(RaPP, self).__init__()
        self.enc_layer_list = [nn.Linear(in_dim, 4),
                               nn.ReLU(True),
                                nn.Linear(4, 4),
                                nn.ReLU(True),
                                nn.Linear(4, 4),
                                nn.ReLU(True),
                                nn.Linear(4, 2)
                               ]
        self.encoder = nn.Sequential(*self.enc_layer_list)
        self.decoder = nn.Sequential(nn.Linear(2, 4),
                                     nn.ReLU(True),
                                     nn.Linear(4, 4),
                                     nn.ReLU(True),
                                     nn.Linear(4, 4),
                                     nn.ReLU(True),
                                     nn.Linear(4, in_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = RaPP(23)
model.load_state_dict(torch.load('./rl_rapp_0.pt'))

# Loading test dataset
halfcheetah_task_name = ['halfcheetah-random-v0',
                         'halfcheetah-medium-v0',
                         'halfcheetah-expert-v0',
                         'halfcheetah-medium-replay-v0',
                         'halfcheetah-medium-expert-v0']

env = gym.make(halfcheetah_task_name[2])
dataset = env.get_dataset()
all_obs = np.array(dataset['observations'])
all_act = np.array(dataset['actions'])
N = all_obs.shape[0]
_obs = all_obs[:N - 1]
_actions = all_act[:N - 1]
obs_act = np.concatenate([_obs, _actions], axis=1)

with torch.no_grad():
    dif = get_diffs(torch.from_numpy(obs_act), model)
    difs = torch.cat([torch.from_numpy(i) for i in dif], dim=-1).numpy()
    dif = (difs ** 2).mean(axis=1)
    print(dif.shape)

'''
model = FlattenMlp(
            input_size=23,
            output_size=1,
            hidden_sizes=[128, 128],
        )
model.load_state_dict(torch.load('./rl_dropout_0.pt'))
with torch.no_grad():
    uncertainty = model(torch.from_numpy(obs_act)).numpy()
    uncertainty = np.ones_like(uncertainty) # tentative values for test
    print(uncertainty)
'''