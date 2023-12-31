import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, activation=nn.Tanh(), device='cpu'):
        super(ActorCritic, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.device = device
        self.body = nn.Sequential(
            init_(nn.Linear(state_dim, n_latent_var)),
            activation,
            init_(nn.Linear(n_latent_var, n_latent_var)),
            activation
        ).to(self.device)

        # Actor head
        self.action_layer = nn.Sequential(
            self.body,
            init_(nn.Linear(n_latent_var, action_dim)),
            nn.Softmax(dim=-1)
        ).to(self.device)

        # Critic head
        self.value_layer = nn.Sequential(
            self.body,
            init_(nn.Linear(n_latent_var, 1))
        ).to(self.device)


    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        # Receive numpy array
        state = torch.from_numpy(state).float().to(self.device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        # Return numpy array
        return action.cpu().numpy()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state.to(self.device))
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state.to(self.device))

        return action_logprobs, torch.squeeze(state_value), dist_entropy