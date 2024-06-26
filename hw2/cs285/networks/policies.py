import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions
import time
from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""

        tensor_obs = ptu.from_numpy(obs)
        #print("tensor_obs_shape:", tensor_obs.shape)
        
 
        if self.discrete:
            action = self.forward(tensor_obs)
            # Sample an action from the probability distribution
            action = distributions.Categorical(probs=action)
            action = action.sample()
            action = ptu.to_numpy(action)
            #print("get action:", action)
            return action
        else:
            action_means, std = self.forward(tensor_obs)
            #print(action_means)
            cov_matrix = torch.diag(std)
            actions = F.tanh(distributions.MultivariateNormal(action_means,cov_matrix).sample())
            return ptu.to_numpy(actions)
            

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            logits = self.logits_net(obs)
            ac_prob = F.softmax(logits)
            #print(logits)
            return  ac_prob
            
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            means = self.mean_net(obs)
            std = torch.exp(self.logstd)
            return means, std

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError

        


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: implement the policy gradient actor update.
        self.optimizer.zero_grad()

        if self.discrete:
            actions = torch.tensor(actions, dtype=torch.int64)
            ac_prob = self.forward(obs)
            log_probs = torch.log(ac_prob)
            selected_log_probs = log_probs.gather(1, actions).squeeze(1)
        else:
            means, std = self.forward(obs)
            covariance_matrix = torch.diag(std)
            dist = distributions.MultivariateNormal(means, covariance_matrix)
            selected_log_probs = dist.log_prob(actions)
            #print("log_prob:", selected_log_probs)
            #time.sleep(1)
        
        loss = torch.neg(torch.mean(torch.mul(selected_log_probs, advantages)))
        loss.backward()
        self.optimizer.step()
        
        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
