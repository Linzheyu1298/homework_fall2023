from typing import Optional, Sequence
import numpy as np
import torch

from cs285.networks.policies import MLPPolicyPG
from cs285.networks.critics import ValueCritic
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
import time


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        #print("step1 input rewards:", rewards)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)
        #print("q_values:", q_values)
        #print("len=", len(q_values))

        # TODO: flatten the lists of arrays into single arrays, so that the rest of the code can be written in a vectorized
        # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
        # beyond this point.
        #print(obs.shape)
        def flatten_and_concat(arrays):
            """Flatten and concatenate a list of arrays."""
            return np.concatenate([np.asarray(arr).reshape(-1, arr.shape[-1] if arr.ndim > 1 else 1) for arr in arrays])
        
        assert len(obs) == len(actions) and len(rewards) == len(terminals) and len(terminals) == len(obs)
        obs = np.array(flatten_and_concat(obs))
        actions = np.array(flatten_and_concat(actions))
        rewards = np.array(flatten_and_concat(rewards))
        terminals = np.array(flatten_and_concat(terminals))
        #print(np.shape(terminals))

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        #print(q_values)

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG/policy
        # TODO: update the PG actor/policy network once using the advantages
        info: dict = self.actor.update(obs, actions, advantages)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            # TODO: perform `self.baseline_gradient_steps` updates to the critic/baseline network
            for _ in range(self.baseline_gradient_steps):
                critic_info: dict = self.critic.update(obs, q_values)

            info.update(critic_info)

        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            # TODO: use the helper function self._discounted_return to calculate the Q-values
            q_values = self._discounted_return(rewards)
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            # TODO: use the helper function self._discounted_reward_to_go to calculate the Q-values
            q_values = self._discounted_reward_to_go(rewards)

        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        #print(q_values)
        if self.critic is None:
            # TODO: if no baseline, then what are the advantages?
            advantages = q_values.copy()
            #print("q values=", q_values)

        else:
            # TODO: run the critic and use it as a baseline
            values = ptu.to_numpy(self.critic.forward(ptu.from_numpy(obs))).squeeze(1)            
            #print("values.shape",values.shape)
            #print("q_values.shape",q_values.shape)
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                # TODO: if using a baseline, but not GAE, what are the advantages?
                advantages = q_values - values
            else:
                # TODO: implement GAE
                batch_size = obs.shape[0]

                # HINT: append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    # TODO: recursively compute advantage estimates starting from timestep T.
                    # HINT: use terminals to handle edge cases. terminals[i] is 1 if the state is the last in its
                    # trajectory, and 0 otherwise.
                    if terminals[i] == 1:
                        # last_state
                        delta = rewards[i] - values[i]
                    else:
                        # other states
                        delta = rewards[i] + self.gamma * values[i+1] - values[i]

                    advantages[i-1] =  delta + self.gamma * self.gae_lambda * advantages[i]

                # remove dummy advantage
                advantages = advantages[:-1]

        # TODO: normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        if self.normalize_advantages:
            std = (np.std(advantages) + 1e-8) if np.std(advantages) != 0 else 1
            print("std=", std)
            advantages = (advantages - np.mean(advantages, axis=-1)) / std

        return advantages

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """
        #rewards: a list of nparray, each nparray represents one traj
        values = []
        for traj_re in rewards:    
            gamma_list = [pow(self.gamma,i) for i in range(len(traj_re))]
            disc_rw = np.dot(gamma_list, traj_re)
            value = np.full(len(traj_re),disc_rw)
            values.append(value)
            #print(value)        
        return np.concatenate(values)


    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        values = []

        for traj_re in rewards:
            n = len(traj_re)
            rtg = np.zeros(n)
            rtg[-1] = traj_re[-1]
            for t in reversed(range(n-1)):
                rtg[t] = traj_re[t] + self.gamma * rtg[t+1]
            values.append(rtg)
        
        return np.concatenate(values)
