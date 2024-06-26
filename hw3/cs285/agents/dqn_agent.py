from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import cs285.infrastructure.pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.02) -> int:
        """
        Used for evaluation.
        """
        with torch.no_grad():
            observation = ptu.from_numpy(np.asarray(observation))[None]
            #print("ob=", observation.shape)
            # TODO(student): get the action from the critic using an epsilon-greedy strategy
            not_max_prob = epsilon / (self.num_actions - 1)
            max_prob = 1- epsilon
            prob = torch.full((self.num_actions, ), not_max_prob)
            #print("prob=", prob)
            ac_prob = self.critic(observation)
            #print(ac_prob)
            prob[torch.argmax(ac_prob)] = max_prob
            #print("final_prob=", prob)
            action = torch.multinomial(prob,num_samples=1)

        return ptu.to_numpy(action).squeeze(0).item()

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape

        # Compute target values
            # TODO(student): compute target values
        with torch.no_grad():
            next_qa_values = self.target_critic(next_obs)

            if self.use_double_q:
                next_critic_action = torch.argmax(self.critic(next_obs), dim=1).unsqueeze(1)
                next_q_values = torch.gather(next_qa_values, 1, next_critic_action).squeeze(1)
            else:
                next_action = torch.argmax(next_qa_values, dim=1)
                next_q_values, _ = torch.max(next_qa_values, dim=1)

            #print(next_qa_values)
            #print(next_q_values)
            not_done = ~done
            not_done = not_done.float()
            target_values = torch.add(reward, torch.mul(torch.mul(self.discount, next_q_values), not_done))
            
        # TODO(student): train the critic with the target values
        qa_values = self.critic(obs)
        #print(qa_values.shape)
        action = action.unsqueeze(1)
        #print(action.shape)
        q_values =  torch.gather(qa_values, 1, action).squeeze(1) # Compute from the data actions; see torch.gather
        loss = self.critic_loss(q_values, target_values)
        #print(loss)

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(student): update the critic, and the target if needed
        critic_stats = self.update_critic(obs,action,reward,next_obs,done)
        
        if step % self.target_update_period == 0:
            self.update_target_critic()

        return critic_stats
