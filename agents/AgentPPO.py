import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from itertools import chain

from agents.Actor_Critic import Actor_Critic
from problems import DVRPSR_Environment
import numpy as np

# Set the random seed for PyTorch, NumPy, and Python's random module
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

class AgentPPO:
    def __init__(self,
                 customer_feature,
                 vehicle_feature,
                 customers_count,
                 model_size=128,
                 encoder_layer=3,
                 num_head=8,
                 ff_size_actor=128,
                 ff_size_critic=128,
                 tanh_xplor=10,
                 greedy=False,
                 learning_rate=3e-4,
                 ppo_epoch=3,
                 batch_size=128,
                 entropy_value=0.2,
                 epsilon_clip=0.2,
                 max_grad_norm=2):

        self.policy = Actor_Critic(customer_feature, vehicle_feature, customers_count, model_size,
                                   encoder_layer, num_head, ff_size_actor, ff_size_critic,
                                   tanh_xplor, greedy)
        self.old_policy = Actor_Critic(customer_feature, vehicle_feature, customers_count, model_size,
                                       encoder_layer, num_head, ff_size_actor, ff_size_critic,
                                       tanh_xplor, greedy)

        self.old_policy.load_state_dict(self.policy.state_dict())

        self.ppo_epoch = ppo_epoch
        self.batch_size = batch_size
        self.entropy_value = entropy_value
        self.epsilon_clip = epsilon_clip
        # initialize the Adam optimizer
        self.optim = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': learning_rate},
                        {'params': self.policy.critic.parameters(), 'lr': 1e-4}])
        self.MSE_loss = nn.MSELoss(reduction='mean')
        self.max_grad_norm = max_grad_norm

    def advantage_normalization(self, advantage):
        std = advantage.std()
        # assert std != 0. and not torch.isnan(std), 'Need nonzero std'
        norm_advantage = (advantage - advantage.mean()) / (std + 1e-7)
        return norm_advantage

    def get_returns(self, R):
        returns = []
        discounted_returns = torch.zeros_like(R[0])
        for reward in reversed(R):
            discounted_returns = reward + (0.99 * discounted_returns)
            returns.insert(0, discounted_returns)

        returns = torch.stack(returns).permute(1, 0, 2)
        return returns

    def update(self, memory, epoch, env=None, env_params=None, device=None):
        self.policy.to(device)
        returns = self.get_returns(memory.rewards)
        old_rewards = returns.sum(dim=1).squeeze(-1).to(device)
        old_rewards = self.advantage_normalization(old_rewards)

        old_nodes = torch.stack(memory.nodes).to(device)

        old_values = torch.stack(memory.values).permute(1, 0).squeeze(-1).to(device)
        old_log_probs = torch.stack(memory.log_probs).to(device)
        old_actions = torch.stack(memory.actions).to(device)

        advantages = (old_rewards.detach() - old_values.detach())

        lr_scheduler = LambdaLR(self.optim, lr_lambda=lambda f: 0.98**epoch)
        env = env if env is not None else DVRPSR_Environment
        loss_t, norm_R, critic_R, loss_a, loss_mse, loss_e, ratios, grads = [], [], [], [], [], [], [], []

        for i in range(self.ppo_epoch):
            self.policy.train()
            dyna_env = env(None, old_nodes, *env_params)
            entropy, log_probs, values = self.policy.evaluate(dyna_env, old_actions.permute(1, 0, 2))

            R_norm = old_rewards
            mse_loss = self.MSE_loss(R_norm, values)
            ratio = torch.exp(log_probs - old_log_probs.detach())

            # PPO overall loss function
            actor_loss1 = ratio * advantages
            actor_loss2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            actor_loss = torch.min(actor_loss1, actor_loss2).mean()

            # total loss
            loss = actor_loss + 0.5 * mse_loss - self.entropy_value*entropy.mean()

            # optimizer and backpropogation
            self.optim.zero_grad()
            loss.mean().backward()
            grad_norm = clip_grad_norm_(chain.from_iterable(grp["params"] for grp in self.optim.param_groups),
                                        self.max_grad_norm)

            # print(R_norm.size(), values.size(), advantages.size(),
            #       actor_loss.size(), mse_loss.size(), loss.size(), entropy.size())

            self.optim.step()
            lr_scheduler.step()

            norm_R.append(torch.mean(R_norm.detach()).item())
            loss_t.append(torch.mean(loss.detach()).item())
            loss_a.append(torch.mean(actor_loss.detach()).item())
            loss_mse.append(torch.mean(mse_loss.detach()).item())
            loss_e.append(torch.mean(self.entropy_value * entropy.detach()).item())
            critic_R.append(torch.mean(values.detach()).item())
            ratios.append(torch.mean(ratio.detach()).item())
            grads.append(torch.mean(grad_norm.detach()).item())

        self.old_policy.load_state_dict(self.policy.state_dict())
        return loss_t, loss_a, loss_mse, loss_e, norm_R, critic_R, ratios, grads


if __name__ == '__main__':
    raise Exception('Cannot be called from main')
