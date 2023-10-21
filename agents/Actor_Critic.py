import torch
import torch.nn as nn
from nets.GraphAttentionModel import GraphAttentionModel
from nets.GraphAttentionModel_torch import GraphAttentionModel_torch
from torch.distributions.categorical import Categorical
from nets import Critic


class Actor_Critic(nn.Module):
    def __init__(self,
                 customer_feature,
                 vehicle_feature,
                 customers_count,
                 model_size=128,
                 encoder_layer=3,
                 num_head=8,
                 ff_size_actor=128,
                 ff_size_critic=512,
                 tanh_xplor=10,
                 greedy=False):
        super(Actor_Critic, self).__init__()
        self.critic = Critic(customers_count, ff_size_critic)
        self.actor = GraphAttentionModel(customers_count, customer_feature, vehicle_feature, model_size, encoder_layer,
                                               num_head, ff_size_actor, tanh_xplor)
        self.greedy = greedy

    def act(self, env):
        env.reset()
        actions, logps, rewards, values = [], [], [], []
        while not env.done:
            if env.new_customer:
                self.actor.encoder(env, env.customer_mask)
            prop, x = self.actor.decoder(env)
            dist = Categorical(prop)

            if self.greedy:
                _, customer_index = prop.max(dim=-1)
            else:
                customer_index = dist.sample()

            logp = dist.log_prob(customer_index)
            val = self.critic(x, env.current_vehicle_mask, customer_index)
            actions.append((env.current_vehicle_index, customer_index))
            logps.append(logp)
            rewards.append(env.step(customer_index))
            values.append(val)

        logps = torch.cat(logps, dim=1).sum(dim=1)
        values = torch.cat(values, dim=1).sum(dim=1)
        return actions, logps, rewards, values

    def evaluate(self, env, old_actions):
        env.reset()
        entropys, old_actions_logps, values = [], [], []
        steps = old_actions.size(0)

        for i in range(steps):
            if env.new_customer:
                self.actor.encoder(env, env.customer_mask)

            old_action = old_actions[i, :, :]
            next_action = old_actions[i + 1, :, :] if i < steps - 1 else old_action
            next_vehicle_index = next_action[:, 0].unsqueeze(-1)
            prop, x = self.actor.decoder(env)

            dist = Categorical(prop)
            old_actions_logp = dist.log_prob(old_action[:, 1].unsqueeze(-1))
            entropy = dist.entropy()

            # get values from critic networks
            val = self.critic(x, env.current_vehicle_mask, old_action[:, 1].unsqueeze(-1))

            customer_index = old_action[:, 1].unsqueeze(-1)
            env.step(customer_index, next_vehicle_index)

            old_actions_logps.append(old_actions_logp)
            entropys.append(entropy)
            values.append(val)

        entropys = torch.cat(entropys, dim=1)
        num_e = entropys.ne(0).float().sum(1)
        entropys = entropys.sum(1) / num_e
        values = torch.cat(values, dim=1).sum(dim=1)

        old_actions_logps = torch.cat(old_actions_logps, dim=1).sum(dim=1)
        return entropys, old_actions_logps, values
