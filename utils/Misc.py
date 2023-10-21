import torch
import os.path
from itertools import repeat, zip_longest
import numpy as np

def formate_old_actions(actions):
    max_len = max([action[0].size(0) for action in actions])
    old_actions = []

    for i in range(max_len):
        old_action = [[action[0][i].item() if i < action[0].size(0) else 0 for action in actions],
                      [action[1][i].item() if i < action[1].size(0) else 0 for action in actions]]
        old_actions.append(old_action)

    return old_actions


def _pad_with_zeros(src_it):
    yield from src_it
    yield from repeat(0)

def eval_apriori_routes(dyna, routes, rollout_count):
    mean_cost = dyna.nodes.new_zeros(dyna.minibatch)
    for c in range(rollout_count):
        dyna.reset()
        routes_it = [[_pad_with_zeros(route) for route in inst_routes] for inst_routes in routes]
        rewards = []
        while not dyna.done:
            cust_idx = dyna.nodes.new_tensor([[next(routes_it[n][i.item()])]
                                             for n,i in enumerate(dyna.current_vehicle_index)],
                                             dtype = torch.int64)
            rewards.append(dyna.step(cust_idx))

        #rewards = torch.cat(rewards, dim=1)
        #print(rewards)
        #rewards = rewards.sum(dim=1)

        mean_cost += torch.stack(rewards).sum(dim=0).squeeze(-1)
    return mean_cost / rollout_count