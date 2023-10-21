
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):

    # critic will take environment as imput and ouput the values for loss function
    # which is basically the estimation of complexity of actions

    def __init__(self, customers_count, ff_size=512):
        super(Critic, self).__init__()
        self.ff_layer1 = nn.Linear(customers_count, ff_size, bias=False)
        self.ff_layer2 = nn.Linear(ff_size, customers_count, bias=False)

    def forward(self, x, current_vehicle_mask=None, customer_index=None):
        x[current_vehicle_mask] = 0
        value = F.relu(self.ff_layer1(x))
        value = self.ff_layer2(value)
        val = value.gather(2, customer_index.unsqueeze(-1)).expand(-1, 1, -1)
        return val.squeeze(1)