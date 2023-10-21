import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math


class GraphAttentionModel_torch(nn.Module):
    def __init__(self, num_customers, customer_feature, vehicle_feature, model_size=128, encoder_num=3,
                 num_head=8, ff_size=128, tanh_xplor=10):
        super(GraphAttentionModel_torch, self).__init__()
        # get models parameters for encoding-decoding
        self.model_size = model_size
        self.scaling_factor = 1 / math.sqrt(self.model_size)
        self.tanh_xplor = tanh_xplor

        self.customer_embedding = nn.Linear(customer_feature, self.model_size)
        self.depot_embedding = nn.Linear(customer_feature, self.model_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_size,
                                                   nhead=num_head,
                                                   dim_feedforward = 1024,
                                                   dropout=0.0)

        self.customer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                                      num_layers=encoder_num)

        self.vehicle_embedding = nn.Linear(vehicle_feature, self.model_size, bias=False)

        self.fleet_attention = nn.MultiheadAttention(model_size, num_head, kdim=model_size, vdim=model_size)
        self.vehicle_attention = nn.MultiheadAttention(model_size, num_head)
        self.customer_projection = nn.Linear(self.model_size, self.model_size)

    def encoder(self, env, customer_mask=None):
        customer_embed = torch.cat((self.depot_embedding(env.nodes[:, 0:1, :]),
                                    self.customer_embedding(env.nodes[:, 1:, :])), dim=1)
        if customer_mask is not None:
            customer_embed[customer_mask] = 0

        customer_representation = self.customer_encoder.forward(customer_embed)
        self.customer_representation = self.customer_projection(customer_representation)

        if customer_mask is not None:
            self.customer_representation[customer_mask] = 0

    def decoder(self, env):

        vehicles = self.vehicle_embedding(env.vehicles)

        fleet_representation, _ = self.fleet_attention.forward(query=vehicles.permute(1,0,2),
                                                               key=self.customer_representation.permute(1,0,2),
                                                               value=self.customer_representation.permute(1,0,2),
                                                               need_weights=False)

        vehicle_query = fleet_representation.gather(0,
                                                    env.current_vehicle_index.unsqueeze(2).expand(
                                                     -1, -1, self.model_size))

        vehicle_representation, _ = self.vehicle_attention.forward(query=vehicle_query.permute(1,0,2),
                                                                   key=fleet_representation,
                                                                   value=fleet_representation,
                                                                   need_weights=False)

        compact = torch.bmm(vehicle_representation.permute(1, 0, 2),
                            self.customer_representation.permute(0, 2, 1))
        compact *= self.scaling_factor
        x = compact.clone()
        if self.tanh_xplor is not None:
            compact = self.tanh_xplor * compact.tanh()

        compact[env.current_vehicle_mask] = -float('inf')

        ###########################################################################################
        # compact[:, :, 0] = -(self.tanh_xplor)
        # ##########################################################################################

        prop = F.softmax(compact, dim=-1)
        return prop, x

    def forward(self):
        raise NotImplementedError
