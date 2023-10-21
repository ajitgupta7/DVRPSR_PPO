import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from nets.GraphMultiHeadAttention import GraphMultiHeadAttention
from nets.Encoder import GraphEncoder
torch.autograd.set_detect_anomaly(True)


class GraphAttentionModel(nn.Module):
    def __init__(self, num_customers, customer_feature, vehicle_feature, model_size=128, encoder_layer=3,
                 num_head=8, ff_size=128, tanh_xplor=10):
        super(GraphAttentionModel, self).__init__()
        # get models parameters for encoding-decoding
        self.model_size = model_size
        self.scaling_factor = 1 / math.sqrt(self.model_size)
        self.tanh_xplor = tanh_xplor

        # Initialize encoder and embeddings
        self.customer_encoder = GraphEncoder(encoder_layer=encoder_layer,
                                             num_head=num_head,
                                             model_size=model_size,
                                             ff_size=ff_size)

        self.customer_embedding = nn.Linear(customer_feature, model_size)
        self.depot_embedding = nn.Linear(customer_feature, model_size)
        self.vehicle_embedding = nn.Linear(vehicle_feature, self.model_size, bias=False)

        self.fleet_attention = GraphMultiHeadAttention(num_head, model_size)
        self.vehicle_attention = GraphMultiHeadAttention(num_head, model_size)

        self.graph_projection = nn.Linear(self.model_size, self.model_size)
        self.customer_projection = nn.Linear(self.model_size, self.model_size) # TODO: MLP instaed of nn.Linear

    def encoder(self, env, customer_mask=None):
        customer_embed = torch.cat((self.depot_embedding(env.nodes[:, 0:1, :]),
                                    self.customer_embedding(env.nodes[:, 1:, :])), dim=1)

        if customer_mask is not None:
            customer_embed[customer_mask] = 0

        self.customer_encoding, self.graph_encoding = self.customer_encoder(customer_embed, mask=customer_mask)
        self.graph_representation = self.graph_projection(self.graph_encoding.unsqueeze(1))
        self.customer_representation = self.customer_projection(self.customer_encoding)
        if customer_mask is not None:
            self.customer_representation[customer_mask] = 0

    def decoder(self, env):

        vehicles_embed = self.vehicle_embedding(env.vehicles)
        fleet_representation = self.fleet_attention(vehicles_embed,
                                                    self.graph_representation,
                                                    self.graph_representation)
        vehicle_query = fleet_representation.gather(1,
                                                    env.current_vehicle_index.unsqueeze(2).expand(
                                                     -1, -1, self.model_size))
        vehicle_representation = self.vehicle_attention(vehicle_query,
                                                        fleet_representation,
                                                        fleet_representation)
        compact = torch.bmm(vehicle_representation,
                            self.customer_representation.transpose(2, 1))
        compact *= self.scaling_factor

        x = compact.clone()

        if self.tanh_xplor is not None:
            compact = self.tanh_xplor * compact.tanh()
        compact[env.current_vehicle_mask] = -float('inf')

        ###########################################################################################
        # waiting heuristic in case there is no customer, vehicle should wait at current location
        # compact[:, :, 0] = -(self.tanh_xplor)
        # ##########################################################################################

        prop = F.softmax(compact, dim=-1)
        return prop, x

    def forward(self):
        raise NotImplementedError
