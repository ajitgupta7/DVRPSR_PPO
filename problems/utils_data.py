import torch
import random
import math
import pandas as pd
import numpy as np
import networkx as nx
import sys

def initialize_graph():
    coordinates = pd.read_csv("../vienna_data/vienna_dist.csv", header=None, sep=' ')
    coordinates.columns = ['coord1', 'coord2', 'dist']
    graph = nx.DiGraph()

    # add the rows to the graph for shortest path and distance calculations
    for _, row in coordinates.iterrows():
        graph.add_edge(row['coord1'], row['coord2'], weight=row['dist'])

    return graph


def precompute_shortest_path(graph, start_node, end_node):
    shortest_path = nx.shortest_path(graph, start_node, end_node)
     # TODO: distance need to be normalized afterwords
    shortest_path_length = sum(graph.get_edge_data(u, v)['weight']
                               for u, v in zip(shortest_path, shortest_path[1:]))

    return shortest_path, shortest_path_length


def get_distanceLL(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance


def get_NearestNodeLL(lat, lon, lats, lons):
    nearest = (-1, sys.float_info.max)
    for i in range(len(lats)):
        dist = get_distanceLL(lat, lon, lats[i], lons[i])
        if dist < nearest[1]:
            nearest = (i, dist)
    return nearest[0]


def get_depot_location(data_vienna):
    ll = (48.178808, 16.438460)
    lat = ll[0] / 180 * math.pi
    lon = ll[1] / 180 * math.pi
    lats = data_vienna['lats']
    lons = data_vienna['lons']
    depot = get_NearestNodeLL(lat, lon, lats, lons)
    depot_coordinates = np.array(data_vienna[data_vienna['id'] == depot][['id', 'xcoords', 'ycoords']])

    return depot_coordinates


def get_customers_coordinates(data_vienna, batch_size, customers_count, depot):
    torch.manual_seed(42)

    # Excluding depot id from the customers selection
    data_vienna_without_depot = data_vienna[data_vienna['id'] != int(depot[0][0])].reset_index()

    # Sample customers indices for all batches at once
    sampled_customers = torch.multinomial(torch.tensor(data_vienna_without_depot['id'], dtype=torch.float32),
                                          num_samples=batch_size * customers_count, replacement=True)

    sampled_customers = sampled_customers.reshape(batch_size, customers_count)

    # Gather the sampled locations using the indices
    sampled_locations = data_vienna_without_depot.loc[sampled_customers.flatten()].reset_index(drop=True)

    # Reshape the locations to match the batch size
    locations = sampled_locations.groupby(sampled_locations.index // customers_count)

    # Create PyTorch tensors for the batched data
    locations_tensors = []
    for _, batch in locations:
        id_tensor = torch.tensor(batch['id'].values, dtype=torch.long)
        coords_tensor = torch.tensor(batch[['xcoords', 'ycoords']].values, dtype=torch.float)
        batch_tensor = torch.cat((id_tensor.unsqueeze(1), coords_tensor), dim=1)
        locations_tensors.append(batch_tensor)

    return torch.stack(locations_tensors)

def generateRandomDynamicRequests(batch_size=2,
                                  V=20,
                                  V_static=10,
                                  fDmean=10,
                                  fDstd=2.5,
                                  Lambda=0.025,
                                  horizon=400,
                                  dep=0,
                                  u=0):
    gen = random.Random()
    gen.seed()  # uses the default system seed
    unifDist = gen.random  # uniform distribution
    durDist = lambda: max(0.01, gen.gauss(fDmean, fDstd))  # normal distribution with fDmean and fDstd

    # TODO: in actual data , we need to add a depo node with corrdinate, which should be removed from selected
    #       nodes as well.

    requests = []
    for b in range(batch_size):

        static_request = []
        dynamic_request = []
        u = 0

        while True:
            unif = unifDist()
            u += -(1 / (Lambda)) * math.log(unif)
            if (u > (horizon)) or (len(dynamic_request) > (V - V_static + 2)):
                break
            d = round(durDist(), 2)
            while d <= 0:
                d = round(durDist(), 2)

            dynamic_request.append([d, round(u, 2)])

        for j in range(V - len(dynamic_request)):
            d = round(durDist(), 2)
            while d <= 0:
                d = round(durDist(), 2)
            static_request.append([d, 0])

        request = static_request + dynamic_request
        random.shuffle(request)
        requests.append(request)

    return torch.tensor(requests)