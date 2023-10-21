import torch
from torch.multiprocessing import Pool
from problems.utils_data import precompute_shortest_path

def compute_edge_attributes(graph, batch, V):
    edges = torch.zeros((V + 1, V + 1, 1))
    for i, id1 in enumerate(batch):
        for j, id2 in enumerate(batch):
            _, distance = precompute_shortest_path(graph, int(id1), int(id2))
            edges[i][j][0] = distance
    return edges.reshape(-1, 1)

def get_edges_attributes_parallel(batch_size, graph, depot, locations, V):
    print('Initializing edges')

    edge_depot = torch.zeros((batch_size, 1, 1))
    edge_depot[:, :, :1] = depot[0][0]
    edge_data = torch.cat((edge_depot, locations[:, :, None, 0]), dim=1)

    edges_index = torch.tensor([[i, j] for i in range(V + 1) for j in range(V + 1)], dtype=torch.long).t()
    with Pool() as pool:
            results = pool.starmap(compute_edge_attributes, [(graph, batch, V) for batch in edge_data])

    edges_batch = torch.stack(results)

    return edges_index, edges_batch
