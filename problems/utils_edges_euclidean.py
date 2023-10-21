import torch
from torch.multiprocessing import Pool

def get_disatcne(id1, id2):
    return ((id1[0] - id2[0]) ** 2 + (id1[1] - id2[1]) ** 2) ** 0.5

def compute_edge_attributes(batch, V):
    edges = torch.zeros((V + 1, V + 1, 1), dtype=torch.float32)
    for i, node1 in enumerate(batch):
        for j, node2 in enumerate(batch):
            distance = get_disatcne(node1, node2)
            edges[i][j][0] = distance
    return edges.reshape(-1, 1)

def get_edges_attributes_parallel(batch_size, locations, V):
    print('Initialzing edges')

    edge_data = locations[:, :, 0:2]
    # generate edge index
    edges_index = []

    for i in range(V + 1):
        for j in range(V + 1):
            edges_index.append([i, j])
    edges_index = torch.LongTensor(edges_index)
    edges_index = edges_index.transpose(dim0=0, dim1=1)

    # generate nodes attributes
    with Pool() as pool:
            results = pool.starmap(compute_edge_attributes, [(batch, V) for batch in edge_data])

    edges_batch = torch.stack(results)

    return edges_index, edges_batch
