import tqdm
from tqdm import tqdm

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from multiprocessing import Pool


def ortool_solver(nodes, vehicle_count, vehicle_time_budget, vehicle_speed, late_cost):
    manager = pywrapcp.RoutingIndexManager(nodes.size(0), vehicle_count, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_idx, to_idx):

        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return int(nodes[from_node, :2].sub(nodes[to_node, :2]).pow(2).sum().pow(0.5))

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    horizon = vehicle_time_budget

    def time_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return int(nodes[from_node, 2] + nodes[from_node, :2].sub(nodes[to_node, :2]).pow(2).sum().pow(0.5) / vehicle_speed)

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(time_callback_index, horizon, 2 * horizon, True, "Time")
    time_dimention = routing.GetDimensionOrDie("Time")

    for i in range(vehicle_count):
        idx = routing.End(i)
        time_dimention.SetCumulVarSoftUpperBound(idx, horizon, late_cost)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    assign = routing.SolveWithParameters(params)

    routes = []
    for i in range(vehicle_count):
        route = []
        idx = routing.Start(i)
        while not routing.IsEnd(idx):
            idx = assign.Value(routing.NextVar(idx))
            route.append(manager.IndexToNode(idx))
        routes.append(route)

    return routes


def ortool_solve(data, late_cost=1):
    with Pool() as p:
        with tqdm(total=data.batch_size, desc="Calling ORTools") as pbar:
            results = [
                p.apply_async(ortool_solver, (nodes, data.vehicle_count, data.vehicle_time_budget, data.vehicle_speed, late_cost),
                              callback=lambda _: pbar.update()) for nodes in data.nodes_generate()]

            routes = [res.get() for res in results]
    return routes