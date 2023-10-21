## run this code to get Vienna lats, lons and coordinates, these are identify as id (Nodes)
## based on: https://github.com/amflorio/dvrp-stochastic-requests/tree/main/in
## paper: https://www.sciencedirect.com/science/article/pii/S0377221722005677
## we use this paper to compare our results on Vienna dataset

import math
import pandas as pd


def getXYCoords(lat, lon, network):
    lat0 = (48.2206 / 180) * math.pi if network == "vienna" else (51.438 / 180) * math.pi
    offsetx = -0.185 if network == "vienna" else -0.059
    offsety = -0.841 if network == "vienna" else -0.897
    scaling = 25000 if network == "vienna" else 25000
    x = (lon * math.cos(lat0) + offsetx) * scaling
    y = (lat + offsety) * scaling
    return (x, y)


def loadRealWorld(network):
    filename = "./" + network + ".xy"
    with open(filename, "r") as fcoords:
        V = int(fcoords.readline())
        xcoords = [0] * V
        ycoords = [0] * V
        lats = [0] * V
        lons = [0] * V
        node_id = [0] * V
        for i in range(V):
            line = fcoords.readline().split()
            n = int(line[0])
            node_id[n] = int(line[0])
            lat = float(line[1])
            lon = float(line[2])
            xy = getXYCoords(lat, lon, network)
            xcoords[n] = xy[0]
            ycoords[n] = xy[1]
            lats[n] = lat
            lons[n] = lon
            if n != i:
                print("loadRealWorld(): inconsistency: n, i =", n, i)
                exit(-1)

    filename = "./" + network + ".d"
    with open(filename, "r") as fdists:
        A = int(fdists.readline())
        linksFromV = [[] for _ in range(V)]
        linksToV = [[] for _ in range(V)]
        links_ = []
        for l in range(A):
            line = fdists.readline().split()
            s = int(line[0])
            t = int(line[1])
            d = float(line[2])
            link = (s, t, d)
            links_.append(link)
            linksFromV[link[0]].append(link)
            linksToV[link[1]].append(link)

    # consistency check: every vertex needs to have at least 1 link to/from
    for i in range(V):
        if len(linksFromV[i]) == 0 or len(linksToV[i]) == 0:
            print("loadRealWorld(): inconsistency")
            exit(-1)

    return node_id, xcoords, ycoords, lats, lons, links_


# Example usage
network = "vienna"
n, xcoords, ycoords, lats, lons, links = loadRealWorld(network)

# create a dataframe and save as csv formate
data = {'id': n,
        'lats': lats,
        'lons': lons,
        'xcoords': xcoords,
        'ycoords': ycoords}
data = pd.DataFrame(data)
data.to_csv('vienna_cordinates.csv', index=False)
