# Copyright (c) 2025 Zijun Liao
# Licensed under the MIT License.

import time
import torch
import gurobipy as gp
from gurobipy import GRB

def distance(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

def solve_tsp_for_multiple_problems(cities_tensor):

    n_problems = cities_tensor.shape[0] # test_batch_size * aug_factor
    n_cities = cities_tensor.shape[1]

    results = []

    for i in range(n_problems):

        cities = cities_tensor[i].cpu().numpy()

        dist = [[distance(cities[i], cities[j]) for j in range(n_cities)] for i in range(n_cities)]
        
        model = gp.Model(f"TSP_{i+1}")

        model.setParam(GRB.Param.OutputFlag, 0)
        
        x = model.addVars(n_cities, n_cities, vtype=GRB.BINARY, name="x")

        model.setObjective(gp.quicksum(dist[i][j] * x[i, j] for i in range(n_cities) for j in range(n_cities) if i != j), GRB.MINIMIZE)

        for i in range(n_cities):
            model.addConstr(gp.quicksum(x[i, j] for j in range(n_cities) if i != j) == 1)
            model.addConstr(gp.quicksum(x[j, i] for j in range(n_cities) if i != j) == 1)

        u = model.addVars(n_cities, vtype=GRB.CONTINUOUS, lb=0, ub=n_cities-1, name="u")
        for i in range(1, n_cities):
            for j in range(1, n_cities):
                if i != j:
                    model.addConstr(u[i] - u[j] + n_cities * x[i, j] <= n_cities - 1)

        start_time = time.time()

        model.optimize()

        end_time = time.time()

        solving_time = end_time - start_time

        if model.status == GRB.OPTIMAL:
            tour = []
            for i in range(n_cities):
                for j in range(n_cities):
                    if i != j and x[i, j].x > 0.5:
                        tour.append((i, j))
            total_distance = model.objVal
            results.append((tour, total_distance, solving_time))
        else:
            if model.SolCount > 0:
                results.append((None, model.objVal, solving_time))
            else:
                results.append((None, None, solving_time))

    return results

def gurobi_tsp(cities):

    results = solve_tsp_for_multiple_problems(cities)

    total_distances = torch.tensor([result[1] for result in results])

    return total_distances
