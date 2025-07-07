import tsplib95
import numpy as np
import os

files = []
names = []

root = r'TSPLIB\res'
for file_name in os.listdir(root):
    if not file_name.endswith('.tsp') or 'dsj1000' in file_name:
        continue
    files.append(os.path.join(root, file_name))
    names.append(file_name.split('.')[0])


opts = {}
with open(r'TSPLIB\res\_Optimal solutions for symmetric TSPs.value') as file:
    for line in file.readlines():
        if ':' in line and '(' not in line:
            name, opt = line.split(':')
            name = name.strip()
            opt = float(opt)
            opts[name] = opt


eps = 0.005
problems = []
for name, file in zip(names, files):
    opt = opts[name]
    problem = tsplib95.load(file)

    nodes = problem.get_nodes()

    try:
        coords = [problem.node_coords[i] for i in nodes]
        coords = np.array(coords, dtype=np.float64)
    except KeyError:
        continue

    if coords.shape[0] >= 1000:
        continue

    if problem.edge_weight_type != 'EUC_2D':
        continue
    
    x_max = np.max(coords[:, 0])
    x_min = np.min(coords[:, 0])
    y_max = np.max(coords[:, 1])
    y_min = np.min(coords[:, 1])
    factor = max(x_max - x_min, y_max - y_min)
    
    coords[:, 0] = (coords[:, 0] - x_min) / factor
    coords[:, 1] = (coords[:, 1] - y_min) / factor
    
    opt /= factor

    assert np.all(coords >= 0) and np.all(coords <= 1)
    problems.append((name, coords, opt))
