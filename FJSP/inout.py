# Copyright (c) 2025 Zijun Liao
# Licensed under the MIT License.

import torch
import os
import numpy as np
import re


class DesFJSP():
    def __init__(
            self, 
            num_jobs = 10,
            num_machines = 5,
            num_operation = (4,6),
            num_assign_machines = (1,5),
            avg_pt = (1, 20),) -> None:
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.num_operation = num_operation
        self.num_assign_machines = num_assign_machines
        self.avg_pt = avg_pt
    
    @property
    def name(self):
        return "FJSP-{}x({}-{})x{}-{}".format(
            self.num_jobs, self.num_operation, self.num_assign_machines, 
            self.num_machines, self.avg_pt
        )
    @property
    def size(self):
        return "FS{}x{}x{}".format(self.num_jobs, np.max(self.num_operation), self.num_machines)


def standardize(input: torch.Tensor, dim: int = 0, eps: float = 1e-6):
    """
    Remove mean and divide for standard deviation (z-score).

    Args:
        input: Input tensor to be standardized.
        dim: Dimension of the standardization.
        eps: A value added to the denominator for numerical stability.
    """
    means = input.mean(dim=dim, keepdim=True)
    stds = input.std(dim=dim, keepdim=True) + eps
    return (input - means) / stds


def read_basic(f_path: str):
    """
    Load the basic information about a FJSP instance.
    """
    with open(f_path) as f:
        lines = f.readlines()
    name = os.path.basename(f_path).split('.')[0]

    NMK = re.split("[ \t]+", lines[0].strip())
    N, M, _ = NMK
    N, M = int(N), int(M)

    max_n_op = 0
    process_times = []
    for i in range(1, N+1):
        data = [int(t) for t in \
            re.split('[ \t]+', lines[i].strip())]
        K = data[0]
        max_n_op = max(max_n_op, K)
        job_process_time = []

        p = 1
        for _ in range(K):
            operation_process_time = [-1 for _ in range(M)]
            num_machine = data[p]
            for _ in range(num_machine):
                machine = data[p+1]-1
                pt = data[p+2]
                assert machine >= 0
                operation_process_time[machine] = pt
                p += 2
            p += 1
            job_process_time.append(operation_process_time)
        #
        process_times.append(job_process_time)

    # 
    for job in process_times:
        if len(job) < max_n_op:
            job += [[0,] + [-1 for _ in range(M-1)] 
                    for _ in range(max_n_op - len(job))]
    process_times = np.array(process_times, np.int32)

    makespan = int(lines[N+1]) if len(lines) > N+1 else None

    # Remove the machine which is empty
    process_times = np.stack(
        [process_times[:, :, m] for m in range(M) 
         if np.any(process_times[:, :, m] >= 0)],
        axis=2)

    return name, process_times, makespan


def gen_fjsp_instance(instance: DesFJSP, random=None) -> np.ndarray:
    if random is None:
        random = np.random.RandomState()
    N = instance.num_jobs
    N = random.randint(*N) if isinstance(N, tuple) else N
    M = instance.num_machines
    M = random.randint(*M) if isinstance(M, tuple) else M
    K = instance.num_operation
    K = max(K)-1 if isinstance(K, tuple) else K

    data = -np.ones((N, K, M), dtype=np.int32)
    for job in range(N):
        k = instance.num_operation
        k = random.randint(*k) if isinstance(k, tuple) else k

        for op in range(k):
            m = instance.num_assign_machines
            m = random.randint(*m) if isinstance(m, tuple) else m
            assign_m = random.choice(M, m, replace=False)
            pt = random.randint(*instance.avg_pt)
            data[job, op, assign_m] = pt
    
    return data


def save_fjsp_instance(instance: np.ndarray, path: str):
    with open(path, 'w') as f:
        N, K, M = instance.shape
        f.write(f"{N} {M} {5}\n")
        for job in range(N):
            f.write(f"{K} ")
            for op in range(K):
                m = np.where(instance[job, op] >= 0)[0]
                f.write(f"{len(m)} ")
                for machine in m:
                    f.write(f"{machine+1} {instance[job, op, machine]} ")
            f.write('\n')


def cluster_edges(data: np.ndarray, device: str = 'cpu'):
    """
    Make the cluster of jobs, operations and machines as an edge list.

    :return:
        The edges of the clusters.
    """
    num_j, num_o, num_m = data.shape
    c = -1
    def count():
        nonlocal c
        c += 1
        return c
    ops = {
        (i, j, k): count()
        for i in range(num_j)
        for j in range(num_o)
        for k in range(num_m)
        if data[i, j, k] >= 0
    }

    # Job cluster
    edges_j = []
    # Operation cluster
    edges_o = []
    for i in range(num_j):
        lasts = []
        lasts_ = []
        for j in range(num_o):
            for k in range(num_m):
                if data[i, j, k] < 0:
                    continue
                cur_o = ops[(i, j, k)]
                for last_o in lasts:
                    edges_j.append((last_o, cur_o))
                for last_o in lasts_:
                    edges_o.append((last_o, cur_o))
                    edges_o.append((cur_o, last_o))
                lasts_.append(cur_o)
            lasts = lasts_
            lasts_ = []
    #
    edges_j = torch.tensor(edges_j, dtype=torch.long, device=device)
    edges_o = torch.tensor(edges_o, dtype=torch.long, device=device)

    # Machine cluster
    edges_m = []
    for m in range(num_m):
        # Get the operations on the machine
        machine = data[:, :, m]
        mac_ops = []

        for i in range(num_j):
            for j in range(num_o):
                if machine[i, j] < 0:
                    continue
                cur_o = ops[(i, j, m)]
                for o in mac_ops:
                    edges_m.append((o, cur_o))
                    edges_m.append((cur_o, o))
                mac_ops.append(cur_o)
    #
    edges_m = torch.tensor(edges_m, dtype=torch.long, device=device)

    return edges_j, edges_o, edges_m


def extract_features(data: np.ndarray, device: str = 'cpu'):
    """
    Compute the base set of features from the instance information.

    :return:
        The set of features
    """
    num_j, num_o, num_m = data.shape
    num_ops = len(data[data >= 0])
    q = np.array([0.25, 0.5, 0.75])
    _max = data.max()
    data = data / _max

    # Job-related
    fea_j_q = np.zeros((num_ops, 3))
    fea_j_d = np.zeros((num_ops, 3))
    count = 0
    for j in range(num_j):
        data_j = data[j]
        costs = data_j[data_j >= 0]
        j_q = np.quantile(costs, q).T
        fea_j_q[count] = j_q
        for cost in costs:
            j_d = cost - j_q
            fea_j_q[count] = j_q
            fea_j_d[count] = j_d
            count += 1
    assert count == num_ops

    # Machine-related
    fea_m_q = np.zeros((num_ops, 3))
    fea_m_d = np.zeros((num_ops, 3))
    mac_q = np.zeros((num_m, 3))
    count = 0
    for m in range(num_m):
        data_m = data[:, :, m]
        costs = data_m[data_m >= 0]
        m_q = np.quantile(costs, q).T
        mac_q[m] = m_q
    for j, o, m in zip(*np.where(data >= 0)):
        fea_m_q[count] = mac_q[m]
        fea_m_d[count] = data[j, o, m] - mac_q[m]
        count += 1
    assert count == num_ops

    # Operation-related
    costs = data[data >= 0]
    avg_cost = np.mean(data, axis=2, where=data >= 0)

    avg_sum = np.sum(avg_cost, axis=1, keepdims=True)

    avg_cumsum = np.cumsum(avg_cost, axis=1)

    avg_completion = avg_cumsum / avg_sum

    avg_remain = (avg_sum - avg_cumsum + avg_cost) / avg_sum

    fea = np.zeros((num_ops, 3))
    for i, (j, o, m) in enumerate(zip(*np.where(data >= 0))):
        fea[i, 0] = costs[i]
        fea[i, 1] = avg_completion[j, o]
        fea[i, 2] = avg_remain[j, o]
    #
    features = np.concatenate([
            fea, 
            fea_j_q, fea_j_d, 
            fea_m_q, fea_m_d], 
        axis=1)
    assert features.shape == (num_ops, 15)
    return torch.FloatTensor(features).to(device)


def load_data(path, device: str = 'cpu'):
    """
    Load a FJSP instance from path and return a PyTorch Data object.

    Args:
        path: The path to the input instance.
        device: Either cpu or cuda.
    Return:
        Dict containing the information about the instance
    """
    # Load the instance from the instance.fjs file
    name, instance, ms = read_basic(path)
    num_j, num_o, num_m = instance.shape

    # Make the graph
    edges_j, edges_o, edges_m = cluster_edges(instance, device=device)

    # Prepare the features
    x = extract_features(instance, device=device)
    x = standardize(x, dim=0)

    # Make the data object of the loaded instance
    data = dict(
        name=name, path=path,
        j=num_j, o=num_o, m=num_m,
        shape=f"{num_j}x{num_o}x{num_m}",
        x=x,                  # Features
        job_edges=edges_j.t().contiguous(),
        ops_edges=edges_o.t().contiguous(),
        mac_edges=edges_m.t().contiguous(),
        data=instance,
        makespan=ms         # Optional
    )
    return data


def load_dataset(path: str = './dataset/',
                 use_cached: bool = True,
                 iter_num: int = None,
                 device: str = 'cpu'):
    """
    Load the dataset.

    Args:
        path: Path to the folder that contains the instances.
        use_cached: Whether to use the cached dataset.
        iter_num: Number of instances to load.
        device: Either cpu or cuda.
    Returns:
        instances: (list)
    """
    print(f"Loading {path} ...")
    c_path = os.path.join(path, 'cached.pt' if iter_num is None else f'cached_{iter_num}.pt')
    if use_cached and os.path.exists(c_path):
        print(f'\tUsing {c_path} ...')
        instances = torch.load(c_path, map_location=device)
    else:
        print('\tExtracting features ...')
        instances = []
        for file in os.listdir(path):
            print(f"\t\t{file}")
            if file.startswith('.') or not file.endswith('.fjs'):
                continue
            instances.append(load_data(os.path.join(path, file),
                                       device=device))
            if iter_num is not None and len(instances) >= iter_num:
                break
        torch.save(instances, c_path)
    print(f"Number of dataset instances = {len(instances)}")
    return instances

