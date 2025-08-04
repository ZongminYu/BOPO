# Copyright (c) 2025 Zijun Liao
# Licensed under the MIT License.

import torch
import numpy as np
import os

def get_random_problems(batch_size, problem_size):
    problems = torch.rand(size=(batch_size, problem_size, 2))
    # problems.shape: (batch, problem, 2)
    return problems


def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems

def read_tsp_file(file_path):
    """
    从 .tsp 文件中读取节点坐标。
    :param file_path: .tsp 文件的路径
    :return: 一个字典，键是节点编号，值是 (x, y) 坐标
    """
    coordinates = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        start_reading = False
        for line in lines:
            if line.strip() == "NODE_COORD_SECTION":
                start_reading = True
                continue
            if start_reading:
                parts = line.strip().split()
                if len(parts) == 3:
                    node_id, x, y = parts
                    coordinates[int(node_id)] = (float(x), float(y))
    return coordinates

def get_local_dataset_problems(batch_size, problem_size, episode, problems_dir):
    """
    加载 TSP 问题数据。
    :param batch_size: 批次大小
    :param problem_size: 问题规模（节点数量）
    :param episode: 起始序号
    :return: problems 张量，形状为 (batch_size, problem_size, 2)
    """
    problems = torch.zeros((batch_size, problem_size, 2), dtype=torch.float32)
    for i in range(batch_size):
        numbers = i + 1 + episode
        problem_path = os.path.join(problems_dir, f"dataset_n{problem_size}_b{numbers}.tsp")
        # print(f"Loading problem from: {problem_path}")
        coordinates = read_tsp_file(problem_path)
        # 将坐标转换为张量
        problem = torch.tensor([coordinates[node_id] for node_id in sorted(coordinates.keys())], dtype=torch.float32)
        
        # 将当前问题的张量存储到 problems 中
        problems[i] = problem

    return problems