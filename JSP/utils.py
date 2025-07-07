import os
from datetime import datetime
import numpy as np

def find_nearest_distances(seq):
    L = len(seq)

    result_l = np.arange(L, dtype=np.int32)
    # 记录左侧最近的相同数字距离
    last_seen = {}
    for i in range(L):
        val = seq[i]
        if val in last_seen:
            result_l[i] = i - last_seen[val] - 1
        last_seen[val] = i

    result_r = np.arange(L, dtype=np.int32)[::-1]
    # 记录右侧最近的相同数字距离
    last_seen = {}
    for i in range(L-1, -1, -1):
        val = seq[i]
        if val in last_seen:
            result_r[i] = last_seen[val] - i - 1
        last_seen[val] = i

    return result_l, result_r


def shift(a, idx, bias):
    b = np.copy(a)
    if bias < 0:
        move_indices = np.arange(idx, idx + bias - 1, -1)
    else:
        move_indices = np.arange(idx, idx + bias + 1)

    for i in range(len(move_indices) - 1):
        b[move_indices[i]], b[move_indices[i + 1]] = b[move_indices[i + 1]], b[move_indices[i]]
    
    return b


class AverageMeter:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += (val * n)
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0


class ObjMeter:
    def __init__(self, name = 'makespan'):
        self.sum = {}
        self.count = {}
        self.meter = name

    def update(self, ins: dict, val: float):
        """
        Update with a new value for an instance.

        Args:
            ins: JSP instance.
            val: objective value (e.g. makespan) of the solution.
        Returns:
            None
        """
        shape = ins['shape']
        if shape not in self.sum:
            self.sum[shape] = val
            self.count[shape] = 1
        else:
            self.sum[ins['shape']] += val
            self.count[shape] += 1

    def __str__(self):
        out = ""
        for shape in sorted(self.sum):
            val = self.sum[shape] / self.count[shape]
            out += f"\t\t\t{shape:5}: AVG {self.meter}={val:4.3f}\n"
        return out[:-1]

    @property
    def avg(self):
        """ Compute total average value regardless of shapes. """
        return sum(self.sum.values()) / sum(self.count.values()) if self.count \
            else 0


class Logger(object):

    def __init__(self, file_name: str = 'log'):
        #
        self.line = None
        if not os.path.exists('./output/logs'):
            os.makedirs('./output/logs')
        self.file_path = f"./output/logs/{file_name}_" +\
                         f"{datetime.now().strftime('%d-%m-%H:%M')}.txt"

    def train(self, step: int, loss: float, makespan: float):
        self.line = f"step:{step:4},loss:{loss:.3f},makespan:{makespan:.3f}"

    def validation(self, gap: float = 0.):
        self.line += f",val_gap:{gap:.3f}"

    def flush(self):
        # Flush line
        with open(self.file_path, 'a+') as f:
            f.write(f"{datetime.now().strftime('%d-%m-%H:%M')},{self.line}\n")
