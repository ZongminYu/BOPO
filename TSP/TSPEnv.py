import os
from dataclasses import dataclass
import torch

from utils.TSProblemDef import get_random_problems, augment_xy_data_by_8_fold
from utils.guribo import guribo_tsp


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    B_IDX: torch.Tensor
    # shape: (batch, B)
    problem_size: int
    current_node: torch.Tensor = None
    # shape: (batch, B)
    ninf_mask: torch.Tensor = None
    # shape: (batch, B, node)
    

class TSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size'] if 'problem_size' in env_params else None
        self.sols_num = env_params['B'] if 'B' in env_params else None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.B_IDX = None
        # IDX.shape: (batch, B)
        self.problems = None
        self.opts = None
        # shape: (batch, problem_size, 2)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, B)
        self.selected_node_list = None
        # shape: (batch, B, 0~problem)

    def load_problems(self, batch_size, aug_factor=1, file_path=None, device='cpu'):
        self.batch_size = batch_size
        self.opts = None

        if file_path is not None and os.path.exists(file_path):
            self.problems, self.opts = torch.load(file_path, map_location=device)
            if aug_factor == 1:
                self.problems = self.problems[ :self.problems.shape[0]//8]
            # print('load problems from', file_path)
        else:
            self.problems = get_random_problems(batch_size, self.problem_size)
            # print('load random problems')
        # problems.shape: (batch, nodes, 2)

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                if file_path is None or not os.path.exists(file_path):
                    self.problems = augment_xy_data_by_8_fold(self.problems)
                # shape: (8*batch, problem, 2)
            else:
                raise NotImplementedError

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.sols_num)
        self.B_IDX = torch.arange(self.sols_num)[None, :].expand(self.batch_size, self.sols_num)

    def load_tsplib(self, batch_size, aug_factor=1):
        import tspins
        self.batch_size = batch_size
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
            else:
                raise NotImplementedError

        for name, problem, opt in tspins.problems:
            self.opts = torch.Tensor([opt,]).unsqueeze(0).expand(batch_size, -1)
            self.problems = torch.Tensor(problem).unsqueeze(0).expand(batch_size, -1, -1)
            self.problem_size = problem.shape[0]
            self.sols_num = self.problem_size
            
            assert aug_factor == 8
            if aug_factor > 1:
                if aug_factor == 8:
                    self.problems = augment_xy_data_by_8_fold(self.problems)
                    # shape: (8*batch, problem, 2)
                else:
                    raise NotImplementedError
            self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.sols_num)
            self.B_IDX = torch.arange(self.sols_num)[None, :].expand(self.batch_size, self.sols_num)
            
            yield name, problem.shape[0], opt

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, B)
        self.selected_node_list = torch.zeros((self.batch_size, self.sols_num, 0), dtype=torch.long)
        # shape: (batch, B, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, B_IDX=self.B_IDX, problem_size=self.problem_size)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.sols_num, self.problem_size))
        # shape: (batch, B, problem)

        goal = None
        done = False
        return Reset_State(self.problems), goal, done

    def pre_step(self):
        goal = None
        done = False
        return self.step_state, goal, done

    def step(self, selected):
        # selected.shape: (batch, B)

        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, B)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, B, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, B)
        self.step_state.ninf_mask[self.BATCH_IDX, self.B_IDX, self.current_node] = float('-inf')
        # shape: (batch, B, node)

        # returning values
        done = (self.selected_count == self.problem_size)
        if done:
            goal = -self._get_travel_distance()  # note the minus sign!
        else:
            goal = None

        return self.step_state, goal, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.problem_size, 2)
        # shape: (batch, B, problem, 2)
        seq_expanded = self.problems[:, None, :, :].expand(self.batch_size, self.sols_num, self.problem_size, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, B, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, B, problem)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, B)
        return travel_distances
    
    def guribo_test_tsp(self):
        if self.opts is None:
            self.opts = guribo_tsp(self.problems)
        return self.opts

    def save(self, file_path):
        torch.save((self.problems, self.opts), file_path)
