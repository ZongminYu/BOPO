
import torch

import os
import time
from logging import getLogger

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model

from utils.utils import *


class TSPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL

        self.env = Env(**self.env_params)

        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()
        gap_AM = AverageMeter()
        aug_gap_AM = AverageMeter()
        time_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            st = time.time()
            score, aug_score, score_gap, aug_score_gap = self._test_one_batch(batch_size, episode)
            used_time = time.time() - st

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)
            gap_AM.update(score_gap, batch_size)
            aug_gap_AM.update(aug_score_gap, batch_size)
            time_AM.update(used_time, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))
            self.logger.info("score_gap:{:.3f} %, aug_score_gap:{:.3f} %".format(score_gap * 100, aug_score_gap * 100))
            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" TIME: {:.4f} ".format(time_AM.sum))
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))
                self.logger.info(" NO-AUG GAP: {:.4f} %".format(gap_AM.avg * 100))
                self.logger.info(" AUGMENTATION GAP: {:.4f} %".format(aug_gap_AM.avg * 100))

    def _test_one_batch(self, batch_size, episode=0):
        # f_path = f'./data/tsp_n{self.env.problem_size}_{episode}.pkl'
        f_path = self.tester_params.get('local_dataset_path', None)
        use_local_dataset = self.tester_params.get('use_local_dataset', False)
        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor, f_path, use_local_dataset, episode,
                                   device=self.device) 
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # Rollout
        ###############################################
        state, goal, done, selected_node_list = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, B)
            state, goal, done, selected_node_list = self.env.step(selected)

        # Return
        ###############################################
        # ground_truth = self.env.gurobi_test_tsp()
        
        # if not os.path.exists(f_path):
        #     self.env.save(f_path)

        # print("goal.shape: ", goal.shape)
        aug_goal = goal.reshape(aug_factor, batch_size, self.env.sols_num)
        # print("aug_goal.shape: ", aug_goal.shape)
        # shape: (augmentation, batch, B)

        max_pomo_goal, max_pomo_goal_index = aug_goal.max(dim=2)  # get best results
        # shape: (augmentation, batch)
        # print("max_pomo_goal.shape: ", max_pomo_goal.shape)
        # print("ground_truth.shape: ", ground_truth.shape)
        # no_aug_gap = (-max_pomo_goal[0, :] / ground_truth[:max_pomo_goal.shape[1]] - 1 ).mean()
        no_aug_score = -max_pomo_goal[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_goal, max_aug_pomo_goal_index = max_pomo_goal.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        # aug_gap = (-max_aug_pomo_goal / ground_truth[:max_pomo_goal.shape[1]] - 1).mean()
        aug_score = -max_aug_pomo_goal.float().mean()  # negative sign to make positive value

        # Save solution file
        if use_local_dataset:
            selected_node_list = selected_node_list.reshape(aug_factor, batch_size, self.env_params['problem_size'], -1)
            # shape: (aug_factor, batch, problem_size, problem_size)

            expanded_idx = max_pomo_goal_index.long().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, selected_node_list.size(-1))
            # shape: (aug_factor, batch, problem_size, 1)

            max_aug_goal_selected = torch.gather(selected_node_list, dim=2, index=expanded_idx).squeeze(2)
            # shape: (aug_factor, batch, problem_size)

            max_no_aug_selected = max_aug_goal_selected[0, :, :]
            # shape: (batch, problem_size)

            solutions_no_aug_dir = os.path.join(get_result_folder(), "solutions_no_aug")
            self._save_solution_file(solutions_no_aug_dir, batch_size, self.env_params['problem_size'], episode, max_no_aug_selected, -max_pomo_goal[0, :])

            max_aug_selected = max_aug_goal_selected[max_aug_pomo_goal_index, torch.arange(max_aug_goal_selected.size(1)), :]
            # shape: (batch, problem_size)
            solutions_aug_dir = os.path.join(get_result_folder(), "solutions_aug")
            self._save_solution_file(solutions_aug_dir, batch_size, self.env_params['problem_size'], episode, max_aug_selected, -max_aug_pomo_goal)

        # return no_aug_score.item(), aug_score.item(), no_aug_gap.item(), aug_gap.item()
        return no_aug_score.item(), aug_score.item(), 0, 0
    
    def _save_solution_file(self, solutions_dir, batch_size, problem_size, episode, max_no_aug_selected, max_rewards):
        """
        将 max_no_aug_selected 保存为 .sol 文件。
        :param solutions_dir: 解决方案目录
        :param batch_size: 批次大小
        :param problem_size: 问题规模（节点数量）
        :param episode: 起始序号
        :param max_no_aug_selected: 选择的节点序列张量，形状为 (batch_size, problem_size)
        """
        os.makedirs(solutions_dir, exist_ok=True)  # 确保解决方案目录存在

        # 只使用第一个扩增的节点序列

        for i in range(batch_size):
            numbers = i + 1 + episode
            solution_path = os.path.join(solutions_dir, f"dataset_n{problem_size}_b{numbers}.sol")

            # 提取当前批次的节点序列
            tour = max_no_aug_selected[i, :].tolist()  # 假设节点序列存储在第一个维度

            max_reward = max_rewards[i].item()
            # 写入 .sol 文件
            with open(solution_path, 'w') as file:
                file.write(f"{problem_size} {max_reward}\n")  # 写入节点数量
                # 每行写入10个节点，节点之间用空格分隔
                for j in range(0, len(tour), 10):  # 每10个节点为一组
                    # 获取当前组的节点（最多10个）
                    nodes_group = tour[j:j + 10]
                    # 将节点转换为字符串，并用空格分隔
                    nodes_str = " ".join(map(str, map(int, nodes_group)))
                    # 写入文件，每个组占一行
                    file.write(nodes_str + "\n")


class TSPLIBTester:
    def __init__(self,
                 model_params,
                 tester_params):

        # save arguments
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL

        self.env = Env()

        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()

        time_AM = {
            '<100' : AverageMeter(),
            '100~200' : AverageMeter(),
            '200~500' : AverageMeter(),
            '500~1000' : AverageMeter(),
        }
        score_AM = {
            '<100' : AverageMeter(),
            '100~200' : AverageMeter(),
            '200~500' : AverageMeter(),
            '500~1000' : AverageMeter(),
        }
        aug_score_AM = {
            '<100' : AverageMeter(),
            '100~200' : AverageMeter(),
            '200~500' : AverageMeter(),
            '500~1000' : AverageMeter(),
        }
        gap_AM = {
            '<100' : AverageMeter(),
            '100~200' : AverageMeter(),
            '200~500' : AverageMeter(),
            '500~1000' : AverageMeter(),
        }
        aug_gap_AM = {
            '<100' : AverageMeter(),
            '100~200' : AverageMeter(),
            '200~500' : AverageMeter(),
            '500~1000' : AverageMeter(),
        }
        
        def size2size(size):
            if size < 100:
                return '<100'
            elif size < 200:
                return '100~200'
            elif size < 500:
                return '200~500'
            elif size < 1000:
                return '500~1000'
            else:
                return '>=1000'

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1
        self.aug_factor = aug_factor

        batch_size = 1
        for name, size, opt in self.env.load_tsplib(batch_size, aug_factor):
            self.logger.info("tsplib: {}".format(name))

            st = time.time()
            score, aug_score, score_gap, aug_score_gap = self._test_one_batch(batch_size)
            used_time = time.time() - st

            size = size2size(size)
            time_AM[size].update(used_time, batch_size)
            score_AM[size].update(score, batch_size)
            aug_score_AM[size].update(aug_score, batch_size)
            gap_AM[size].update(score_gap, batch_size)
            aug_gap_AM[size].update(aug_score_gap, batch_size)

            ############################
            # Logs
            ############################
            self.logger.info("opt: {:.3f}, score:{:.3f}, aug_score:{:.3f}".format(
                opt, score, aug_score))
            self.logger.info("score_gap:{:.3f}, aug_score_gap:{:.3f}".format(score_gap, aug_score_gap))

        for size in score_AM:
            self.logger.info(" *** Test Done size {} x {} *** ".format(size, score_AM[size].count))
            self.logger.info(" TIME: {:.4f} ".format(time_AM[size].avg))
            self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM[size].avg))
            self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM[size].avg))
            self.logger.info(" NO-AUG GAP: {:.4f} ".format(gap_AM[size].avg * 100))
            self.logger.info(" AUGMENTATION GAP: {:.4f} ".format(aug_gap_AM[size].avg * 100))

    def _test_one_batch(self, batch_size):
        # Ready
        ###############################################
        # self.model.eval()
        with torch.no_grad():
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # Rollout
        ###############################################
        state, goal, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, B)
            state, goal, done = self.env.step(selected)

        # Return
        ###############################################
        # ground_truth = self.env.gurobi_test_tsp()

        aug_goal = goal.reshape(self.aug_factor, batch_size, self.env.sols_num)
        # shape: (augmentation, batch, B)

        max_pomo_goal, _ = aug_goal.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        # no_aug_gap = (-max_pomo_goal[0, :] / ground_truth - 1 ).mean()
        no_aug_score = -max_pomo_goal[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_goal, _ = max_pomo_goal.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        # aug_gap = (-max_aug_pomo_goal / ground_truth - 1).mean()
        aug_score = -max_aug_pomo_goal.float().mean()  # negative sign to make positive value

        # return no_aug_score.item(), aug_score.item(), no_aug_gap.item(), aug_gap.item()
        return no_aug_score.item(), aug_score.item(), 0, 0

