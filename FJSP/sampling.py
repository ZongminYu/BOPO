import torch
import torch.nn.functional as F

import numpy as np
from dataclasses import dataclass


@dataclass
class Solutions:
    mss: torch.Tensor
    logits: torch.Tensor
    trajs: torch.Tensor


def SROLoss(info_b: Solutions, info_w: Solutions) -> torch.Tensor:
    logits_b = info_b.logits
    targets_b = info_b.trajs
    ms_b = info_b.mss

    logits_w = info_w.logits
    targets_w = info_w.trajs
    ms_w = info_w.mss

    bs, s, n = logits_b.shape
    logits_b = logits_b.view(-1, n)
    targets_b = targets_b.view(-1)
    logits_w = logits_w.view(-1, n)
    targets_w = targets_w.view(-1)

    log_probs_b = F.log_softmax(logits_b, dim=-1)
    log_probs_w = F.log_softmax(logits_w, dim=-1)

    # log likelihood loss
    loss_b = log_probs_b[torch.arange(bs*s), targets_b].view(bs, s).mean(dim=-1)
    loss_w = log_probs_w[torch.arange(bs*s), targets_w].view(bs, s).mean(dim=-1)

    # POCO loss
    ms_factor = ms_w / ms_b
    loss = - torch.log( torch.sigmoid( ms_factor * (loss_b-loss_w) ) ).mean()

    return loss


class FlexibleJobShopStates:
    """
    Flexible Job Shop state for parallel executions.

    Args:
        device: Where to create tensors.
    """
    # Number of features in the internal state
    size = 5

    def __init__(self, device: str = 'cpu', eps: float = 1e-5):
        self.num_j = None       # Number of jobs
        self.num_o = None       # Number of jobs' operations
        self.num_m = None       # Number of machines
        self.num_assign = None  # Number of operations assigned to each job
        self.max_num_ops = None # Max number of operations assigned to a job
        self.action_dim = None  # Number of possible actions

        self.data = None        # Data of the instance: data[i, j, k] := Cost of Job i Operation j on Machine k
        self.machines = None    # Machine assigment of each operation
        self.costs = None       # Cost of each operation

        self.action_2_job = None    # Action to Job ID
        self.action_2_op = None     # Action to Operation ID
        self.ops_No = None          # Operation number

        self._factor = None     # Max cost
        self._eps = eps
        self._q = np.array([0.25, 0.5, 0.75])
        #
        self.dev = device       # Tensor device
        self.bs = None          # Batch size
        #
        self.job_ct = None      # Completion time of jobs in the partial sol
        self.job_idx = None     # Index of active operation in jobs
        #
        self.mac_ct = None        # Completion time of machines in the partial sol

    def init_state(self, ins: dict, bs: int = 1):
        """
        Initialize the state of the job shop.

        Args:
            ins: JSP instance.
            bs: number of parallel states.
        Return:
            - The parallel states.
            - The mask of active operations for each state.
        """
        self.data = ins['data']
        self.num_j, self.num_m, self.num_o = ins['j'], ins['m'], ins['o']
        self._factor = np.max(self.data)
        self.bs = bs
        
        # Completion times
        self.job_ct = np.zeros((bs, self.num_j), dtype=np.float32)
        self.mac_ct = np.zeros((bs, self.num_m), dtype=np.float32)
        
        # action to Job ID and Op ID
        self.num_assign = np.sum(self.data >= 0, axis=2)
        self.max_num_ops = np.max(self.num_assign)
        self.action_dim = self.num_j * self.max_num_ops
        self.action_2_job = np.zeros((self.action_dim,), dtype=np.int32)
        self.action_2_op = np.zeros((self.action_dim,), dtype=np.int32)
        tmp = 0
        for j in range(self.num_j):
            self.action_2_job[tmp : tmp + self.max_num_ops] = j
            self.action_2_op[tmp : tmp + self.max_num_ops] = np.arange(self.max_num_ops)
            tmp = tmp + self.max_num_ops

        # Get the info of operations
        # job+idx+op -> mac & pt
        self.ops_No = -np.ones((self.num_j, self.num_o, self.max_num_ops), dtype=np.int32)
        self.job_idx = np.zeros((bs, self.num_j), dtype=np.int32)

        self.machines = np.zeros((self.num_j, self.num_o, self.max_num_ops), dtype=np.int32)
        self.costs = -np.ones((self.num_j, self.num_o, self.max_num_ops), dtype=np.float32)
        
        No = 0
        for i in range(self.num_j):
            for j in range(self.num_o):
                count = 0
                for k in range(self.num_m):
                    if self.data[i, j, k] >= 0:
                        self.machines[i, j, count] = k
                        self.costs[i, j, count] = self.data[i, j, k]
                        self.ops_No[i, j, count] = No
                        No += 1
                        count += 1
        self.costs /= self._factor

        # Create the initial state and mask
        j_states = torch.zeros((bs, self.action_dim, self.size), dtype=torch.float32,
                                device=self.dev)
        m_states = torch.zeros((bs, self.action_dim, self.size), dtype=torch.float32,
                                device=self.dev)

        return (j_states, m_states), self.mask

    @property
    def mask(self):
        """
        Boolean mask that points out the assignable operations / actions.

        Return:
            Tensor with shape (bs, action_dim).
        """
        row_idx = np.arange(self.num_j).reshape(1, -1).repeat(self.bs, axis=0).flatten()
        job_idx = self.job_idx.flatten()
        _job_idx = job_idx % self.num_o
        num = np.where(job_idx < self.num_o, self.num_assign[row_idx, _job_idx], 0).reshape(self.bs, -1, 1)
        
        idx = np.arange(self.max_num_ops).reshape(1, 1, -1).repeat(self.bs, axis=0).repeat(self.num_j, axis=1)
        mask = np.where(idx < num, 1, 0).reshape(self.bs, -1)

        return torch.FloatTensor(mask).to(self.dev)

    @property
    def done(self):
        return torch.all(self.mask == 0)

    @property
    def ops(self):
        """
        The index of active/ready operations for each job.

        Return:
            Tensor with shape (bs, action_dim).
        """
        idx = np.arange(self.num_j).reshape(1, -1).repeat(self.bs, axis=0)
        ops = self.ops_No[idx, self.job_idx % self.num_o].reshape(self.bs, -1)
        return torch.LongTensor(ops).to(self.dev)

    @property
    def makespan(self) -> np.ndarray:
        """
        Compute makespan of solutions / partial solutions.
        """
        return self.mac_ct.max(-1) * self._factor

    def __schedule__(self, action: np.ndarray):
        """ Schedule the selected jobs and update completion times. """
        idx = np.arange(self.bs)           # Batch index
        jobs = self.action_2_job[action]   # Job index
        ops = self.action_2_op[action]     # Operation index
        job_idx = self.job_idx[idx, jobs]  # Index of active operation in jobs

        if job_idx.max() >= self.num_o:
            raise ValueError('Job already completed.')
        
        macs = self.machines[jobs, job_idx % self.num_o, ops]
        pts = self.costs[jobs, job_idx % self.num_o, ops]

        if np.any(pts < 0):
            raise ValueError('Operation is not assignable.')

        # Update completion times
        m_ct = self.mac_ct[idx, macs]
        j_ct = self.job_ct[idx, jobs]
        ct = np.where(m_ct > j_ct, m_ct, j_ct) + pts
        self.mac_ct[idx, macs] = ct
        self.job_ct[idx, jobs] = ct

        # Activate the following operation on job, if any
        self.job_idx[idx, jobs] += 1

    def update(self, action: torch.Tensor):
        """
        Update the internal state.

        Args:
            action: Index of the operations scheduled at this step.
                Shape (bs,).
        """
        # Schedule the selected operations
        action = action.cpu().numpy()
        self.__schedule__(action)
        #
        idx = np.arange(self.bs).reshape(-1, 1).repeat(self.action_dim, axis=1).flatten()
        jobs = self.action_2_job.reshape(1, -1).repeat(self.bs, axis=0).flatten()
        ops = self.action_2_op.reshape(1, -1).repeat(self.bs, axis=0).flatten()
        #
        # shape: (bs, action_dim)
        j_ct = self.job_ct[idx, jobs].reshape(self.bs, -1)
        # shape: (bs, 1)
        curr_ms = j_ct.max(axis=-1, keepdims=True) + self._eps
        #
        job_idx = self.job_idx[idx, jobs]
        macs = self.machines[jobs, job_idx % self.num_o, ops]
        # shape: (bs, action_dim)
        m_ct = self.mac_ct[idx, macs].reshape(self.bs, -1)
        #
        j_states = -np.ones((self.bs, self.action_dim, self.size), dtype=np.float32)
        m_states = -np.ones((self.bs, self.action_dim, self.size), dtype=np.float32)
        
        q_j = np.quantile(self.job_ct, self._q, -1).T
        j_states[..., 0] = j_ct / curr_ms
        j_states[..., 1] = j_ct - self.job_ct.mean(-1, keepdims=True)
        j_states[..., 2:5] = np.expand_dims(j_ct, 2) - np.expand_dims(q_j, 1)
        j_states = torch.FloatTensor(j_states).to(self.dev)

        q_m = np.quantile(self.mac_ct, self._q, -1).T
        m_states[..., 0] = m_ct / curr_ms
        m_states[..., 1] = m_ct - self.mac_ct.mean(-1, keepdims=True)
        m_states[..., 2:5] = np.expand_dims(m_ct, 2) - np.expand_dims(q_m, 1)
        m_states = torch.FloatTensor(m_states).to(self.dev)

        return (j_states, m_states), self.mask


def solve_problem(ins, bs, device, 
                  encoder, decoder, 
                  use_greedy=False,):
    num_j, num_o, num_m = ins['j'], ins['o'], ins['m']
    num_ops = num_j * num_o
    #
    fjsp = FlexibleJobShopStates(device)
    state, mask = fjsp.init_state(ins, bs)

    # Reserve space for the solution
    trajs = -torch.ones((bs, num_ops), dtype=torch.long, device=device)
    ptrs = -torch.ones((bs, num_ops, fjsp.action_dim), dtype=torch.float32,
                       device=device)

    # Encoding step
    embed = encoder(ins['x'].to(device), 
                    ops_egdes=ins['ops_edges'].to(device), 
                    job_edges=ins['job_edges'].to(device), 
                    mac_edges=ins['mac_edges'].to(device))
    zeros = torch.zeros((bs, 1, encoder.out_size), dtype=torch.float32,
                        device=device)

    last_ops = h = c = None
    # Decoding steps
    for i in range(num_ops):
        # Generate logits and mak the completed jobs
        ops = fjsp.ops
        if last_ops is None:
            logits, (h, c) = decoder(embed[ops], state, zeros, h, c)
        else:
            logits, (h, c) = decoder(embed[ops], state, embed[last_ops], h, c)
        logits = logits + mask.log()

        policies = torch.distributions.Categorical(logits=logits)

        # Select the next (masked) operation to be scheduled
        actions = policies.sample()
        if use_greedy:
            # Leave one to greedy
            actions[0] = logits[0].argmax()

        # Add the node and pointers to the solution
        trajs[:, i] = actions
        ptrs[:, i] = logits
        #
        last_ops = fjsp.ops.gather(1, actions.unsqueeze(-1))
        state, mask = fjsp.update(actions)
    assert fjsp.done

    return trajs, ptrs, fjsp.makespan, fjsp


def sample_training_pair(
                        ins: dict,
                        encoder: torch.nn.Module,
                        decoder: torch.nn.Module,
                        B: int = 32,
                        K: int = 16,
                        use_greedy: bool = True,
                        device: str = 'cpu'):
    """
    Sample multiple trajectories while training.

    Args:
        ins: The instance to solve.
        encoder: Pointer Encoder.
        decoder: Pointer Decoder
        bs: Batch size (number of parallel solutions to create).
        K: Number of solutions to collect comparison.
        use_greedy: Hybrid greedy & sampling rollout.
        device: Either cpu or cuda.
    """
    encoder.train()
    decoder.train()

    num_j, num_o, num_m = ins['j'], ins['o'], ins['m']
    num_ops = num_j * num_o
    
    trajs, ptrs, makespan, fjsp = solve_problem(ins, B, device,
                                                encoder, decoder,
                                                use_greedy=use_greedy,)

    # Generate the solution pairs (y_better, y_worse)
    # Regular sample the solutions
    assert B % K == 0
    all_idx = [i for i in range(B)]
    all_idx.sort(key=lambda x: makespan[x])
    idx = all_idx[::B//K]
    
    num_pairs = K - 1

    trajs_b = -torch.ones((num_pairs, num_ops), dtype=torch.long, device=device)
    ptrs_b = -torch.ones((num_pairs, num_ops, fjsp.action_dim), dtype=torch.float32, device=device)
    ms_b = torch.ones((num_pairs), dtype=torch.float32, device=device)

    trajs_w = -torch.ones((num_pairs, num_ops), dtype=torch.long, device=device)
    ptrs_w = -torch.ones((num_pairs, num_ops, fjsp.action_dim), dtype=torch.float32, device=device)
    ms_w = torch.ones((num_pairs), dtype=torch.float32, device=device)

    makespan_t = torch.FloatTensor(makespan).to(device)

    count = 0
    for i in idx[1:]:
        trajs_b[count] = trajs[idx[0]]
        ptrs_b[count] = ptrs[idx[0]]
        ms_b[count] = makespan_t[idx[0]]
        #
        trajs_w[count] = trajs[i]
        ptrs_w[count] = ptrs[i]
        ms_w[count] = makespan_t[i]
        count += 1
    
    assert count == num_pairs

    better_info = Solutions(trajs=trajs_b, logits=ptrs_b, mss=ms_b)
    worse_info = Solutions(trajs=trajs_w, logits=ptrs_w, mss=ms_w)
    return better_info, worse_info, makespan


@torch.no_grad()
def sampling(
            ins: dict,
            encoder: torch.nn.Module,
            decoder: torch.nn.Module,
            bs: int = 32,
            use_greedy: bool = True,
            device: str = 'cpu'):
    """
    Sampling at inference.

    Args:
        ins: The instance to solve.
        encoder: Pointer Encoder.
        decoder: Pointer Decoder
        bs: Batch size  (number of parallel solutions to create).
        use_greedy: Hybrid greedy & sampling rollout.
        device: Either cpu or cuda.
    """
    encoder.eval()
    decoder.eval()
    #
    _, _, makespan, _ = solve_problem(ins, bs, device,
                                      encoder, decoder,
                                      use_greedy=use_greedy,)
    return makespan

