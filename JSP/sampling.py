import torch
import torch.nn.functional as F

import numpy as np
from dataclasses import dataclass
from utils import find_nearest_distances, shift


@dataclass
class Solutions:
    mss: torch.Tensor
    logits: torch.Tensor
    trajs: torch.Tensor


def SROLoss(info_w: Solutions, info_l: Solutions) -> torch.Tensor:
    logits_w = info_w.logits
    targets_w = info_w.trajs
    ms_w = info_w.mss

    logits_l = info_l.logits
    targets_l = info_l.trajs
    ms_l = info_l.mss

    bs, s, n = logits_w.shape
    logits_w = logits_w.view(-1, n)
    targets_w = targets_w.view(-1)
    logits_l = logits_l.view(-1, n)
    targets_l = targets_l.view(-1)

    log_probs_w = F.log_softmax(logits_w, dim=-1)
    log_probs_l = F.log_softmax(logits_l, dim=-1)

    # CrossEntropyLoss / nll loss
    loss_w = log_probs_w[torch.arange(bs*s), targets_w].view(bs, s).mean(dim=-1)
    loss_l = log_probs_l[torch.arange(bs*s), targets_l].view(bs, s).mean(dim=-1)

    # SRO loss
    ms_factor = ms_l / ms_w
    loss = - torch.log( torch.sigmoid( ms_factor * (loss_w-loss_l) ) ).mean()
    
    return loss, ms_factor.max().item()


class JobShopStates:
    """
    Job Shop state for parallel executions.

    Args:
        device: Where to create tensors.
    """
    # Number of features in the internal state
    size = 11

    def __init__(self, device: str = 'cpu', eps: float = 1e-5):
        self.num_j = None       # Number of jobs
        self.num_m = None       # Number of machines
        self.machines = None    # Machine assigment of each operation
        self.costs = None       # Cost of each operation
        self._factor = None     # Max cost
        self._eps = eps
        self._q = torch.tensor([0.25, 0.5, 0.75], device=device)
        #
        self.dev = device       # Tensor device
        self._bs_idx = None     # Batch index for accessing info
        self.bs = None          # Batch size
        #
        self.j_ct = None        # Completion time of jobs in the partial sol
        self.j_idx = None       # Index of active operation in jobs
        self.j_st = None
        #
        self.m_ct = None        # Completion time of machines in the partial sol
        #
        self.history = None     # History of the operations

    def init_state(self, ins: dict, bs: int = 1, use_aug: bool = False):
        """
        Initialize the state of the job shop.

        Args:
            ins: JSP instance.
            bs: Batch size (number of parallel states).
        Return:
            - The parallel states.
            - The mask of active operations for each state.
        """
        self.num_j, self.num_m = ins['j'], ins['m']
        self.machines = ins['machines'].view(-1).to(self.dev)
        self._factor = ins['costs'].max()
        self.costs = ins['costs'].view(-1).to(self.dev) / self._factor
        self.bs = bs
        self._bs_idx = torch.arange(bs, device=self.dev)
        #
        self.j_st = torch.arange(0, self.num_j * self.num_m, self.num_m,
                                 device=self.dev)
        self.j_idx = torch.zeros((bs, self.num_j), dtype=torch.int32,
                                 device=self.dev)
        self.j_ct = torch.zeros((bs, self.num_j), dtype=torch.float32,
                                device=self.dev)
        #
        self.m_ct = torch.zeros((bs, self.num_m), dtype=torch.float32,
                                device=self.dev)

        # Create the initial state and mask
        states = torch.zeros((bs, self.num_j, self.size), dtype=torch.float32,
                             device=self.dev)

        # Discord the history of the jobs & machines
        self.use_aug = use_aug
        if use_aug:
            self.count = 0
            self.history = -np.ones((bs, self.num_m*self.num_j, 2), dtype=np.int32)
        return states, self.mask.to(torch.float32)

    @property
    def mask(self):
        """
        Boolean mask that points out the uncompleted jobs.

        Return:
            Tensor with shape (bs, num jobs).
        """
        return self.j_idx < self.num_m

    @property
    def ops(self):
        """
        The index of active/ready operations for each job.
        Note that for the completed job the active operation is the one with
        index 0.

        Return:
            Tensor with shape (bs, num jobs).
        """
        return self.j_st + (self.j_idx % self.num_m)

    @property
    def makespan(self):
        """
        Compute makespan of solutions.
        """
        return self.m_ct.max(-1)[0] * self._factor

    def __schedule__(self, jobs: torch.Tensor):
        """ Schedule the selected jobs and update completion times. """
        _idx = self._bs_idx           # Batch index
        _ops = self.ops[_idx, jobs]   # Active operations of selected jobs
        macs = self.machines[_ops]    # Machines of active operations

        if self.use_aug:
            self.history[_idx.cpu().numpy(), self.count, 0] = jobs.detach().cpu().numpy()
            self.history[_idx.cpu().numpy(), self.count, 1] = macs.cpu().numpy()
            self.count += 1

        # Update completion times
        ct = torch.maximum(self.m_ct[_idx, macs],
                           self.j_ct[_idx, jobs]) + self.costs[_ops]
        self.m_ct[_idx, macs] = ct
        self.j_ct[_idx, jobs] = ct

        # Activate the following operation on job, if any
        self.j_idx[_idx, jobs] += 1

    def update(self, jobs: torch.Tensor):
        """
        Update the internal state.

        Args:
            jobs: Index of the job scheduled at the last step.
                Shape (batch size).
        """
        # Schedule the selected operations
        self.__schedule__(jobs)

        _idx = self._bs_idx  # Batch index
        job_mac = self.machines[self.ops]  # Machines of active ops
        mac_ct = self.m_ct.gather(1, job_mac)  # Completion time of machines
        curr_ms = self.j_ct.max(-1, keepdim=True)[0] + self._eps
        #
        n_states = -torch.ones((self.bs, self.num_j, self.size),
                               device=self.dev)
        n_states[..., 0] = self.j_ct - mac_ct
        # Distance of each job from quantiles computed among all jobs
        q_j = torch.quantile(self.j_ct, self._q, -1).T
        n_states[..., 1:4] = self.j_ct.unsqueeze(-1) - q_j.unsqueeze(1)
        n_states[..., 4] = self.j_ct - self.j_ct.mean(-1, keepdim=True)
        n_states[..., 5] = self.j_ct / curr_ms
        # Distance of each job from quantiles computed among all machines
        q_m = torch.quantile(self.m_ct, self._q, -1).T
        n_states[..., 6:9] = mac_ct.unsqueeze(-1) - q_m.unsqueeze(1)
        n_states[..., 9] = mac_ct - self.m_ct.mean(-1, keepdim=True)
        n_states[..., 10] = mac_ct / curr_ms

        return n_states, self.mask.to(torch.float32)

    def __call__(self, jobs: torch.Tensor, states: torch.Tensor):
        """
        Update the internal state at inference.

        Args:
            jobs: Index of the job scheduled at the last step.
                Shape (batch size).
        """
        # Schedule the selected operations
        self.__schedule__(jobs)

        _idx = self._bs_idx  # Batch index
        job_mac = self.machines[self.ops]  # Machines of active ops
        mac_ct = self.m_ct.gather(1, job_mac)  # Completion time of machines
        curr_ms = self.j_ct.max(-1, keepdim=True)[0] + self._eps
        #
        states[..., 0] = self.j_ct - mac_ct
        # Distance of each job from quantiles computed among all jobs
        q_j = torch.quantile(self.j_ct, self._q, -1).T
        states[..., 1:4] = self.j_ct.unsqueeze(-1) - q_j.unsqueeze(1)
        states[..., 4] = self.j_ct - self.j_ct.mean(-1, keepdim=True)
        states[..., 5] = self.j_ct / curr_ms
        # Distance of each job from quantiles computed among all machines
        q_m = torch.quantile(self.m_ct, self._q, -1).T
        states[..., 6:9] = mac_ct.unsqueeze(-1) - q_m.unsqueeze(1)
        states[..., 9] = mac_ct - self.m_ct.mean(-1, keepdim=True)
        states[..., 10] = mac_ct / curr_ms

        return self.mask.to(torch.float32)

    def augment(self, aug_idx: int, aug_num: int):
        if not self.use_aug:
            raise RuntimeError('Augmentation is not enabled.')
        his = self.history[aug_idx]
        jobs = his[..., 0]
        macs = his[..., 1]
        # drop the tail
        jobs = jobs[:-1]
        macs = macs[:-1]
        
        job_l, job_r = find_nearest_distances(jobs)
        mac_l, mac_r = find_nearest_distances(macs)

        l_bias = np.where(job_l < mac_l, job_l, mac_l)
        r_bias = np.where(job_r < mac_r, job_r, mac_r)

        bias = l_bias + r_bias

        if np.sum(bias) < aug_num:
            print(jobs)
            print(macs)
            print('WARNING! No enough bias for augmentation.')
            raise RuntimeError
        
        new_jobs = -np.ones((aug_num, jobs.shape[0]), dtype=np.int32)

        bias = np.stack([l_bias, r_bias], axis=1) 
        idxs = np.unravel_index(np.argsort(-bias, axis=None), bias.shape)
        # (idx, 0/1): 0 for left, 1 for right

        count = 0
        for idx, right in zip(*idxs):
            if right:
                b = r_bias[idx]
            else:
                b = - l_bias[idx]
            jobs_ = shift(jobs, idx, b)
            new_jobs[count] = jobs_
            count += 1

            if count >= aug_num:
                break
        
        return new_jobs


@torch.no_grad()
def greedy(ins: dict,
           encoder: torch.nn.Module,
           decoder: torch.nn.Module,
           device: str = 'cpu'):
    """
    Sampling at inference.

    Args:
        ins: The instance to solve.
        encoder: Pointer Encoder.
        decoder: Pointer Decoder
        device: Either cpu or cuda.
    """
    num_j, num_m = ins['j'], ins['m']
    machines = ins['machines'].view(-1)
    encoder.eval()
    decoder.eval()

    # Reserve space for the solution
    sols = -torch.ones((num_m, num_j), dtype=torch.long, device=device)
    m_idx = torch.zeros(num_m, dtype=torch.long, device=device)
    #
    jsp = JobShopStates(device)
    state, mask = jsp.init_state(ins, 1)

    # Encoding step
    embed = encoder(ins['x'], edge_index=ins['edge_index'])

    # Decoding steps, (in the last step, there is only one job to schedule)
    for i in range(num_j * num_m - 1):
        # Take the embeddings of ready operations and generate probabilities
        ops = jsp.ops
        logits = decoder(embed[ops], state) + mask.log()
        scores = F.softmax(logits, -1)

        # Select the next (masked) operation to be scheduled
        jobs = scores.max(1)[1]

        # Add the selected operations to the solution matrices
        s_ops = ops[0, jobs]
        m = machines[s_ops]
        s_idx = m_idx[m]
        sols[m, s_idx] = s_ops
        m_idx[m] += 1
        # Update the context of the solutions
        mask = jsp(jobs, state)

    # Schedule last job/operation
    jsp(mask.float().argmax(-1), state)
    return sols, jsp.makespan


def sample_training_pair(
                        ins: dict,
                        encoder: torch.nn.Module,
                        decoder: torch.nn.Module,
                        bs: int = 32,
                        K: int = 6,
                        use_greedy: bool = False,
                        device: str = 'cpu'):
    """
    Sample multiple trajectories while training.

    Args:
        ins: The instance to solve.
        encoder: Pointer Encoder.
        decoder: Pointer Decoder
        bs: Batch size (number of parallel solutions to create).
        K: Number of solutions to collect comparison.
        use_greedy: Use hybrid rollout.
        device: Either cpu or cuda.
    """
    encoder.train()
    decoder.train()
    num_j, num_m = ins['j'], ins['m']
    # We don't need to learn from the last step, everything is masked but a job
    num_ops = num_j * num_m - 1

    # Reserve space for the solution
    trajs = -torch.ones((bs, num_ops), dtype=torch.long, device=device)
    ptrs = -torch.ones((bs, num_ops, num_j), dtype=torch.float32,
                       device=device)
    #
    jsp = JobShopStates(device)
    state, mask = jsp.init_state(ins, bs)

    # Encoding step
    embed = encoder(ins['x'].to(device), 
                    job_edges=ins['job_edges'].to(device), 
                    mac_edges=ins['mac_edges'].to(device))

    last_ops = h = c = None
    # Decoding steps
    for i in range(num_ops):
        # Generate logits and mak the completed jobs
        ops = jsp.ops
        if last_ops is None:
            zeros = torch.zeros((bs, 1, encoder.out_size), dtype=torch.float32).to(device)
            logits, (h, c) = decoder(embed[ops], state, zeros, h, c)
        else:
            logits, (h, c) = decoder(embed[ops], state, embed[last_ops], h, c)
        logits = logits + mask.log()

        policies = F.softmax(logits, -1)

        # Select the next (masked) operation to be scheduled
        jobs = policies.multinomial(1, replacement=False).squeeze(1)
        if use_greedy:
            # Leave one to greedy
            jobs[0] = policies[0].argmax()

        # Add the node and pointers to the solution
        trajs[:, i] = jobs
        ptrs[:, i] = logits
        #
        last_ops = jsp.ops.gather(1, jobs.unsqueeze(-1))
        state, mask = jsp.update(jobs)

    # Schedule last job/operation
    jsp(mask.float().argmax(-1), state)

    # Generate the solution pairs (pi_better, pi_worse)
    makespan = jsp.makespan
    
    # Regular sample the solutions
    assert bs % K == 0
    all_idx = [i for i in range(bs)]
    all_idx.sort(key=lambda x: makespan[x].item())
    idx = all_idx[::bs//K]

    num_pairs = K - 1

    trajs_w = -torch.ones((num_pairs, num_ops), dtype=torch.long, device=device)
    ptrs_w = -torch.ones((num_pairs, num_ops, num_j), dtype=torch.float32, device=device)
    ms_w = torch.ones((num_pairs), dtype=torch.float32, device=device)
    #
    trajs_l = -torch.ones((num_pairs, num_ops), dtype=torch.long, device=device)
    ptrs_l = -torch.ones((num_pairs, num_ops, num_j), dtype=torch.float32, device=device)
    ms_l = torch.ones((num_pairs), dtype=torch.float32, device=device)

    count = 0
    for i in idx[1:]:
        trajs_w[count] = trajs[idx[0]]
        ptrs_w[count] = ptrs[idx[0]]
        ms_w[count] = makespan[idx[0]]
        #
        trajs_l[count] = trajs[i]
        ptrs_l[count] = ptrs[i]
        ms_l[count] = makespan[i]
        count += 1

    winner_info = Solutions(trajs=trajs_w, logits=ptrs_w, mss=ms_w)
    loser_info = Solutions(trajs=trajs_l, logits=ptrs_l, mss=ms_l)
    return winner_info, loser_info, torch.min(jsp.makespan)


@torch.no_grad()
def sampling(
            ins: dict,
            encoder: torch.nn.Module,
            decoder: torch.nn.Module,
            bs: int = 32,
            use_greedy: bool = False,
            device: str = 'cpu'):
    """
    Sampling at inference.

    Args:
        ins: The instance to solve.
        encoder: Pointer Encoder.
        decoder: Pointer Decoder
        bs: Batch size  (number of parallel solutions to create).
        device: Either cpu or cuda.
    """
    num_j, num_m = ins['j'], ins['m']
    machines = ins['machines'].view(-1)
    encoder.eval()
    decoder.eval()

    # Reserve space for the solution
    sols = -torch.ones((bs, num_m, num_j), dtype=torch.long, device=device)
    _idx = torch.arange(bs, device=device)
    m_idx = torch.zeros((bs, num_m), dtype=torch.long, device=device)
    entropies = torch.zeros((bs, num_j*num_m - 1), dtype=torch.float32, device=device)
    log_probs = torch.zeros((bs, num_j*num_m - 1), dtype=torch.float32, device=device)
    #
    jsp = JobShopStates(device)
    state, mask = jsp.init_state(ins, bs)

    # Encoding step
    embed = encoder(ins['x'], job_edges=ins['job_edges'], mac_edges=ins['mac_edges'])
    last_ops = h = c = None

    # Decoding steps, (in the last step, there is only one job to schedule)
    for i in range(num_j * num_m - 1):
        #
        ops = jsp.ops
        if last_ops is None:
            zeros = torch.zeros((bs, 1, encoder.out_size), dtype=torch.float32).to(device)
            logits, (h, c) = decoder(embed[ops], state, zeros, h, c)
        else:
            logits, (h, c) = decoder(embed[ops], state, embed[last_ops], h, c)
        logits = logits + mask.log()

        probs = torch.distributions.Categorical(logits=logits)
        entropy = probs.entropy()

        # Select the next (masked) operation to be scheduled
        if bs != 1:
            jobs = probs.sample()
            if use_greedy:
                # Leave one to greedy
                jobs[0] = logits[0].argmax()
        else:
            jobs = logits[0].argmax().unsqueeze(0)

        # Add the selected operations to the solution matrices
        entropies[_idx, i] = entropy
        log_probs[_idx, i] = probs.log_prob(jobs)
        s_ops = ops[_idx, jobs]
        m = machines[s_ops]
        s_idx = m_idx[_idx, m]
        sols[_idx, m, s_idx] = s_ops
        m_idx[_idx, m] += 1
        # Update the context of the solutions
        last_ops = jsp.ops.gather(1, jobs.unsqueeze(-1))
        mask = jsp(jobs, state)

    # Schedule last job/operation
    jsp(mask.float().argmax(-1), state)
    log_probs = log_probs.sum(-1)
    return sols, jsp.makespan, entropies, log_probs


def solve_jsp(ins, bs, device, 
              encoder, decoder, 
              use_greedy=False, 
              use_aug=False,
              jobs_selected: torch.Tensor=None):
    num_j, num_m = ins['j'], ins['m']
    # We don't need to learn from the last step, everything is masked but a job
    num_ops = num_j * num_m - 1

    # Reserve space for the solution
    trajs = -torch.ones((bs, num_ops), dtype=torch.long, device=device)
    ptrs = -torch.ones((bs, num_ops, num_j), dtype=torch.float32,
                       device=device)
    #
    jsp = JobShopStates(device)
    state, mask = jsp.init_state(ins, bs, use_aug)

    # Encoding step
    embed = encoder(ins['x'].to(device), 
                    job_edges=ins['job_edges'].to(device), 
                    mac_edges=ins['mac_edges'].to(device))

    last_ops = h = c = None
    # Decoding steps
    for i in range(num_ops):
        # Generate logits and mak the completed jobs
        ops = jsp.ops
        if last_ops is None:
            zeros = torch.zeros((bs, 1, encoder.out_size), dtype=torch.float32).to(device)
            logits, (h, c) = decoder(embed[ops], state, zeros, h, c)
        else:
            logits, (h, c) = decoder(embed[ops], state, embed[last_ops], h, c)
        logits = logits + mask.log()

        # Select the next (masked) operation to be scheduled
        # jobs shape [bs]
        if jobs_selected is not None:
            jobs = jobs_selected[:, i]
        else:
            policies = F.softmax(logits, -1)

            jobs = policies.multinomial(1, replacement=False).squeeze(1)
            if use_greedy:
                # Leave one to greedy
                jobs[0] = policies[0].argmax()

        # Add the node and pointers to the solution
        trajs[:, i] = jobs
        ptrs[:, i] = logits
        #
        last_ops = jsp.ops.gather(1, jobs.unsqueeze(-1))
        state, mask = jsp.update(jobs)

    # Schedule last job/operation
    jsp(mask.float().argmax(-1), state)

    # Generate the solution pairs (pi_better, pi_worse)
    makespan = jsp.makespan

    return trajs, ptrs, makespan, jsp
