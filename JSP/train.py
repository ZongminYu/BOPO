import torch
import random
import numpy as np
import sampling as stg
from PointerNet import CAMEncoder3, LSTMDecoder2
from argparse import ArgumentParser
from inout import load_dataset
from tqdm import tqdm
from time import time
from utils import *


import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Number of steps to wait before probing for improvements
PROBE_EVERY = 2500


@torch.no_grad()
def validation(encoder: torch.nn.Module,
               decoder: torch.nn.Module,
               val_set: list,
               device,
               use_greedy: bool = False,
               num_sols: int = 16,):
    """
    Test the model at the end of each epoch.

    Args:
        encoder: Encoder.
        decoder: Decoder.
        val_set: Validation set.
        num_sols: Number of solution to generate for each instance.
        seed: Random seed.
    """
    encoder.eval()
    decoder.eval()
    gaps = ObjMeter('Gap')
    ent = ObjMeter('Entropy')

    # For each instance in the benchmark
    for ins in val_set:
        # Sample multiple solutions
        s, mss, entropies, _ = stg.sampling(ins, encoder, decoder, 
                                              bs=num_sols, device=device, 
                                              use_greedy=use_greedy)

        # Log info
        min_gap = (mss.min().item() / ins['makespan'] - 1) * 100
        gaps.update(ins, min_gap)
        ent.update(ins, entropies.mean().item())

    # Print stats
    avg_gap = gaps.avg
    print(f"\t\tVal set: AVG Gap={avg_gap:.3f}")
    print(gaps)
    print(f"\t\tVal set: AVG Entropy={ent.avg:.3f}")
    print(ent)
    return avg_gap


def train(encoder: torch.nn.Module,
          decoder: torch.nn.Module,
          train_set: list,
          val_set: list,
          device,
          epochs: int = 20,
          num_cmp: int = 6,
          num_sols: int = 128,
          val_sols: int = 128,
          model_path: str = 'checkpoints/PointerNet.pt'):
    """
    Train the Pointer Network.

    Args:
        encoder: Encoder to train.
        decoder: Decoder to train.
        train_set: Training set.
        val_set: Validation set.
        epochs: Number of epochs.
        virtual_bs: Virtual batch size that gives the number of instances
            predicted before back-propagation.
        num_sols: Number of solutions to use in back-propagation.
        model_path:
    """
    _best =  None
    size = len(train_set)
    indices = list(range(size))
    ### OPTIMIZER
    opti = torch.optim.Adam(list(_enc.parameters()) +
                            list(_dec.parameters()), lr=args.lr)
    #
    print("Training ...")
    for epoch in range(epochs):
        print(f"Epoch {epoch} start ...")
        losses = AverageMeter()
        f_gaps = AverageMeter()
        gaps = ObjMeter()
        np.random.shuffle(indices)
        cnt = 0
        st = stt = time()

        # For each instance in the training set
        for idx, i in enumerate(indices):
            if idx % 50 == 0:
                print(f"\tsteps:{idx}/{size} ({time()-st:.2f}s)")
                st = time()

            ins = train_set[i]
            cnt += 1

            # Training step (sample solutions)
            info_winner, info_loser, ms = stg.sample_training_pair(
                                                            ins, encoder, decoder, 
                                                            bs=num_sols, K=num_cmp,
                                                            use_greedy=args.use_greedy,
                                                            device=device)

            # Compute loss: self labeling optimization
            loss, max_f = stg.SROLoss(info_winner, info_loser)

            # Normal backpropagation
            loss.backward()
            opti.step()
            opti.zero_grad()

            # log info
            losses.update(loss.item())
            f_gaps.update(max_f)
            gaps.update(ins, (ms.item() / ins['makespan'] - 1) * 100)

            # Probe model
            if idx > 0 and idx % PROBE_EVERY == 0:
                val_gap = validation(encoder, decoder, val_set, num_sols=val_sols, device=device, use_greedy=args.use_greedy)
                
                if _best is None or val_gap < _best:
                    _best = val_gap
                    torch.save((encoder.state_dict(), decoder), model_path)
                    print(f'\t\tModel saved in epoch {epoch} / step {idx} / iter {cnt}')

        # ...log the running loss
        avg_loss, avg_gap, avg_f = losses.avg, gaps.avg, f_gaps.avg
        logger.train(epoch, avg_loss, avg_gap)
        print(f'\tEPOCH {epoch:02}: avg loss={avg_loss:.4f}, avg f={avg_f:.4f} ({(time()-stt)/3600:.2f}h)')
        print(f'\t\tGreedy used: {greedy_used/size}; Greedy best: {greedy_best/size}')
        print(f"\t\tTrain: AVG Gap={avg_gap:2.3f}")
        print(gaps)
        greedy_used = greedy_best = 0

        # Test model and save
        val_gap = validation(encoder, decoder, val_set, num_sols=val_sols, device=device, use_greedy=args.use_greedy)
        logger.validation(val_gap)
        if _best is None or val_gap < _best:
            _best = val_gap
            torch.save((encoder.state_dict(), decoder), model_path)
            print(f'\t\tModel saved in epoch {epoch} / step {idx} / iter {cnt}')
        #
        logger.flush()

#
parser = ArgumentParser(description='BOPO for the JSP')
parser.add_argument("-data_path", type=str, default="./dataset5k",
                    required=False, help="Path to the training data.")
parser.add_argument("-model_path", type=str, required=False,
                    default=None, help="Path to the model.")
parser.add_argument("-enc_hidden", type=int, default=64, required=False,
                    help="Hidden size of the encoder.")
parser.add_argument("-enc_out", type=int, default=128, required=False,
                    help="Output size of the encoder.")
parser.add_argument("-mem_hidden", type=int, default=64, required=False,
                    help="Hidden size of the memory network.")
parser.add_argument("-mem_out", type=int, default=128, required=False,
                    help="Output size of the memory network.")
parser.add_argument("-clf_hidden", type=int, default=128, required=False,
                    help="Hidden size of the classifier.")
parser.add_argument("-lr", type=float, default=0.00002, required=False,
                    help="Learning rate in the first checkpoint.")
parser.add_argument("-epochs", type=int, default=1, required=False,
                    help="Number of epochs.")
parser.add_argument("-K", type=int, default=16, required=False,
                    help="Number of solutions to collect comparison.")
parser.add_argument("-B", type=int, default=256, required=False,
                    help="Number of sampled solutions.")
parser.add_argument("-B2", type=int, default=256, required=False,
                    help="Number of sampled solutions.")
parser.add_argument("-device", type=int, default=0, required=False,
                    help="The GPU card.")
parser.add_argument("-seed", type=int, default=None, required=False,
                    help="The random seed.")
parser.add_argument("-tag", type=str, default="Test", required=False, 
                    help="The tag of training model.")
parser.add_argument("-greedy", type=int, default=1, required=False,
                    help="Sample one greedy sol into the sol set.")
args = parser.parse_args()
args.use_greedy = args.greedy != 0

if args.seed is None:
    args.seed = random.randint(0, 10000)

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.set_device(args.device)
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

print(args)
#
run_name = f"CamLstmNet-K{args.K}-B{args.B}-{args.tag}"
logger = Logger(run_name)

if __name__ == '__main__':
    # Training device
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    ### TRAINING and VALIDATION
    train_set = load_dataset(args.data_path, split=True)
    val_set = load_dataset('./benchmarks/validation', device=device, split=True)

    ### MAKE MODEL
    _enc = CAMEncoder3(train_set[0]['x'].shape[1],
                      hidden_size=args.enc_hidden,
                      embed_size=args.enc_out).to(device)
    _dec = LSTMDecoder2(encoder_size=_enc.out_size,
                      context_size=stg.JobShopStates.size,
                      hidden_size=args.mem_hidden,
                      att_size=args.mem_out).to(device)

    # Load model if necessary
    if args.model_path is not None:
        print(f"Loading {args.model_path}.")
        m_path = f"{args.model_path}"
        _enc_w, _dec = torch.load(args.model_path, map_location=device)
        _enc.load_state_dict(_enc_w)
    else:
        m_path = f"checkpoints/{run_name}.pt"

        if os.path.exists(f"checkpoints/{run_name}.pt"):
            raise RuntimeError(f"The model {run_name} has been trained.")

    print(_enc)
    print(_dec)

    #
    train(_enc, _dec, train_set, val_set,
          epochs=args.epochs,
          num_sols=args.B,
          num_cmp=args.K,
          model_path=m_path,
          val_sols=args.B2,
          device=device)
