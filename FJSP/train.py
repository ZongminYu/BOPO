import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import sampling as stg
from net import CAMEncoder, LSTMDecoder
from argparse import ArgumentParser
from inout import load_dataset
from time import time
from utils import *

# Number of steps to wait before probing for improvements
PROBE_EVERY = 2500
PRINT_EVERY = 100
DEBUG = True

parser = ArgumentParser(description='PointerNet arguments for the JSP')
### model arguments
parser.add_argument("-tag", type=str, required=True,
                    help="The tag of training model.")
parser.add_argument("-data_path", type=str, default="./dataset",
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
### training arguments
parser.add_argument("-lr", type=float, default=0.00002, required=False,
                    help="Learning rate in the first checkpoint.")
parser.add_argument("-epochs", type=int, default=5, required=False,
                    help="Number of epochs.")
parser.add_argument("-K", type=int, default=16, required=False,
                    help="Number of solutions to collect comparison.")
parser.add_argument("-B", type=int, default=128, required=False,
                    help="Number of sampled solutions.")
parser.add_argument("-greedy", type=int, default=1, required=False,
                    help="Use hybrid rollout or not.")
parser.add_argument("-device", type=int, default=0, required=False,
                    help="The GPU card.")
parser.add_argument("-seed", type=int, default=69, required=False,
                    help="The random seed.")
args = parser.parse_args()
args.use_greedy = args.greedy != 0

if DEBUG:
    PRINT_EVERY = 10
    args.epochs = 2
    args.B = 32
    args.K = 8
    # args.device = 'cpu'
    print("DEBUG MODE ON!!!")

args.device = f'cuda:{args.device}' \
             if torch.cuda.is_available() and isinstance(args.device, int) \
             else 'cpu'

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.set_device(args.device)
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

print(args)
#
if DEBUG:
    run_name = f"MGL-B{args.B}-K{args.K}-debug"
else:
    run_name = f"MGL-B{args.B}-K{args.K}-{args.tag}"
logger = Logger(run_name)

#

@torch.no_grad()
def validation(encoder: torch.nn.Module,
               decoder: torch.nn.Module,
               val_set: list,
               device,
               use_greedy: bool = False,
               B: int = 16,):
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
    has_gap = False
    gaps = ObjMeter('Gap')
    objs = ObjMeter('Obj.')

    # For each instance in the benchmark
    for ins in val_set:
        # Sample multiple solutions
        mss = stg.sampling(ins, encoder, decoder, 
                           bs=B, device=device, 
                           use_greedy=use_greedy)

        # Log info
        if 'makespan' in ins and ins['makespan'] is not None:
            min_gap = (mss.min() / ins['makespan'] - 1) * 100
            has_gap = True
        else:
            min_gap = mss.min()
        objs.update(ins, mss.min())
        gaps.update(ins, min_gap)

    # Print stats
    if has_gap:
        print(f"\t\tVal set: AVG Gap={gaps.avg:.3f}")
        print(gaps)
    else:
        print(f"\t\tVal set: AVG Obj.={objs.avg:.3f}")
        print(objs)
    return gaps.avg


def train(encoder: torch.nn.Module,
          decoder: torch.nn.Module,
          train_set: list,
          val_set: list,
          device,
          use_greedy: bool = False,
          epochs: int = 20,
          num_cmp: int = 6,
          num_sols: int = 128,
          val_sols: int = 256,
          model_path: str = 'checkpoints/PointerNet.pt'):
    """
    Train the Pointer Network.

    Args:
        encoder: Encoder to train.
        decoder: Decoder to train.
        train_set: Training set.
        val_set: Validation set.
        epochs: Number of epochs.
        num_cmp: Number of solutions to collect comparison.
        num_sols: Number of solutions to use in back-propagation.
        val_sols: Number of solutions to generate for validation.
        model_path: Save path for the model.
    """
    _best =  None
    size = len(train_set)
    indices = list(range(size))
    ### OPTIMIZER
    opti = torch.optim.Adam(list(encoder.parameters()) +
                            list(decoder.parameters()), lr=args.lr)
    #
    print("Training ...")
    for epoch in range(epochs):
        print(f"Epoch {epoch} start ...")
        losses = AverageMeter()
        gaps = ObjMeter()
        np.random.shuffle(indices)
        cnt = 0
        st = stt = time()

        # For each instance in the training set
        for idx, i in enumerate(indices):
            if idx % PRINT_EVERY == 0 and idx <= PROBE_EVERY:
                print(f"\tsteps:{idx}/{size} ({time()-st:.2f}s)")
                st = time()

            ins = train_set[i]
            cnt += 1

            # Training step (sample solutions)
            info_better, info_worse, mss = stg.sample_training_pair(
                                                            ins, encoder, decoder, 
                                                            B=num_sols, K=num_cmp,
                                                            use_greedy=use_greedy,
                                                            device=device)

            # Compute loss: self labeling optimization
            loss = stg.SROLoss(info_better, info_worse)

            # Normal backpropagation
            loss.backward()
            opti.step()
            opti.zero_grad()

            # log info
            losses.update(loss.item())
            gaps.update(ins, np.min(mss))

            # Probe model
            if idx > 0 and idx % PROBE_EVERY == 0:
                val_gap = validation(encoder, decoder, val_set, 
                                     B=val_sols, device=device, 
                                     use_greedy=use_greedy)
                if _best is None or val_gap < _best:
                    _best = val_gap
                    torch.save((encoder.state_dict(), decoder), model_path)
                    print(f'\t\tModel saved in epoch {epoch} / step {idx} / iter {cnt}')

        # ...log the running loss
        avg_loss, avg_gap = losses.avg, gaps.avg
        logger.train(epoch, avg_loss, avg_gap)
        print(f'\tEPOCH {epoch:02}: avg loss={avg_loss:.4f} ({(time()-stt)/3600:.2f}h)')
        print(f"\t\tTrain: AVG Obj.={avg_gap:2.3f}")
        print(gaps)

        # Test model and save
        val_gap = validation(encoder, decoder, val_set, 
                             B=val_sols, device=device, 
                             use_greedy=use_greedy)
        logger.validation(val_gap)
        if _best is None or val_gap < _best:
            _best = val_gap
            torch.save((encoder.state_dict(), decoder), model_path)
            print(f'\t\tModel saved in epoch {epoch} / step {idx} / iter {cnt}')
        #
        logger.flush()


if __name__ == '__main__':
    # Training device
    device = args.device
    print(f"Using device: {device}")

    ### TRAINING and VALIDATION
    if DEBUG:
        train_set = load_dataset(args.data_path, device='cpu', iter_num=100)
        val_set = load_dataset('./benchmarks/validation', device=device, iter_num=100)
    else:
        train_set = load_dataset(args.data_path, device='cpu')
        val_set = load_dataset('./benchmarks/validation', device=device)

    ### MAKE MODEL
    enc = CAMEncoder(train_set[0]['x'].shape[1],
                      hidden_size=args.enc_hidden,
                      embed_size=args.enc_out).to(device)
    dec = LSTMDecoder(encoder_size=enc.out_size,
                      context_size=stg.FlexibleJobShopStates.size,
                      hidden_size=args.mem_hidden,
                      att_size=args.mem_out).to(device)

    # Load model if necessary
    if args.model_path is not None:
        print(f"Loading {args.model_path}.")
        m_path = f"{args.model_path}"
        _enc_w, dec = torch.load(args.model_path, map_location=device)
        enc.load_state_dict(_enc_w)
    else:
        m_path = f"checkpoints/{run_name}.pt"

        if os.path.exists(f"checkpoints/{run_name}.pt") and not DEBUG:
            raise RuntimeError(f"The model {run_name} has been trained.")

    print(enc)
    print(dec)

    #
    train(enc, dec, train_set, val_set,
          epochs=args.epochs,
          num_sols=args.B,
          num_cmp=args.K,
          use_greedy=args.use_greedy,
          model_path=m_path,
          device=device)
