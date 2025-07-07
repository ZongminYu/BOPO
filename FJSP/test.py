import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import argparse
import pandas as pd

from sampling import sampling
from inout import load_data
from time import time
from utils import ObjMeter


@torch.no_grad()
def validation(encoder: torch.nn.Module,
               decoder: torch.nn.Module,
               ins,
               B: int = 32,
               seed: int = None):
    """

    Args:
        encoder: Encoder.
        decoder: Decoder.
        ins: JSP instance.
        B: Number of solution to generate for each instance.
        seed: Random seed.
    """
    if seed is not None:
        torch.manual_seed(seed)
    encoder.eval()
    decoder.eval()

    st = time()
    mss = sampling(ins, encoder, decoder, 
                   bs=B, device=dev, 
                   use_greedy=args.use_greedy)
    exe_time = time() - st

    #
    _gaps = (mss / ins['makespan'] - 1) * 100
    min_gap = _gaps.min()
    max_gap = _gaps.max()
    print(f'\t- {ins["name"]} = {min_gap:.3f}%~{max_gap:.3f}%')
    results = {'NAME': ins['name'],
               'UB': ins['makespan'],
               'MS': mss.min(),
               'MS-AVG': mss.mean(),
               'MS-STD': mss.std(),
               'GAP': min_gap,
               'GAP-MAX': max_gap,
               'GAP-AVG': _gaps.mean(),
               'GAP-STD': _gaps.std(),
               'TIME': exe_time}
    return results


#
parser = argparse.ArgumentParser(description='Test Pointer Net')
parser.add_argument("-model_path", type=str, required=False,
                    default="./checkpoints/MGL-B128-K16.pt",
                    help="Path to the model.")
parser.add_argument("-benchmark", type=str, required=False,
                    default='LA-e', help="Name of the benchmark for testing.")
parser.add_argument("-B", type=int, default=512, required=False,
                    help="Number of sampled solutions for each instance.")
parser.add_argument("-seed", type=int, default=69,
                    required=False, help="Random seed.")
parser.add_argument("-greedy", type=int, default=1,
                    required=False, help="Use greedy policy when sampling.")
parser.add_argument("-device", type=int, default=0, required=False,
                    help="The No. GPU card.")
args = parser.parse_args()
args.use_greedy = args.greedy != 0
print(args)

if __name__ == '__main__':
    from net import CAMEncoder
    # Testing device
    dev = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    print(f"Using {dev}...")

    # Load the model
    print(f"Loading {args.model_path}")
    enc_w, dec_ = torch.load(args.model_path, map_location=dev)
    enc_ = CAMEncoder(15).to(dev)   # Load weights to avoid bug with new PyG
    enc_.load_state_dict(enc_w)
    m_name = args.model_path.rsplit('/', 1)[1].split('.', 1)[0]

    #
    path = f'./benchmarks/{args.benchmark}'
    if not os.path.exists('./output/'):
        os.makedirs('./output/')
    out_file = f'output/{m_name}_{args.benchmark}_B{args.B}.csv'

    #
    gaps = ObjMeter('GAP')
    header = True
    for file in os.listdir(path):
        if not file.endswith('.fjs'):
            continue
        if file.startswith('.') or file.startswith('cached'):
            continue
        # Solve the instance
        instance = load_data(os.path.join(path, file), device=dev)
        res = validation(enc_, dec_, instance,
                         B=args.B, seed=args.seed)

        # Save results
        pd.DataFrame([res]).to_csv(out_file, index=False, mode='a+', sep=',', header=header)
        header = False
        gaps.update(instance, res['GAP'])

    #
    print(f"\t\t{args.benchmark} set: AVG Gap={gaps.avg:2.3f}")
    print(gaps)
