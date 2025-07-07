import argparse
import torch
import random
import pandas as pd
import os
from sampling import sampling, greedy
from inout import load_data_v2
from time import time
from utils import ObjMeter

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

@torch.no_grad()
def validation(encoder: torch.nn.Module,
               decoder: torch.nn.Module,
               ins,
               beta: int = 32,
               seed: int = None):
    """

    Args:
        encoder: Encoder.
        decoder: Decoder.
        ins: JSP instance.
        beta: Number of solution to generate for each instance.
        seed: Random seed.
    """
    if seed is not None:
        torch.manual_seed(seed)
    encoder.eval()
    decoder.eval()

    st = time()
    s, mss, entropies, log_p = sampling(ins, encoder, decoder, 
                                             bs=beta, device=dev, use_greedy=args.use_greedy)
    exe_time = time() - st

    #
    _gaps = (mss / ins['makespan'] - 1) * 100
    min_gap = _gaps.min().item()
    max_gap = _gaps.max().item()
    print(f'\t- {ins["name"]} = {min_gap:.3f}%~{max_gap:.3f}%')
    results = {'NAME': ins['name'],
               'UB': ins['makespan'],
               'MS': mss.min().item(),
               'MS-AVG': mss.mean().item(),
               'MS-STD': mss.std().item(),
               'GAP': min_gap,
               'GAP-MAX': max_gap,
               'GAP-AVG': _gaps.mean().item(),
               'GAP-STD': _gaps.std().item(),
               'TIME': f"{exe_time:2.3f}",
               }
    return results


#
parser = argparse.ArgumentParser(description='Test Pointer Net')
parser.add_argument("-model_path", type=str, required=False,
                    default="./checkpoints/CamLstmNet-K16-B256.pt",
                    help="Path to the model.")
parser.add_argument("-benchmark", type=str, required=False,
                    default='LA', help="Name of the benchmark for testing.")
parser.add_argument("-B", type=int, default=512, required=False,
                    help="Number of sampled solutions for each instance.")
parser.add_argument("-seed", type=int, default=None,
                    required=False, help="Random seed.")
parser.add_argument("-greedy", type=int, default=1,
                    required=False, help="Use greedy policy when sampling.")
parser.add_argument("-device", type=int, default=0, required=False,
                    help="The No. GPU card.")
args = parser.parse_args()
args.use_greedy = args.greedy != 0

if args.seed is None:
    args.seed = random.randint(0, 10000)

print(args)

if __name__ == '__main__':
    from PointerNet import CAMEncoder3
    # Testing device
    dev = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    print(f"Using {dev}...")

    # Load the model
    print(f"Loading {args.model_path}")
    enc_w, dec_ = torch.load(args.model_path, map_location=dev)
    enc_ = CAMEncoder3(15).to(dev)   # Load weights to avoid bug with new PyG
    enc_.load_state_dict(enc_w)
    m_name = args.model_path.rsplit('/', 1)[1].split('.', 1)[0]

    #
    path = f'./benchmarks/{args.benchmark}'
    if not os.path.exists('./output/'):
        os.makedirs('./output/')
    out_file = f'output/{m_name}_{args.benchmark}_B{args.B}_{"greedy" if args.use_greedy else "sample"}.csv'

    #
    gaps = ObjMeter()
    header = True
    for file in sorted(os.listdir(path)):
        if file.startswith('.') or file.startswith('cached'):
            continue
        # Solve the instance
        instance = load_data_v2(os.path.join(path, file), device=dev)
        res = validation(enc_, dec_, instance,
                         beta=args.B, seed=args.seed)

        # Save results
        pd.DataFrame([res]).to_csv(out_file, index=False, mode='a+', sep=',', header=header)
        header = False
        gaps.update(instance, res['GAP'])

    #
    print(f"\t\t{args.benchmark} set: AVG Gap={gaps.avg:2.3f}")
    print(gaps)
