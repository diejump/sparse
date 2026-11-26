"""Research entry point with exactly three method-level numeric controls."""
import argparse
from dataclasses import dataclass
import torch
from envelope_experiment import run

@dataclass(frozen=True)
class RangeReferenceParameters:
    gamma: float = 2.0
    rho: float = 0.2
    alpha: float = 0.005

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',required=True);p.add_argument('--output',required=True)
    p.add_argument('--gamma',type=float,default=2.);p.add_argument('--rho',type=float,default=.2)
    p.add_argument('--alpha',type=float,default=.005)
    p.add_argument('--architecture',choices=['GCN','GAT','GraphSage'],default='GCN')
    p.add_argument('--seed',type=int,default=41);p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--device',default='cpu')
    a=p.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    run(a.snapshot,a.output,a.architecture,a.seed,a.device,a.gamma,a.rho,a.alpha,
        repair='median',score='range',variant='full',epochs=a.epochs,structure=True)

if __name__=='__main__':main()
