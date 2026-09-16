"""Single entry point for the frozen OX2R reproducibility workflow."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(script, args):
    cmd=[sys.executable, str(ROOT/'scripts'/script), *args]
    print('\n$ '+' '.join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)

def main():
    ap=argparse.ArgumentParser(description='Reproduce the OX2R benchmark and chemical-space analyses from frozen data.')
    ap.add_argument('--out',default='results/reproduction')
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--clusters',type=int,default=120)
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    run('benchmark_frozen.py',['--input','data/ox2r_curated_6415.csv','--out',str(out/'primary')])
    run('similarity_stratified.py',['--input','data/ox2r_curated_6415.csv','--out',str(out/'similarity'),'--bootstrap',str(args.bootstrap)])
    run('cluster_split.py',['--input','data/ox2r_curated_6415.csv','--out',str(out/'cluster'),'--clusters',str(args.clusters)])
    print('\nReproduction workflow completed.')
if __name__=='__main__': main()
