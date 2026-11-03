"""Run a small assigned list serially; GPU ownership belongs to shared launch."""
import argparse,subprocess,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('jobs',nargs='+',help='arm:seed')
    args=p.parse_args()
    for job in args.jobs:
        arm,seed=job.split(':')
        subprocess.run([sys.executable,str(Path(__file__).with_name('run.py')),
                        '--arm',arm,'--seed',seed],check=True)


if __name__=='__main__':main()
