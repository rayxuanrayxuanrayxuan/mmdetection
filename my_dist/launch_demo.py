import argparse
import os

import torch.distributed as dist
import torch.multiprocessing as mp

# 因为 init_process_group 的 init_method是 None，故 MASTER_ADDR 和 MASTER_PORT 必须要设置，内部会读取改环境变量
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=1)
    args = parser.parse_args()
    return args


def example(rank, world_size):
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    print(f'rank={dist.get_rank()}')


def main():
    args = parse_args()
    world_size = args.num_gpus
    mp.spawn(example, args=(world_size, ), nprocs=world_size, join=True)


# python my_dist/launch_demo.py --num_gpus 8
if __name__ == '__main__':
    main()
