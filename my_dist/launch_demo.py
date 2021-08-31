import argparse
import os
import random

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from my_dist.d2_dist import all_gather

# 因为 init_process_group 的 init_method是 None，故 MASTER_ADDR 和 MASTER_PORT 必须要设置，内部会读取改环境变量
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--with_cuda', type=bool, default=True)
    args = parser.parse_args()
    return args


# demo1
def print_rank():
    print(f'rank={dist.get_rank()}')


# demo2
def mmdet_sync_dict(backend, with_cuda=True):
    from mmdet.core.utils.dist_utils import all_reduce_dict
    rand_dict = {'a': torch.ones((2, 2)),
                 'b': torch.ones((1, 3))}
    if with_cuda:
        for key, value in rand_dict.items():
            rand_dict[key] = value.cuda()

    # 1 内部的 all_reduce 和 broadcast, 在 cpu 模式下仅仅 gloo 支持，在 gpu 模式下 gloo + nccl 都支持
    # 2 内部为何要先同步 key，原因是字典是无序的，不同卡上面顺序可能不一样，通过同步，可以保证大家都一样，后面代码就不会同步错
    # 3 内部的 group = _get_global_gloo_group() 明显是多余的
    # 4 内部要记录shape和总数numel，是为了所有 tensor flatten 后能够还原，这个写法非常通用
    # 5 因为内部会 cat，所以要求所有字典内对象都是同一个类型，否则不合理
    # 6 如果想省掉字典同步这个过程，可以参考 d2 写法先对 key 进行排序，然后同步
    reduce_out = all_reduce_dict(rand_dict)

    if dist.get_rank() == 0:
        print(reduce_out, flush=True)


# demo3
# 模拟分布式多卡 test，然后最终合并所有结果
# mmdet 中实现的 collect_results_gpu 和 d2 all_gather 功能相同
# 但是 mmdet 在 cpu 场景下，又写了一套 collect_results_cpu，感觉没有必要吧，
# 其是通过在不同 rank 写对应的临时文件实现
def d2_all_gather(backend, with_cuda=True):
    random_len = random.randint(1, 10)
    print(f'rank={dist.get_rank()},len={random_len}')
    rand_dict = dict(a=torch.randn((random.randint(1, 10), 2)))

    if with_cuda:
        for key, value in rand_dict.items():
            rand_dict[key] = value.cuda()

    data = [rand_dict] * random_len

    # all_gather op 中 gloo 仅仅支持 cpu, nccl 仅仅支持 gpu
    # 如果不指定则默认采用 gloo，但是此时 gpu 就会报错，所以需要指定后端
    # dist.group.WORLD 表示采用目前已经开启的默认后端，不重新创建
    all_data = all_gather(data, dist.group.WORLD)

    if dist.get_rank() == 0:
        print(f'total len={len(all_data)}, {all_data}', flush=True)


# demo3
def mmcv_sync_bn():
    from collections import OrderedDict
    from torch._utils import (_flatten_dense_tensors, _take_tensors,
                              _unflatten_dense_tensors)

    model = None

    # 1 mm 系列中同步 bn 的写法
    # tensors = list[Tensor]

    import torch.nn.parallel.comm #这个文件下面有很多高效内置的同步代码实现
    # comm.broadcast_coalesced 好像有可直接调用的含义(ddp内部自动调用)
    def _allreduce_coalesced(tensors, world_size, bucket_size_mb=-1):
        if bucket_size_mb > 0:
            # 如果要同步的数据量过大，可以考虑采用迭代器切分进行同步
            # _take_tensors 实现了该功能
            # 默认是类型全部相同
            bucket_size_bytes = bucket_size_mb * 1024 * 1024
            buckets = _take_tensors(tensors, bucket_size_bytes)
        else:
            # 否则对不同类型的数据进行分组，因为不同类型需要分开来进行同步
            buckets = OrderedDict()
            for tensor in tensors:
                tp = tensor.type()
                if tp not in buckets:
                    buckets[tp] = []
                buckets[tp].append(tensor)
            buckets = buckets.values()

        # 对不同组进行单独进行同步
        for bucket in buckets:
            flat_tensors = _flatten_dense_tensors(bucket)
            dist.all_reduce(flat_tensors)
            flat_tensors.div_(world_size)
            for tensor, synced in zip(
                    bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
                # 原地 copy
                tensor.copy_(synced)

    def allreduce_params(params, coalesce=True, bucket_size_mb=-1):
        """Allreduce parameters.

        Args:
            params (list[torch.Parameters]): List of parameters or buffers of a
                model.
            coalesce (bool, optional): Whether allreduce parameters as a whole.
                Defaults to True.
            bucket_size_mb (int, optional): Size of bucket, the unit is MB.
                Defaults to -1.
        """
        world_size = dist.get_world_size()
        if world_size == 1:
            return
        params = [param.data for param in params]
        if coalesce:
            # 合并到一起然后进行同步
            _allreduce_coalesced(params, world_size, bucket_size_mb)
        else:
            # 不合并，每个 tensor 单独同步，非常慢
            for tensor in params:
                dist.all_reduce(tensor.div_(world_size))

    allreduce_params(model.buffers())

    # 2 yolox 的同步写法如下：
    from mmdet.core.utils.dist_utils import all_reduce_dict

    def get_norm_states(module):
        async_norm_states = OrderedDict()
        for name, child in module.named_modules():
            if isinstance(child, nn.modules.batchnorm._NormBase):
                for k, v in child.state_dict().items():
                    async_norm_states['.'.join([name, k])] = v
        return async_norm_states

    norm_states = get_norm_states(model)
    # 前面分析的同步字典函数
    norm_states = all_reduce_dict(norm_states, op='mean')
    # 重新load
    model.load_state_dict(norm_states, strict=False)


def example(rank, world_size, with_cuda):
    if with_cuda:
        # 必不可少，否则会出错
        torch.cuda.set_device(rank)
        backend = 'nccl'
    else:
        backend = 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # print_rank()
    # mmdet_sync_dict(backend, with_cuda)
    d2_all_gather(backend, with_cuda)


def main():
    args = parse_args()
    world_size = args.num_gpus
    with_cuda = args.with_cuda
    mp.spawn(example, args=(world_size, with_cuda), nprocs=world_size, join=True)


# python my_dist/launch_demo.py --num_gpus 2
if __name__ == '__main__':
    main()
