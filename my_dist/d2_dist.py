import torch.distributed as dist
import torch
import pickle
import functools
import logging


# d2 的同步字典写法，要求内部的value shape完全相同，可以改进下
def d2_reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.

    Args:
        input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        average (bool): whether to do average or sum

    Returns:
        a dict with the same keys as input_dict, after reduction.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                dist.get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    # 计算每个 tensor 的所有长度和，例如 0 卡是80,1 卡是 7
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)

    # 最终返回是有多少个 world_size 就返回 list 长度是多少，list内部是否一样长度无所谓
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)
    ]
    # 提前统计下一共有多少数据量，后面接受时候要初始化对应长度的tensor
    dist.all_gather(size_list, local_size, group=group)

    # 此时可以得到 list 内部每个 tensor 的长度，此时就可以得到 [80,7]
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        # 如果 tensor 长度不是完全相同，则需要 padding
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


# 这个函数很通用，支持 cuda 和 cpu 对象
# 1. 将输入对象序列化，然后转为 tensor,相当于一维数组了
# 2. 同步统计每个 rank 的字节大小size_list，因为不同大小的 tensor 无法 all_gather
#所以还需要求最大长度，并且不够的 pad 0
# 3. 提前准备 tensor_list，然后对当前被pad后的 tensor 进行聚合
# 4. 统计后，移除 pad 部分，然后反序列化
def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    if dist.get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    # 任何可以序列化的对象都可以，全部序列化后返回
    # 已经被拉平了，相当于一维向量
    tensor = _serialize_to_tensor(data, group)

    # list 内部每个 tensor 的长度， 该 rank 下对应 tensor 的 pad tensor
    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # 有多少个 rank，长度即为 rank 长，一维向量
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list
    ]
    # 收集
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        # 去除额外 pad 的内容
        buffer = tensor.cpu().numpy().tobytes()[:size]
        # 反序列话，维度不变
        data_list.append(pickle.loads(buffer))

    return data_list

