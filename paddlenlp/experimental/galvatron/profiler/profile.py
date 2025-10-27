import os
import paddle
try:
    import paddle.base.core as core
    def sync():
        return paddle.device.synchronize()
except ImportError:
    import paddle.fluid.core as core
    def sync():
        return paddle.device.cuda.synchronize()

import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from paddle.distributed import stream
import sys
import time
import json


default_dtype = paddle.float32


def get_nranks(group):
    return group.nranks if group is not None else paddle.distributed.get_world_size()


def get_root(group):
    return min(group.ranks) if group is not None else 0


def get_sizeof(dtype):
    return core.size_of_dtype(dtype)


def is_rank_0():
    return paddle.distributed.get_rank() == 0


def round_nbytes(nbytes, group, dtype):
    nranks = get_nranks(group)
    sizeof = get_sizeof(dtype)
    align = nranks * sizeof
    return (nbytes + align - 1) // align * align, nranks, sizeof


def get_full_tensor(nbytes, group, dtype):
    nbytes, _, sizeof = round_nbytes(nbytes, group, dtype)
    return paddle.zeros([nbytes // sizeof], dtype=dtype), nbytes


def get_partial_tensor(nbytes, group, dtype):
    nbytes, nranks, sizeof = round_nbytes(nbytes, group, dtype)
    return paddle.zeros([nbytes // (nranks * sizeof)], dtype=dtype)


def run_test(func, nbytes, coeff, warmup_num=20, test_num=100):
    paddle.distributed.barrier()
    for _ in range(warmup_num):
        func()
    paddle.distributed.barrier()
    start_t = time.time_ns()
    for _ in range(test_num):
        func()
    sync()
    end_t = time.time_ns()
    cost_t = (end_t - start_t) / test_num
    bw = nbytes / cost_t * coeff
    ret = []
    dist.all_gather_object(ret, bw)
    return min(ret), max(ret)


def test_allreduce(nbytes, group, dtype):
    x, nbytes = get_full_tensor(nbytes, group, dtype)
    nranks = get_nranks(group)
    coeff = 2 * (nranks - 1) / nranks
    func = lambda: stream.all_reduce(x, group=group, use_calc_stream=True, sync_op=True)
    return run_test(func, nbytes, coeff)


def test_reduce(nbytes, group, dtype):
    x, nbytes = get_full_tensor(nbytes, group, dtype)
    root = get_root(group)
    coeff = 1.0
    func = lambda: stream.reduce(x, group=group, dst=root, use_calc_stream=True, sync_op=True)
    return run_test(func, nbytes, coeff)


def test_broadcast(nbytes, group, dtype):
    x, nbytes = get_full_tensor(nbytes, group, dtype)
    root = get_root(group)
    coeff = 1.0
    func = lambda: stream.broadcast(x, group=group, src=root, use_calc_stream=True, sync_op=True)
    return run_test(func, nbytes, coeff)


def test_reducescatter(nbytes, group, dtype):
    x, nbytes = get_full_tensor(nbytes, group, dtype)
    y = get_partial_tensor(nbytes, group, dtype)
    nranks = get_nranks(group)
    coeff = (nranks - 1) / nranks
    func = lambda: stream.reduce_scatter(y, x, group=group, use_calc_stream=True, sync_op=True)
    return run_test(func, nbytes, coeff)


def test_allgather(nbytes, group, dtype):
    x, nbytes = get_full_tensor(nbytes, group, dtype)
    y = get_partial_tensor(nbytes, group, dtype)
    nranks = get_nranks(group)
    coeff = (nranks - 1) / nranks
    func = lambda: stream.all_gather(x, y, group=group, use_calc_stream=True, sync_op=True)
    return run_test(func, nbytes, coeff)


def test_alltoall(nbytes, group, dtype):
    x, nbytes = get_full_tensor(nbytes, group, dtype)
    y, _ = get_full_tensor(nbytes, group, dtype)
    nranks = get_nranks(group)
    coeff = (nranks - 1) / nranks
    func = lambda: stream.alltoall_single(x, y, group=group, use_calc_stream=True, sync_op=True)
    return run_test(func, nbytes, coeff)


def run_all(dtype=default_dtype):
    groups = []
    hcg = fleet.get_hybrid_communicate_group()
    dp_group = hcg.get_data_parallel_group()
    mp_group = hcg.get_model_parallel_group()
    if dp_group is not None and dp_group.nranks > 1:
        groups.append(("dp", dp_group))
    dp_degree = get_nranks(dp_group)

    if mp_group is not None and mp_group.nranks > 1:
        groups.append(("mp", mp_group))
    mp_degree = get_nranks(mp_group)

    nbytes_in_mb = [1024, 2048, 4096, 8192]
    ret = {}
    for group_name, group in groups:
        ret[group_name] = []
        for tmp in nbytes_in_mb:
            nbytes, _, _ = round_nbytes(int(tmp * 1024 * 1024), group, dtype)
            ret[group_name].append({
                "size": nbytes,
                "size_mb": nbytes / (1024 * 1024),
                "allreduce": test_allreduce(nbytes, group, dtype),
                # "reduce": test_reduce(nbytes, group, dtype),
                # "broadcast": test_broadcast(nbytes, group, dtype),
                # "reducescatter": test_reducescatter(nbytes, group, dtype),
                # "allgather": test_allgather(nbytes, group, dtype),
                # "alltoall": test_alltoall(nbytes, group, dtype),
            })
            if is_rank_0():
                print(f"Done for {tmp} MB for {group_name}")
    return ret, f"bandwidth_tp{mp_degree}_dp{dp_degree}.json"


mp_degree = int(sys.argv[1])
strategy = fleet.DistributedStrategy()
strategy.hybrid_configs = {
    "mp_degree": mp_degree,
}
fleet.init(is_collective=True, strategy=strategy)

ret, path = run_all()
if is_rank_0():
    with open(path, "w") as f:
        json.dump(ret, f, indent=2)