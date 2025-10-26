import paddle
import paddle.distributed as dist
import time
from functools import partial

dist.fleet.init()
group = dist.new_group()

start_size = 2
end_size = 2 * 1024 * 1024 * 1024
iteration = 20

if dist.get_rank() == 0:
    op = partial(dist.send, dst=1)
else:
    op = dist.recv
print(op)
while start_size <= end_size:
    data = paddle.ones(shape=[start_size], dtype='float')
    # warmup
    for _ in range(iteration):
        op(data, group=group)
    paddle.device.synchronize()
    start_time = time.time()
    for _ in range(iteration):
        op(data, group=group, sync_op=False)
    paddle.device.synchronize()
    data._clear_data()
    end_time = time.time()
    tensor_size = start_size * 4
    tensor_size_in_GB = tensor_size / (1024 * 1024 * 1024)
    avg_time = (end_time - start_time) / iteration
    bandwitdh = tensor_size_in_GB / avg_time
    print(f'tensor size {tensor_size}B, avg_time: {avg_time*1000:.2f}ms, algorithm bandwidth {bandwitdh:.2f}GB/s')
    start_size *= 2
