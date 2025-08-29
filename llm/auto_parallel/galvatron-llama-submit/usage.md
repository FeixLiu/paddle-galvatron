# `Galvatron`使用文档

## 环境准备
### 基础环境

1. 创建虚拟环境：
```bash
virtualenv -p python3.10 xxx_venv
source xxx_venv/bin/activate
```
2. 安装 paddle 3.1：
```bash
python -m pip install paddlepaddle-gpu==3.1.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
```
3. 热更新Paddle，使用 `hot_fix` 文件夹下的文件替换原文件：
```bash
cp -f hot_fix/* xxx_venv/lib/python3.10/site-packages/paddle/distributed/auto_parallel/
```
4. 安装PaddleNLP依赖： 
```bash
cd PaddleNLP-galvatron
python -m pip install -r requirements.txt
python -m pip install transformers
python -m pip install nvtx
python -m pip uninstall aistudio_sdk
python -m pip install aistudio_sdk==0.2.6
```
5. 安装PaddleNLP-kernel：
```bash
cd PaddleNLP-galvatron/ops/csrc
rm -rf build dist *.egg-info
python setup.py build
cd ..
python setup.py bdist_wheel
pip install dist/*.whl --force-reinstall
```
6. 安装 `Galvatron` ：
```bash
cd PaddleNLP-galvatron
python -m pip install -e .
cd ./paddlenlp/experimental/galvatron/search_engine
python -m pip install pybind11
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) dp_core.cpp -o galvatron_dp_core$(/bin/python3.10-config --extension-suffix)
```
编译结束后，会在 `search_engine` 目录下生成 `galvatron_dp_core.cpython-39-x86_64-linux-gnu.so` 文件。

## 使用`Galvatron`进行性能分析

使用 `Galvatron `的第一步是对硬件环境和模型计算时间进行性能分析。`Galvatron` 会自动将分析结果保存到配置文件中。

### 分析硬件环境性能

*需要多机mpi运行环境*

1. 修改 `profile_hardware.sh` 中的 `START_RANK` 与 `END_RANK` 为集群中的目标机器。同时修改 `interpreter` 指向虚拟环境中的 `python` 。
2. 生成多机profile脚本：
```bash
mpirun bash profile_hardware.sh
```
3. 随后依次运行以下生成的脚本，将在 `configs` 下生成对应的 `json` 文件。
```bash
mpirun rm -rf configs/*
mpirun bash scripts/profile_allreduce.sh
mpirun bash scripts/profile_p2p.sh
mpirun bash scripts/profile_allreduce_sp.sh
mpirun bash scripts/profile_all2all.sh
mpirun bash scripts/profile_overlap.sh
```

#### 注意
1. 如果遇到程序异常退出，直接重新执行对应脚本即可。无需全部重新执行。
2. 上述5个脚本总耗时大概需要1.5小时。

### 分析模型性能

*仅需要单机环境*

1. 修改 `scripts/profile_memory.sh` 与 `scripts/profile_computation.sh`
- 更新 `source` 路径，指向配置好的虚拟环境。
- 更新 `MODEL_ARGS` 中的模型参数，参数与模型大小对照表如下：

|                      | 30B   | 100B   |
|----------------------|-------|--------|
| hidden_size          | 5120  | 8192   |
| intermediate_size    | 25600 | 49152  |
| seq_length           | 32768 | 131072 |
| num_hidden_layers    | 72    | 74     |
| num_attention_head   | 64    | 64     |
| num_key_value_heads  | 8     | 8      |
| vocab_size           | 32000 | 32000  |

同时由于对于 `seq_length` 与 `num_hidden_layers` 的配置在 `MODEL_PROFILER_ARGS` 字段中，这两个字段可以不按表格配置。

- 更新 `scripts/profile_computation.sh` 中的`MODEL_PROFILER_ARGS` 参数：
  - `profile_mode` 配置为 `sequence` 。
  - `profile_max_seq_length` 配置为 `16384` 。*不要超过这个长度，否则H卡会OOM*
  - `profile_min_seq_length` 配置为 `4096` 。
  - `profile_seq_length_step` 配置为 `4096` 。
  - `num_layertype` 配置为 `1` 。

- 更新 `scripts/profile_memory.sh` 中的`MODEL_PROFILER_ARGS` 参数：
  - `profile_mode` 配置为 `static` 。
  - `profile_fixed_seq_length_list` 为所需要的序列长度.
  - 若 `profile_fixed_seq_length_list` 序列长度能够正常运行全部case，则在下一步 `search_dist.sh` 中的 `memory_profile_mode` 设置为` static` 。
  - 若 `profile_fixed_seq_length_list` 序列长度会导致OOM，则将其序列长度逐渐除2减小，直到能够正常运行全部case。此时，将下一步 `search_dist.sh` 中的 `memory_profile_mode` 设置为 `sequence` 。
  - 举例：若在 `profile_fixed_seq_length_list` 为32768的 `static` 模式下，部分case会OOM，
    则将 `profile_fixed_seq_length_list` 设置为16384，此时若所有case均能正常运行，则 `profile_memory` 成功结束，
    随后在 `search_dist.sh中` ，设置 `memory_profile_mode` 为 `sequence` 即可。


2. 执行profile脚本，进行显存、计算速度profile
```bash
bash scripts/profile_memory.sh
bash scripts/profile_computation.sh
```

#### 注意
1. 搜索过程请保证显存独占，如果由于其他用户使用显存导致异常OOM，需要重新执行profile脚本，否则会导致策略搜索失败。
2. 由于需要对两种 `hidden_size` 不同的模型进行profile，所以上述脚本均需要执行两次。
   对于相同 `hidden_size` 的不同规模模型，profile结果可以复用。
   **请在执行完策略搜索，得到当前模型最优配置之后再重新profile新的模型性能。**
3. profile无需对目标seq len进行profile。搜索引擎可以根据短序列的结果进行拟合。
   例如：党最终目标是跑128K的时候，可以在step是4k的情况下去profile 4k-16k长度的开销。
   search engine会根据4k-16k的结果，去预估128k的开销。
4. 模型性能分析大概持续1小时。

### 策略搜索

*仅需要单机环境且无需显存占用*

给定集群和内存预算，`Galvatron` 搜索引擎将自动生成最优并行策略。优化后的并行策略将以 `JSON `文件形式保存在 `configs` 中用于训练。

1. 更新 `scripts/search_dist.sh` 脚本
- 更新 `source` 路径，指向配置好的虚拟环境。
- 修改 `ProfileDataParserArgs` 字段：
  - `time_profile_mode` 设置为 `sequence` 。
  - `profile_mode` 根据实际情况配置为 `static` 或 `sequence` 。 详见*分析模型性能*中对 `scripts/profile_memory.sh` 的描述。
  - `num_layertype` 设置为 `1` 。
  - `hidden_size_list` 、 `layer_num_list` 、 `seqlen_list` 修改为实际的配置，注意，一次只能写一个值。
  - 其余所有config的path修改为实际的path，也就是前两步生成的profile文件。
- 修改 `SearchEngineArgs` 字段：
  - `max_tp_size` 与 `max_pp_size` 分别表示期望的 `tp_size` 与 `pp_size` 的上限。
  - `memory_upper_limit` 表示单卡的显存上限，单位为 `GB` 。

2. 运行`bash scripts/search_dist.sh`，将在`configs`中生成搜索出的最优策略。

#### 注意
1. 如果在策略搜索过程中报错，大概率是模型性能profile的中途有异常退出的情况，请重新进行模型性能profile。
2. 确保 `ProfileDataParserArgs` 中的 `profile_gpu_num` 与 `SearchEngineArgs` 中的 `world_size` 保持对齐。
2. 确保 `ProfileDataParserArgs` 中的 `layernum_list` 与 `SearchEngineArgs` 中的 `layernum` 保持对齐。
3. 策略搜索大概持续1小时。

## 根据搜索结果，训练模型

*需要多机mpi环境*

1. 将搜索结果进行全局同步
```bash
tar -cf configs.tar configs
sh cp.sh configs.tar
tar -xf configs.tar
```
2. 修改 `scripts/train_qwen_fine_grained.sh` 中的 `START_RANK` 与 `END_RANK` 为集群中的目标机器。同时修改 `source` 源为自己的虚拟环境。
3. 修改 `scripts/train_qwen_fine_grained.sh` 中的 `MODEL_ARGS` 为目标模型大小。
4. 修改 `scripts/train_qwen_fine_grained.sh` 中的 `DATA_ARGS` 的 `max_seq_length` 字段为目标序列长度。
5. 根据 `configs/fine_grained_config.json` 的搜错结果修改 `scripts/train_qwen_fine_grained.sh`：
- 将 `per_device_train_batch_size` 、 `gradient_accumulation_steps` 、 `recompute` 、 
`sharding_parallel_degree` 、 `tensor_parallel_degree` 、 `pipeline_parallel_degree` 根据 `configs/fine_grained_config.json` 的结果进行修改。
  - 对于 `sharding_parallel_degree` 、 `tensor_parallel_degree` 、 `pipeline_parallel_degree` ，
    搜索出来的结果 `pp_size` 对应 `pipeline_parallel_degree` ，
    `dp_size_list` 字符串的首字符数字对应 `sharding_parallel_degree` ，
    `tp_size_list` 字符串的首字符数字对应 `tensor_parallel_degree` 。
  - 对于 `recompute` ，若搜索出来的结果中， `recompute_list` 字符串中存在字符1，则将 `recompute` 设置为 `true` 。
  - 对于 `gradient_accumulation_steps` 、 `per_device_train_batch_size` ，搜索出来的结果中
    `gradient_accumulation_steps` 对于 `gradient_accumulation_steps` ，
    `per_device_train_batch_size` 则由公式 `global_batch_size/gradient_accumulation_steps/dp_size` 计算而得。
6. 根据最优策略运行
```bash
cd scripts
sh cp.sh train_qwen_fine_grained.sh
cd ..
mpirun bash scripts/train_qwen_fine_grained.sh
```
