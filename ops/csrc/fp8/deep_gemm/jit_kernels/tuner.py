# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The file has been adapted from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

import copy
import os
from typing import Any, Dict

import paddle

from ..jit import Runtime, build, cpp_format, generate


class JITTuner:
    def __init__(self) -> None:
        self.tuned = {}

    def compile_and_tune_group_gemm_masked(
        self,
        name: str,
        keys: Dict[str, Any],
        space: tuple,
        includes: tuple,
        arg_defs: tuple,
        template: str,
        args: tuple,
    ) -> Runtime:
        # NOTES: we always assume the space and template will not change
        # We also assume the GPU device will not be changed
        # NOTES: the function must have no accumulated side effects
        keys = {k: keys[k] for k in sorted(keys.keys())}
        signature = (name, f"{keys}")
        if signature in self.tuned:
            if os.getenv("DG_JIT_DEBUG", None):
                print(f"Using cached JIT kernel {name} with keys {keys}")
            return self.tuned[signature]

        if os.getenv("DG_JIT_DEBUG", None):
            print(f"Auto-tuning JIT kernel {name} with keys {keys}")

        assert signature not in self.tuned
        assert args is not None
        space = (dict(),) if len(space) == 0 else space

        kernels = []
        for tuned_keys in space:
            assert isinstance(tuned_keys, dict)
            full_keys = copy.deepcopy(keys)
            full_keys.update(tuned_keys)
            code = generate(includes, arg_defs, cpp_format(template, full_keys))

            # Illegal build must raise errors
            kernels.append((build(name, arg_defs, code), tuned_keys))

        best_runtime, best_time, best_keys = None, None, None
        for runtime, tuned_keys in kernels:
            if len(space) > 1:
                # Check kernel validity
                return_code = runtime(*args)
                if return_code != 0:
                    # Pass illegal kernels, e.g. insufficient shared memory capacity
                    if os.getenv("DG_JIT_DEBUG", None):
                        print(
                            f"Illegal JIT kernel {name} with keys {keys} and tuned keys {tuned_keys}: error code {return_code}"
                        )
                    continue

                # Measure performance with L2 flush and a large GEMM kernel before to reduce overhead between kernels
                start_event = paddle.device.cuda.Event(enable_timing=True)
                end_event = paddle.device.cuda.Event(enable_timing=True)
                paddle.empty(int(256e6 // 4), dtype=paddle.int32).zero_()
                paddle.randn((8192, 8192), dtype=paddle.float32, device="cuda") @ paddle.randn(
                    (8192, 8192), dtype=paddle.float32
                )
                start_event.record()
                for i in range(20):
                    assert runtime(*args) == 0
                end_event.record()
                end_event.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)
            else:
                elapsed_time = 0

            # Compare if better
            if best_time is None or elapsed_time < best_time:
                best_runtime, best_time, best_keys = runtime, elapsed_time, tuned_keys
            if os.getenv("DG_JIT_DEBUG", None):
                print(f"Tuned JIT kernel {name} with keys {keys} and tuned keys {tuned_keys} has time {elapsed_time}")
        assert best_runtime is not None, f"Failed to tune JIT kernel {name} with keys {keys}"

        # Cache the best runtime and return
        if os.getenv("DG_JIT_DEBUG", None) or os.getenv("DG_PRINT_AUTOTUNE", None):
            print(f"Best JIT kernel {name} with keys {keys} has tuned keys {best_keys} and time {best_time}")
        self.tuned[signature] = best_runtime
        return best_runtime

    def compile_and_tune(
        self,
        m,
        n,
        k,
        name: str,
        keys: Dict[str, Any],
        space: tuple,
        includes: tuple,
        arg_defs: tuple,
        template: str,
        # args: tuple,
    ) -> Runtime:
        # NOTES: we always assume the space and template will not change
        # We also assume the GPU device will not be changed
        # NOTES: the function must have no accumulated side effects
        signature = (name, m, k, n)
        if signature in self.tuned:
            return self.tuned[signature]
        # keys = {k: keys[k] for k in sorted(keys.keys())}
        # signature = (name, f"{keys}")
        # if signature in self.tuned:
        #     return self.tuned[signature]
        space = (dict(),) if len(space) == 0 else space

        kernels = []
        for tuned_keys in space:
            assert isinstance(tuned_keys, dict)
            full_keys = copy.deepcopy(keys)
            full_keys.update(tuned_keys)
            code = generate(includes, arg_defs, cpp_format(template, full_keys))

            # Illegal build must raise errors
            kernels.append((build(name, arg_defs, code), tuned_keys))

        best_runtime, best_time, best_keys = None, None, None
        for runtime, tuned_keys in kernels:
            elapsed_time = 0

            # Compare if better
            if best_time is None or elapsed_time < best_time:
                best_runtime, best_time, best_keys = runtime, elapsed_time, tuned_keys
            if os.getenv("DG_JIT_DEBUG", None):
                print(f"Tuned JIT kernel {name} with keys {keys} and tuned keys {tuned_keys} has time {elapsed_time}")
        assert best_runtime is not None, f"Failed to tune JIT kernel {name} with keys {keys}"

        # Cache the best runtime and return
        if os.getenv("DG_JIT_DEBUG", None) or os.getenv("DG_PRINT_AUTOTUNE", None):
            print(f"Best JIT kernel {name} with keys {keys} has tuned keys {best_keys} and time {best_time}")
        self.tuned[signature] = best_runtime
        return best_runtime


jit_tuner = JITTuner()
