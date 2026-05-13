#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

"""Patch for vllm.model_executor.kernels.linear to support OOT platforms.

Upstream vLLM's ``possible_kernels`` dictionaries only have entries for
CUDA, ROCM, CPU, and XPU platforms. The Ascend NPU platform registers as
``PlatformEnum.OOT``, which has no entries in these dicts, causing a
``KeyError`` when initializing FP8 linear kernels.

This patch adds ``PlatformEnum.OOT`` entries using Torch-based kernels
that are backend-agnostic.
"""

from vllm.model_executor.kernels.linear import (
    _POSSIBLE_FP8_BLOCK_KERNELS,
    _POSSIBLE_FP8_KERNELS,
    _POSSIBLE_INT8_KERNELS,
    _POSSIBLE_KERNELS,
    _POSSIBLE_MXFP8_KERNELS,
    _POSSIBLE_NVFP4_KERNELS,
    _POSSIBLE_WFP8A16_KERNELS,
)
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    ChannelWiseTorchFP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
    RowWiseTorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.triton import (
    TritonFp8BlockScaledMMKernel,
    TritonInt8ScaledMMLinearKernel,
)
from vllm.platforms import PlatformEnum

# ---------------------------------------------------------------------------
# Add OOT entries to the per-precision kernel dicts
# ---------------------------------------------------------------------------

_POSSIBLE_FP8_KERNELS.setdefault(
    PlatformEnum.OOT,
    [
        PerTensorTorchFP8ScaledMMLinearKernel,
        ChannelWiseTorchFP8ScaledMMLinearKernel,
        RowWiseTorchFP8ScaledMMLinearKernel,
    ],
)

_POSSIBLE_FP8_BLOCK_KERNELS.setdefault(
    PlatformEnum.OOT,
    [
        TritonFp8BlockScaledMMKernel,
        PerTensorTorchFP8ScaledMMLinearKernel,
        ChannelWiseTorchFP8ScaledMMLinearKernel,
        RowWiseTorchFP8ScaledMMLinearKernel,
    ],
)

_POSSIBLE_INT8_KERNELS.setdefault(
    PlatformEnum.OOT,
    [
        TritonInt8ScaledMMLinearKernel,
    ],
)

_POSSIBLE_WFP8A16_KERNELS.setdefault(
    PlatformEnum.OOT,
    [
        PerTensorTorchFP8ScaledMMLinearKernel,
        ChannelWiseTorchFP8ScaledMMLinearKernel,
        RowWiseTorchFP8ScaledMMLinearKernel,
    ],
)

_POSSIBLE_KERNELS.setdefault(PlatformEnum.OOT, [])

_POSSIBLE_MXFP8_KERNELS.setdefault(PlatformEnum.OOT, [])

_POSSIBLE_NVFP4_KERNELS.setdefault(PlatformEnum.OOT, [])
