/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
<<<<<<<< HEAD:csrc/moe/dequant_swiglu_quant/op_host/swi_glu_grad_regbase_tiling.h
 * \file swi_glu_grad_regbase_tiling.h
 * \brief
 */

struct GluBaseTilingData {
    int64_t rowTotal;
    int64_t colTotal;
    int64_t rowBase;
    int64_t colBase;
    int64_t rowTail;
    int64_t colTail;
    int64_t ubSize;
    int64_t rowTileNum;
    int64_t colTileNum;
    int64_t usedCoreNum;
};
========
 * \file tiling_util.cpp
 * \brief
 */

#include "../tiling_base/tiling_util.h"
namespace Ops {
namespace Transformer {
namespace OpTiling {
static const gert::Shape g_vec_1_shape = {1};

const gert::Shape &EnsureNotScalar(const gert::Shape &inShape)
{
    if (inShape.IsScalar()) {
        return g_vec_1_shape;
    }
    return inShape;
}
} // namespace OpTiling
} // namespace Transformer
} // namespace Ops
>>>>>>>> 05b862fc (feat(csrc): support deepseek v4):csrc/moe/causal_conv1d/op_host/tiling_util.cpp
