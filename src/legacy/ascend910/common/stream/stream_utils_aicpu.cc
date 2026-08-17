/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unordered_map>
#include <functional>
#include "stream_utils.h"

HcclResult GetStreamCaptureInfo(rtStream_t stream, rtModel_t& rtModel, bool& isCapture) { return HCCL_SUCCESS; }

HcclResult AddStreamToModel(rtStream_t stream, rtModel_t& rtModel) { return HCCL_SUCCESS; }

HcclResult GetModelId(aclmdlRI& rtModel, u64& modelId) { return HCCL_SUCCESS; }
