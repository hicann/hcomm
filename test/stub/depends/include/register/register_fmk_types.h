/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*
* The code snippet comes from Ascend project.
*
* Copyright 2019-2020 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
 */

#ifndef INC_EXTERNAL_REGISTER_REGISTER_FMK_TYPES_H_
#define INC_EXTERNAL_REGISTER_REGISTER_FMK_TYPES_H_

#include <string>

namespace domi {
///
/// @ingroup domi_omg
/// @brief  AI framework types
///
enum FrameworkType { CAFFE = 0, MINDSPORE = 1, TENSORFLOW = 3, ANDROID_NN = 4, ONNX = 5, FRAMEWORK_RESERVED = 6 };
} // namespace domi

#endif // INC_EXTERNAL_REGISTER_REGISTER_FMK_TYPES_H_
