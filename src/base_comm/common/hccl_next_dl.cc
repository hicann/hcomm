/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_next_dl.h"
#ifdef __cplusplus
extern "C" {
#endif
int __HcclNextDlclose(void* handle) { return dlclose(handle); }

void* __HcclNextDlsym(void* handle, const char* funcName) { return dlsym(handle, funcName); }

void* __HcclNextDlopen(const char* libName, int mode) { return dlopen(libName, mode); }
weak_alias(__HcclNextDlopen, HcclNextDlopen);
weak_alias(__HcclNextDlclose, HcclNextDlclose);
weak_alias(__HcclNextDlsym, HcclNextDlsym);

#ifdef __cplusplus
} // extern "C"
#endif
