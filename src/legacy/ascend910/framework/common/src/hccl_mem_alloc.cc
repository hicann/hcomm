/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_mem_alloc.h"
using namespace hccl;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

HcommResult HcommMemAlloc(void **ptr, size_t size)
{
    CHK_PTR_NULL(ptr);
    CHK_PRT_RET(size == 0, HCCL_ERROR("[HcommMemAlloc] size is zero"), HCCL_E_PARA);

    aclError ret = ACL_SUCCESS;
    int32_t deviceId;
    ret = aclrtGetDevice(&deviceId);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[HcommMemAlloc] GetDevice failed, ret[%d]", ret), HCCL_E_RUNTIME);

    aclrtPhysicalMemProp prop;
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.memAttr = ACL_HBM_MEM_HUGE;
    prop.location.id = deviceId;
    prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    prop.reserve = 0;

    size_t allocSize = size;
    size_t granularity = 0;
    ret = aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED, &granularity);
    CHK_PRT_RET(ret != ACL_SUCCESS || granularity == 0,
        HCCL_ERROR("[HcommMemAlloc] GetAllocationGranularity failed, granularity[%llu], ret[%d]", granularity, ret), HCCL_E_RUNTIME);
    ALIGN_SIZE(allocSize, granularity);
    HCCL_INFO("[HcommMemAlloc] deviceId[%d], granularity[%llu], size[%llu], allocSize[%llu].", deviceId, granularity, size, allocSize);

    ret = aclrtReserveMemAddress(ptr, allocSize, 0, nullptr, 1);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[HcommMemAlloc] ReserveMemAddress failed, "
        "virPtr[%p] size[%llu], ret[%d]", ptr, allocSize, ret), HCCL_E_RUNTIME);

    void *virPtr = *ptr;
    aclrtDrvMemHandle handle;
    ret = aclrtMallocPhysical(&handle, allocSize, &prop, 0);
    if(ret != ACL_SUCCESS) {
        HCCL_ERROR("[HcommMemAlloc] MallocPhysical failed, size[%llu], ret[%d]", allocSize, ret);
        aclrtReleaseMemAddress(virPtr);
        return HCCL_E_RUNTIME;
    }
    HCCL_INFO("[HcommMemAlloc]Start to MapMem virPtr[%p], handle[%p]", virPtr, handle);
    ret = aclrtMapMem(virPtr, allocSize, 0, handle, 0);
    if(ret != ACL_SUCCESS) {
        HCCL_ERROR("[HcommMemAlloc] MapMem virPtr[%p] size[%llu] handle[%p] failed, ret[%d]", virPtr, allocSize, handle, ret);
        aclrtFreePhysical(handle);
        aclrtReleaseMemAddress(virPtr);
        return HCCL_E_RUNTIME;
    }

    return HCCL_SUCCESS;
}

HcommResult HcommMemFree(void *ptr)
{
    if (ptr == nullptr) {
        HCCL_DEBUG("[HcommMemFree] virPtr is nullptr.");
        return HCCL_SUCCESS;
    }
    aclError ret = ACL_SUCCESS;
    aclrtDrvMemHandle handle;
    ret = aclrtMemRetainAllocationHandle(ptr, &handle);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[HcommMemFree] RetainAllocationHandle virPtr[%p] failed, ret[%d]", ptr, ret), HCCL_E_RUNTIME);
    HCCL_INFO("[HcommMemFree]Start to UnmapMem virPtr[%p], handle[%p]", ptr, handle);
    ret = aclrtUnmapMem(ptr);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[HcommMemFree] UnmapMem virPtr[%p] failed, ret[%d]", ptr, ret), HCCL_E_RUNTIME);
    ret = aclrtFreePhysical(handle);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[HcommMemFree] FreePhysical handle[%p] failed, ret[%d]", handle, ret), HCCL_E_RUNTIME);
    ret = aclrtReleaseMemAddress(ptr);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[HcommMemFree] ReleaseMemAddress virPtr[%p] failed, ret[%d]", ptr, ret), HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}
#ifdef __cplusplus
}
#endif // __cplusplus