/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aicpu_ts_sync_data_c_adpt.h"

#include <chrono>
#include <thread>

#include "log.h"
#include "thread.h"
#include "aicpu_ts_thread.h"

constexpr uint32_t SYNC_WAIT_TIMEOUT_SECONDS = 205;
constexpr size_t MSG_TAG_SIZE_BYTE = 256;
constexpr uint32_t TIMEOUT_SIZE_BYTE = 4; // timeout字段长度为4字节，表示超时时间，单位为秒
constexpr uint32_t CTRL_HDR_DATA_SIZE_LEN  = 8; // size_t 在不同平台上长度不同，取最大值

// Msg 数据格式如下（单位：字节）：
// +----------+--------------+-----------+-------------+---------------+-----------------+
// | flag [1] | msgTag [256] | msgId [4] | timeout [4] | data size [8] | data [sizeByte] |
// +----------+--------------+-----------+-------------+---------------+-----------------+
// ^
// handle

static HcclResult WaitFlagReady(uint8_t *srcFlagPtr, uint8_t *srcTimeoutPtr)
{
    HCCL_INFO("[%s] Polling flag START.", __func__);
    const auto timeStart = std::chrono::steady_clock::now();
    auto timeoutSec = std::chrono::seconds(SYNC_WAIT_TIMEOUT_SECONDS);
    uint32_t timeoutValue{SYNC_WAIT_TIMEOUT_SECONDS};
    errno_t ret = memcpy_s(&timeoutValue, sizeof(timeoutValue), srcTimeoutPtr, sizeof(timeoutValue));
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[%s][memcpy_s] Reading timeout ERROR[%d].", __func__, ret), HCCL_E_INTERNAL);
    HCCL_INFO("[%s] Reading timeout from shared mem SUCCESS. timeout = %u seconds.", __func__, timeoutValue);
    if (timeoutValue > 0) {
        HCCL_INFO("[%s] Using timeout from shared mem. timeout = %u seconds.", __func__, timeoutValue);
        timeoutSec = std::chrono::seconds(timeoutValue);
    } else {
        HCCL_INFO("[%s] Using default timeout. timeout = %u seconds.", __func__, SYNC_WAIT_TIMEOUT_SECONDS);
    }
    uint8_t flagReadValue{0};
    while (true) {
        ret = memcpy_s(&flagReadValue, sizeof(flagReadValue), srcFlagPtr, sizeof(flagReadValue));
        CHK_PRT_RET(ret != EOK, HCCL_ERROR("[%s][memcpy_s] Polling flag ERROR[%d].", __func__, ret), HCCL_E_INTERNAL);
        if (flagReadValue == 1) {
            break;
        }
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - timeStart);
        if (elapsed > timeoutSec) {
            HCCL_ERROR("[%s] Polling flag TIMEOUT.", __func__);
            return HCCL_E_TIMEOUT;
        }
    }
    HCCL_INFO("[%s] Polling flag SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
int32_t HcommSendRequest(MsgHandle handle, const char *msgTag, const void *src, size_t sizeByte, uint32_t *msgId)
{
    uint8_t *const dstOnDevShmem = reinterpret_cast<uint8_t *>(handle);
    CHK_PTR_NULL(dstOnDevShmem);
    CHK_PTR_NULL(msgTag);
    CHK_PTR_NULL(src);

    HCCL_INFO("[%s] START. msgHandle[0x%llx], msgTag[%s], src[0x%llx], sizeByte[%zu].", __func__, handle, msgTag, src, sizeByte);

    static uint32_t s_msgId{0};
    const uint8_t flagWriteValue{1};
    uint8_t *const dstFlagPtr = dstOnDevShmem;
    uint8_t *const dstMsgTagPtr = dstFlagPtr + sizeof(flagWriteValue);
    uint8_t *const dstMsgIdPtr = dstMsgTagPtr + MSG_TAG_SIZE_BYTE;
    uint8_t *const dstDataPtr = dstMsgIdPtr + sizeof(s_msgId) + TIMEOUT_SIZE_BYTE;
    errno_t ret = EOK;

    HCCL_INFO("[%s] Writing %zu bytes data from src to shared mem START.", __func__, sizeByte);
    ret = memcpy_s(dstDataPtr, sizeof(sizeByte), &sizeByte, sizeof(sizeByte));
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[%s][memcpy_s] Writing data ERROR[%d].", __func__, ret), HCCL_E_INTERNAL);
    ret = memcpy_s(dstDataPtr + CTRL_HDR_DATA_SIZE_LEN, sizeByte, src, sizeByte);
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[%s][memcpy_s] Writing data ERROR[%d].", __func__, ret), HCCL_E_INTERNAL);
    HCCL_INFO("[%s] Writing %zu bytes data from src to shared mem SUCCESS.", __func__, sizeByte);

    HCCL_INFO("[%s] Writing %zu bytes msgId to shared mem START. msgId = %u.", __func__, sizeof(s_msgId), s_msgId);
    ret = memcpy_s(dstMsgIdPtr, sizeof(s_msgId), &s_msgId, sizeof(s_msgId));
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[%s][memcpy_s] Writing msgId ERROR[%d].", __func__, ret), HCCL_E_INTERNAL);
    HCCL_INFO("[%s] Writing %zu bytes msgId to shared mem SUCCESS. msgId = %u.", __func__, sizeof(s_msgId), s_msgId);

    HCCL_INFO("[%s] Writing %zu bytes msgTag to shared mem START.", __func__, MSG_TAG_SIZE_BYTE);
    ret = memcpy_s(dstMsgTagPtr, MSG_TAG_SIZE_BYTE, msgTag, MSG_TAG_SIZE_BYTE);
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[%s][memcpy_s] Writing msgTag ERROR[%d].", __func__, ret), HCCL_E_INTERNAL);
    HCCL_INFO("[%s] Writing %zu bytes msgTag to shared mem SUCCESS.", __func__, MSG_TAG_SIZE_BYTE);

    asm volatile("dmb sy" ::: "memory"); // 确保之前的内存写入对其他线程可见
    
    HCCL_INFO("[%s] Setting flag = 1 on shared mem START.", __func__);
    ret = memcpy_s(dstFlagPtr, sizeof(flagWriteValue), &flagWriteValue, sizeof(flagWriteValue));
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[%s][memcpy_s] Setting flag ERROR[%d].", __func__, ret), HCCL_E_INTERNAL);
    HCCL_INFO("[%s] Setting flag = 1 on shared mem SUCCESS.", __func__);

    *msgId = s_msgId;
    ++s_msgId;  // Auto goes back to 0 once it reaches UINT32_MAX

    HCCL_INFO("[%s] SUCCESS. msgId[%u].", __func__, *msgId);
    return HCCL_SUCCESS;
}

int32_t HcommWaitResponse(MsgHandle handle, void *dst, size_t sizeByte, uint32_t *msgId)
{
    uint8_t *const srcOnDevShmem = reinterpret_cast<uint8_t *>(handle);
    CHK_PTR_NULL(srcOnDevShmem);
    if (sizeByte > 0) {
        CHK_PTR_NULL(dst);
    }
    CHK_PTR_NULL(msgId);

    HCCL_INFO("[%s] START. msgHandle[0x%llx], dst[0x%llx], sizeByte[%zu].", __func__, handle, dst, sizeByte);

    constexpr size_t sizeByteMsgId = sizeof(uint32_t);
    uint8_t flagReadValue{0};
    uint8_t *const srcFlagPtr = srcOnDevShmem;
    uint8_t *const srcMsgIdPtr = srcFlagPtr + sizeof(flagReadValue) + MSG_TAG_SIZE_BYTE;
    uint8_t *const srcTimeoutPtr = srcMsgIdPtr + sizeByteMsgId;
    uint8_t *const srcDataPtr = srcTimeoutPtr + TIMEOUT_SIZE_BYTE;
    errno_t ret = EOK;

    CHK_RET(WaitFlagReady(srcFlagPtr, srcTimeoutPtr));

    if (sizeByte > 0) {
        HCCL_INFO("[%s] Reading %zu bytes data from shared mem START.", __func__, sizeByte);
        ret = memcpy_s(dst, sizeByte, srcDataPtr, sizeByte);
        CHK_PRT_RET(ret != EOK, HCCL_ERROR("[%s][memcpy_s] Reading data ERROR[%d]", __func__, ret), HCCL_E_INTERNAL);
        HCCL_INFO("[%s] Reading %zu bytes data from shared mem SUCCESS.", __func__, sizeByte);
    }

    HCCL_INFO("[%s] Reading %zu bytes msgId from shared mem START.", __func__, sizeByteMsgId);
    ret = memcpy_s(msgId, sizeByteMsgId, srcMsgIdPtr, sizeByteMsgId);
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[%s][memcpy_s] Reading msgId ERROR[%d].", __func__, ret), HCCL_E_INTERNAL);
    HCCL_INFO("[%s] Reading %zu bytes msgId from shared mem SUCCESS. msgId = %u.", __func__, sizeByteMsgId, *msgId);

    HCCL_INFO("[%s] Setting flag = 0 on shared mem START.", __func__);
    ret = memset_s(srcFlagPtr, sizeof(flagReadValue), 0, sizeof(flagReadValue));
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[%s][memset_s] Resetting flag ERROR[%d]", __func__, ret), HCCL_E_INTERNAL);
    HCCL_INFO("[%s] Setting flag = 0 on shared mem SUCCESS.", __func__);

    HCCL_INFO("[%s] SUCCESS. msgId[%u].", __func__, *msgId);
    return HCCL_SUCCESS;
}

int32_t HcommThreadSynchronize(ThreadHandle thread)
{
    hccl::Thread *threadPtr = reinterpret_cast<hccl::Thread *>(thread);
    CHK_PTR_NULL(threadPtr);

    HCCL_INFO("[%s] START. thread[0x%llx].", __func__, thread);

    if (threadPtr->IsDeviceA5()) {
        HCCL_INFO("[%s] Running on A5.", __func__);
        hccl::AicpuTsThread *aicpuTsThreadPtr = dynamic_cast<hccl::AicpuTsThread *>(threadPtr);
        uint32_t sqHead{0};
        uint32_t sqTail{0};
        HCCL_INFO("[%s] Start waiting for RTSQ's head == tail.", __func__);
        do {
            CHK_RET(aicpuTsThreadPtr->GetSqHeadAndTail(sqHead, sqTail));
        } while (sqHead != sqTail);
        HCCL_INFO("[%s] SUCCESS. RTSQ's head == tail.", __func__);
        return HCCL_SUCCESS;
    }

    HCCL_INFO("[%s] NOT Running on A5. No implementation, return SUCCESS.", __func__);
    return HCCL_SUCCESS;
}
#ifdef __cplusplus
}
#endif  // __cplusplus
