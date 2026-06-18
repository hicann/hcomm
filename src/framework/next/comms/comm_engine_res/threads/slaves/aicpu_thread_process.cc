/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_thread_process.h"
#include <iomanip>

using namespace hccl;

std::mutex AicpuThreadProcess::mutex_;
std::vector<std::shared_ptr<hccl::Thread>> AicpuThreadProcess::threads_;

HcclResult AicpuThreadProcess::InitThreads(ThreadMgrAicpuParam *param)
{
    CHK_PTR_NULL(param);
    u32 threadNum = param->threadNum;
    std::vector<std::shared_ptr<Thread>> outThreads;
    outThreads.reserve(threadNum);
    std::string hcomId(param->hcomId);
    CHK_RET(AicpuThreadProcess::ResumeThread(param, outThreads, false));

    ThreadHandle *threadArray = static_cast<ThreadHandle*>(param->deviceHandle);
    // 空指针校验
    CHK_PTR_NULL(threadArray);
    for (size_t i = 0; i < threadNum; ++i) {
        threadArray[i] = reinterpret_cast<ThreadHandle>(outThreads[i].get());  // 拷贝裸指针
        HCCL_INFO("[HcclCommAicpu][%s] threadArray[%zu] = [%lu]", __func__, i, threadArray[i]);
    }
    threads_.insert(threads_.end(), std::make_move_iterator(outThreads.begin()),
        std::make_move_iterator(outThreads.end()));
    HCCL_INFO("[HcclCommAicpu][%s] comm identifier[%s], init threads num[%u] success",
        __func__, hcomId.c_str(), threadNum);
    return HCCL_SUCCESS;
}

HcclResult AicpuThreadProcess::AicpuThreadInit(ThreadMgrAicpuParam *param)
{
    CHK_RET(hrtSetWorkModeAicpu(true));
    CHK_RET(hrtSetlocalDevice(param->deviceLogicId));
    CHK_RET(hrtSetlocalDeviceType(static_cast<DevType>(param->deviceType)));
    std::lock_guard<std::mutex> addLock(mutex_);
    HcclResult ret = InitThreads(param);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AicpuThreadProcess][AicpuIndOpThreadInit]errNo[0x%016llx] Failed to init threads",
        HCCL_ERROR_CODE(ret)), ret);
    return HCCL_SUCCESS;
}

HcclResult AicpuThreadProcess::AicpuThreadDestroy(ThreadMgrAicpuParam *param)
{
    HCCL_INFO("[AicpuThreadProcess][%s] threadNum[%u]", __func__, param->threadNum);
    std::lock_guard<std::mutex> addLock(mutex_);
    ThreadHandle *threadArray = static_cast<ThreadHandle*>(param->deviceHandle);
    CHK_PTR_NULL(threadArray);

    for (u32 i = 0; i < param->threadNum; ++i) {
        ThreadHandle handle = threadArray[i];
        auto it = std::find_if(threads_.begin(), threads_.end(),
            [handle](const std::shared_ptr<Thread> &ptr) {
                return reinterpret_cast<ThreadHandle>(ptr.get()) == handle;
            });
        if (it == threads_.end()) {
            HCCL_WARNING("[AicpuThreadProcess][%s] thread handle[0x%llx] not found in threads_", __func__, handle);
            continue; // 继续处理其他线程
        }
        // 从容器中移除，shared_ptr 自动释放对象
        threads_.erase(it);
        HCCL_DEBUG("[AicpuThreadProcess][%s] destroyed thread handle[0x%llx]", __func__, handle);
    }

    HCCL_INFO("[AicpuThreadProcess][%s] success", __func__);
    return HCCL_SUCCESS;
}

constexpr u32 MIN_WIDTH = 2;
HcclResult AicpuThreadProcess::ResumeThread(ThreadMgrAicpuParam *param,
    std::vector<std::shared_ptr<Thread>> &outThreads, bool isSupplementNotify)
{
    CHK_PTR_NULL(param);
    u32 threadNum = param->threadNum;
    std::string hcomId(param->hcomId);
    ThreadHandle *threadArray = static_cast<ThreadHandle*>(param->deviceHandle);
    for (u32 i = 0; i < threadNum; ++i) {
        std::string thdUniqueId(param->threadParam[i], THREAD_UNIQUE_ID_MAX_SIZE);
        if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
            std::ostringstream oss;
            oss << "threadParam[" << i << "] raw bytes: ";
            for (u32 j = 0; j < THREAD_UNIQUE_ID_MAX_SIZE; ++j) {
                oss << std::hex << std::setw(MIN_WIDTH) << std::setfill('0')
                    << static_cast<unsigned int>(static_cast<unsigned char>(param->threadParam[i][j])) << " ";
            }
            HCCL_INFO("[HcclCommAicpu][%s] %s", __func__, oss.str().c_str());
        }
        std::shared_ptr<AicpuTsThread> thread;
        EXECEPTION_CATCH((thread = std::make_shared<AicpuTsThread>(thdUniqueId)), return HCCL_E_PTR);
        u32 notifyNum = 0;
        std::string notifyDesc;
        CHK_RET(thread->GetNotifyByUniqueId(notifyNum, notifyDesc));
        if (isSupplementNotify) {
            AicpuTsThread *threadPtr = reinterpret_cast<AicpuTsThread*>(threadArray[i]);
            CHK_PTR_NULL(threadPtr);
            HCCL_INFO("[%s]threadIdx[%u], threadHandle[%llu], notifyNum[%u], newNotifyNum[%u]", __func__, i,
                threadArray[i], threadPtr->GetNotifyNum(), notifyNum);
            CHK_RET(threadPtr->SupplementNotify(notifyNum, notifyDesc));
        } else {
            HcclResult ret = thread->Init();
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[HcclCommAicpu][%s] comm identifier[%s], init threads num[%u] failed at index %u",
                    __func__, hcomId.c_str(), param->threadNum, i);
                return ret;
            }
            outThreads.emplace_back(thread);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuThreadProcess::AicpuThreadSupplementNotify(ThreadMgrAicpuParam *param)
{
    CHK_PTR_NULL(param);
    u32 threadNum = param->threadNum;
    std::string hcomId(param->hcomId);
    std::vector<std::shared_ptr<Thread>> outThreads;
    CHK_RET(AicpuThreadProcess::ResumeThread(param, outThreads, true));

    HCCL_INFO("[HcclCommAicpu][%s] comm identifier[%s], init threads num[%u] success",
        __func__, hcomId.c_str(), threadNum);
    return HCCL_SUCCESS;
}