/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <memory>
#include <unordered_map>
#include <vector>

#include "hcomm_c_adpt.h"
#include "hcomm_c_adpt_common.h"
#include "hcomm_res.h"
#include "hcomm_res_defs.h"
#include "log.h"
#include "thread.h"
#include "cpu_ts_thread.h"
#include "param_check_pub.h"
#include "comm_engine_utils.h"
#include "exception_handler.h"
#include "adapter_rts_common.h"
#include "aicpu_ts_channel_helper.h"

namespace hcomm {
static std::unordered_map<ThreadHandle, std::shared_ptr<hccl::Thread>> g_ThreadMap;
} // namespace hcomm

using namespace hcomm;

HcommResult
HcommThreadAlloc(CommEngine engine, uint32_t threadNum, const uint32_t* notifyNumPerThread, ThreadHandle* threads)
{
    CHK_PTR_NULL(threads);
    CHK_PTR_NULL(notifyNumPerThread);
    (void)HcommResMgrInit();
    const uint32_t notifyNum = notifyNumPerThread[0];
    if (threadNum > 1U) {
        HCCL_RUN_WARNING(
            "[%s] only notifyNumPerThread[0] is used currently, threadNum[%u], notifyNum[0][%u].", __func__, threadNum,
            notifyNum);
    }
    HCCL_INFO(
        "[%s] ThreadAcquire begin. engine[%s], threadNum[%u], notifyPerThread[%u], threads[%p]", __func__,
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum, notifyNum, threads);
    CHK_RET(RefreshCommEngineContext(engine));

    // 1. 参数校验
    CHK_RET(hccl::ValidateThreadParams(threadNum, notifyNum));

    // 2. 获取引擎对应的类型
    hccl::NotifyLoadType notifyLoadType;
    hccl::StreamType streamType;
    CHK_RET(hccl::CommEngineToNotifyLoadType(engine, notifyLoadType));
    CHK_RET(hccl::CommEngineToStreamType(engine, streamType));

    // 3. 创建线程
    std::vector<std::shared_ptr<hccl::Thread>> newThreads;
    hccl::ThreadCreateParams params(engine, threadNum, notifyNum, notifyLoadType, streamType);
    CHK_RET(hccl::CreateAndInitThreads(params, newThreads));

    // 4. 插入全局映射表
    CHK_RET(hccl::SaveThreads(newThreads));

    // 5. 储存线程句柄
    CHK_RET(AicpuTsChannelHelper::EnsureKernelBinLoaded(engine));
    CHK_RET(hccl::StoreThreadHandles(newThreads, threads, engine, AicpuTsChannelHelper::GetBinHandle()));

    HCCL_INFO(
        "[HcommThreadAlloc] ThreadAcquire done: engine[%s] threadNum[%u], notifyPerThread[%u]",
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), threadNum, notifyNum);
    return HCCL_SUCCESS;
}

HcommResult HcommThreadAlloc(CommEngine engine, uint32_t threadNum, uint32_t notifyNumPerThread, ThreadHandle* threads)
{
    return ::HcommThreadAlloc(engine, threadNum, &notifyNumPerThread, threads);
}

HcommResult HcommThreadAllocWithConfig(
    CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config, ThreadHandle* threads)
{
    CHK_PTR_NULL(threads);
    CHK_PTR_NULL(config);
    CHK_PRT_RET(
        type == THREAD_TYPE_INVALID,
        HCCL_ERROR("[%s] thread type[%d] is invalid", __func__, static_cast<int32_t>(type)), (HcommResult)HCCL_E_PARA);
    CHK_PRT_RET(
        engine == COMM_ENGINE_AICPU_TS || engine == COMM_ENGINE_CPU_TS,
        HCCL_ERROR(
            "[%s] commEngine[%d] CPU_TS/AICPU_TS not supported, use engine with ThreadType instead", __func__,
            static_cast<int32_t>(engine)),
        (HcommResult)HCCL_E_PARA);
    CHK_PRT_RET(
        engine == COMM_ENGINE_AIV || engine == COMM_ENGINE_CCU,
        HCCL_ERROR(
            "[%s] commEngine[%d] AIV/CCU not supported, supported engines: CPU/AICPU", __func__,
            static_cast<int32_t>(engine)),
        (HcommResult)HCCL_E_PARA);
    CHK_PRT_RET(
        threadNum == 0, HCCL_ERROR("[%s] threadNum[%u] is invalid", __func__, threadNum), (HcommResult)HCCL_E_PARA);
    HcommResult hcommRet = HcommResMgrInit();
    CHK_PRT_RET(
        hcommRet != HCCL_SUCCESS,
        HCCL_ERROR("[%s] HcommResMgrInit failed, ret[%d]", __func__, static_cast<int32_t>(hcommRet)), hcommRet);
    CHK_RET(RefreshCommEngineContext(engine));

    HCCL_INFO(
        "[%s] begin. engine[%d], threadType[%d], threadNum[%u], threads[%p]", __func__, engine,
        static_cast<int32_t>(type), threadNum, threads);

    hccl::NotifyLoadType notifyLoadType;
    hccl::StreamType streamType;
    CHK_RET(hccl::GetNotifyLoadType(engine, type, notifyLoadType));
    CHK_RET(hccl::GetStreamType(engine, type, streamType));

    std::vector<std::shared_ptr<hccl::Thread>> newThreads;
    newThreads.reserve(threadNum);
    for (uint32_t i = 0; i < threadNum; ++i) {
        CHK_PRT_RET(
            config[i].header.magicWord != HCOMM_THREAD_CONFIG_MAGIC_WORD,
            HCCL_ERROR(
                "[%s] config[%u] magicWord[0x%x] mismatch, expected[0x%x], call ThreadConfigInit first", __func__, i,
                config[i].header.magicWord, HCOMM_THREAD_CONFIG_MAGIC_WORD),
            (HcommResult)HCCL_E_PARA);
        CHK_RET(hccl::ValidateThreadParams(1, config[i].notifyNumPerThread));
        std::shared_ptr<hccl::Thread> threadPtr;
        HcclResult ret
            = hccl::CreateThread(engine, streamType, config[i].notifyNumPerThread, notifyLoadType, threadPtr);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s] Failed to create thread at index[%u], ret[%d]", __func__, i, ret),
            (HcommResult)ret);
        ret = threadPtr->Init();
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s] Failed to init thread at index[%u], ret[%d]", __func__, i, ret),
            (HcommResult)ret);
        newThreads.emplace_back(std::move(threadPtr));
    }

    CHK_RET(hccl::SaveThreads(newThreads));
    CHK_RET(AicpuTsChannelHelper::EnsureKernelBinLoaded(engine));
    CHK_RET(hccl::StoreThreadHandles(newThreads, threads, engine, AicpuTsChannelHelper::GetBinHandle()));

    HCCL_INFO(
        "[%s] done: engine[%d] threadType[%d] threadNum[%u]", __func__, engine, static_cast<int32_t>(type), threadNum);
    return HCCL_SUCCESS;
}

HcommResult HcommThreadFree(const ThreadHandle* threads, uint32_t threadNum)
{
    CHK_PTR_NULL(threads);
    (void)HcommResMgrInit();
    return hccl::FreeThreads(threads, threadNum, AicpuTsChannelHelper::GetBinHandle());
}

HcommResult HcommThreadAllocWithStream(CommEngine engine, rtStream_t stream, uint32_t notifyNum, ThreadHandle* thread)
{
    CHK_PTR_NULL(thread);
    hccl::NotifyLoadType notifyLoadType;
    CHK_RET(CommHostEngineToNotifyLoadType(engine, notifyLoadType));
    std::shared_ptr<hccl::Thread> handle;
    EXCEPTION_CATCH(handle = std::make_shared<hccl::CpuTsThread>(stream, notifyNum, notifyLoadType), return HCCL_E_PTR);
    CHK_RET(handle->Init());

    // 返回第一个句柄
    *thread = reinterpret_cast<ThreadHandle>(handle.get());
    hcomm::g_ThreadMap.emplace(*thread, handle);

    HCCL_INFO(
        "[ThreadMgr] ThreadAcquireWithStream done: engine[%s] stream[%p], "
        "notifyNum[%u]",
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), stream, notifyNum);
    return HCCL_SUCCESS;
}
