/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_device_pub.h"

#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <execinfo.h>
#include <map>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "sim_log.h"
#include "store_sim_memory_manager.h"
#include "db_sim_runner_db.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

uint32_t g_rankId = 0;

HcclVmResult SetCurRankId(uint32_t rankId)
{
    g_rankId = rankId;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult GetCurRankId(uint32_t *rankId)
{
    *rankId = g_rankId;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

uint8_t sqBuffer[HCCL_SQE_SIZE * HCCL_SQE_MAX_CNT];
HcclVmResult GetSqBufferAddr(uint8_t **sqBuff)
{
    *sqBuff = sqBuffer;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

std::map<uint32_t, uint32_t> jettyId2PiValMap;
HcclVmResult GetPiValByJettyId(uint32_t jettyId, uint32_t *piValue)
{
    *piValue = jettyId2PiValMap[jettyId];
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult UpdatePiValByJettyId(uint32_t jettyId, uint32_t piValue)
{
    jettyId2PiValMap[jettyId] = piValue;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

std::map<uint32_t, uint32_t> sqTailMap;
uint32_t GetSqTail(uint32_t sqId)
{
    return sqTailMap[sqId];
}

void UpdateSqTail(uint32_t sqId, uint32_t newTail)
{
    sqTailMap[sqId] = newTail;
}

void *GetRealPtrByDevPtrImpl(void *devPtr, const char *file, int line)
{
    uint64_t devAddr = reinterpret_cast<uint64_t>(devPtr);
    auto virMemRes = RunnerDB::GetOneByPred<sim::VirtualMemBlock>(
        [devAddr](const sim::VirtualMemBlock &virMem)
        { return ((virMem.start_ptr <= devAddr) &&
                    (devAddr < (virMem.start_ptr + virMem.size)) &&
                    (virMem.src_type == (uint8_t)sim::VIR_MEM_TYPE_DEV)); });
    if (!virMemRes.second) {
        HCCL_VM_ERROR("cannot find virMemRes by devAddr[{}], called from {}:{}]", devAddr, file, line);
        return nullptr;
    }

    auto phyMemId = virMemRes.first.phy_mem_id;
    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(phyMemId);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("cannot find phyMemRes by phyMemId[{}], called from {}:{}]", phyMemId, file, line);
        return nullptr;
    };

    std::string memName(phyMemRes->name);
    void *realPtr = sim::MemoryManager::GetInstance().AcquireMemByName(memName.c_str());
    HCCL_VM_INFO("devAddr[{}] realPtr[{}] memName[{}]", devAddr, reinterpret_cast<uint64_t>(realPtr), memName);

    return realPtr;
}

uint32_t GetRankIdByDevAddr(uint64_t devAddr)
{
    auto virMemRes = RunnerDB::GetOneByPred<sim::VirtualMemBlock>(
        [devAddr](const sim::VirtualMemBlock &virMem)
        { return ((virMem.start_ptr <= devAddr) &&
                    (devAddr < (virMem.start_ptr + virMem.size)) &&
                    (virMem.src_type == (uint8_t)sim::VIR_MEM_TYPE_DEV)); });
    if (!virMemRes.second) {
        HCCL_VM_ERROR("cannot find virMemRes by devAddr[{}]", devAddr);
        return 0;
    }

    auto phyMemId = virMemRes.first.phy_mem_id;
    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(phyMemId);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("cannot find phyMemRes by phyMemId[{}]", phyMemId);
        return 0;
    };

    uint32_t device_id = phyMemRes->device_id;
    auto rank = RunnerDB::GetOneByPred<sim::Rank>([device_id](const sim::Rank &rank) { return rank.device_id == device_id; });
    if (!rank.second) {
        HCCL_VM_ERROR("cannot find rank by device_id[{}]", device_id);
        return 0;
    }

    return rank.first.rank_id;
}

HcclAicpuData *GetHcclAicpuDataShmPtr()
{
    void *shmptr = sim::MemoryManager::GetInstance().AcquireMemByName("HcclAicpuData");
    if (shmptr == nullptr) {
        HCCL_VM_ERROR("acquire HcclAicpuData shm failed.");
        return nullptr;
    }

    return reinterpret_cast<HcclAicpuData *>(shmptr);
}

uint32_t GetRankIdByIpAddr(std::string ipAddr)
{
    auto ret = RunnerDB::GetOneByPred<sim::EndPoint>([ipAddr](const sim::EndPoint &p) { return strcmp(p.ip_addr, ipAddr.c_str()) == 0; });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find Port by ipAddr[{}]", ipAddr);
        return 0;
    }

    uint32_t device_id = ret.first.device_id;
    auto rank = RunnerDB::GetOneByPred<sim::Rank>([device_id](const sim::Rank &rank) { return rank.device_id == device_id; });
    if (!rank.second) {
        HCCL_VM_ERROR("cannot find Rank by device_id[{}]", device_id);
        return 0;
    }

    return rank.first.rank_id;
}

void UpdataKfcStatus(uint64_t d2hAddr)
{
    if (d2hAddr == 0) {
        HCCL_VM_ERROR("d2hAddr is 0, cannot update kfc status.");
        return;
    }

    uint8_t devDoneStatus = 3;
    constexpr uint32_t headTailCnt = 66;
    constexpr uint32_t headCntOffset = 4088;
    constexpr uint32_t tailCntOffset = 4092;
    *reinterpret_cast<uint8_t *>(d2hAddr) = devDoneStatus;  // 更新设备状态
    *reinterpret_cast<uint32_t *>(d2hAddr + headCntOffset) = headTailCnt;  // 更新headCnt
    *reinterpret_cast<uint32_t *>(d2hAddr + tailCntOffset) = headTailCnt;  // 更新tailCnt
    HCCL_VM_INFO("update kfc status success.");
}

// 信号处理函数
void SignalHandler(int signum)
{
    const int MAX_STACK_FRAMES = 64;
    void* stack_pointers[MAX_STACK_FRAMES];

    // 1. 获取调用栈地址
    int count = backtrace(stack_pointers, MAX_STACK_FRAMES);
    
    // 2. 打印基本错误信息
    HCCL_VM_ERROR("device process crash captured.");
    HCCL_VM_INFO("========================================");
    HCCL_VM_INFO("  ERROR CRASH DETECTED! Signal: {}", signum);
    HCCL_VM_INFO("========================================");
    
    // 3. 将地址转换为符号信息（函数名+ELF内偏移，供addr2line解析行号）
    for (int i = 0; i < count; ++i) {
        Dl_info info;
        if (dladdr(stack_pointers[i], &info) && info.dli_fname != nullptr) {
            ptrdiff_t elfOffset = reinterpret_cast<char*>(stack_pointers[i])
                - reinterpret_cast<char*>(info.dli_fbase);
            if (info.dli_sname != nullptr) {
                ptrdiff_t symOff = reinterpret_cast<char*>(stack_pointers[i])
                    - reinterpret_cast<char*>(info.dli_saddr);
                HCCL_VM_INFO(" [{:02d}] {}({}+{:#x}) [addr: {:#x}]", i,
                    info.dli_fname, info.dli_sname, symOff, elfOffset);
            } else {
                HCCL_VM_INFO(" [{:02d}] {}(?""?) [addr: {:#x}]", i, info.dli_fname, elfOffset);
            }
        } else {
            HCCL_VM_INFO(" [{:02d}] {} (dladdr failed)", i, stack_pointers[i]);
        }
    }
    
    HCCL_VM_INFO("========================================");
    HCCL_VM_INFO("Resolve: aarch64-linux-gnu-addr2line -e ./bin/device -C -f <addr>");
    HCCL_VM_INFO("========================================");

    // 4. 刷新缓冲区，确保信息在进程退出前输出
    fflush(stdout);

    // 5. 恢复默认处理并退出
    signal(signum, SIG_DFL);
    exit(signum);
}

// 注册所有崩溃相关的信号
void RegisterSignalHandler()
{
    signal(SIGSEGV, SignalHandler); // 段错误
    signal(SIGABRT, SignalHandler); // Abort (如 assert 失败)
    signal(SIGFPE,  SignalHandler); // 算术溢出/除零
    signal(SIGBUS,  SignalHandler); // 总线错误
    signal(SIGILL,  SignalHandler); // 非法指令
}

#ifdef __cplusplus
}
#endif
