/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_comm.h"
#include "exception_handler.h"
#include "rank_graph_v2.h"
#include "kfc.h"
#include "dlhal_function.h"
#include "hcclCommTaskException.h"
#include "symmetric_memory/symmetric_memory.h"
#include "hccl_team_mgr.h"

#include <cstdint>
#include <exception>
#include "launch_aicpu.h"
#include "launch_device.h"

namespace hccl {
void SymmetricMemoryDeleter::operator()(SymmetricMemory* ptr) const { delete ptr; }

CollComm::CollComm(
    void* comm, uint32_t rankId, const std::string& commName, const ManagerCallbacks& callbacks,
    CollCommInitMode initMode)
    : comm_(comm),
      rankId_(rankId),
      commId_(commName),
      config_(commName),
      callbacks_(callbacks),
      initMode_(initMode)
{
    groupScheduleMgr = std::make_shared<GroupScheduleMgr>();
}

CollComm::~CollComm()
{
    if (!IsFullMode()) { // SimpleMode是简化版collComm，只初始化了myrank、rankgraph未初始化以下资源，不需要析构处理
        return;
    }

    // 先注销TaskException，再销毁通信域资源，防止通信域资源销毁后rts回调TaskException
    hcomm::TaskExceptionHost* handler = hcomm::TaskExceptionHost::GetInstance(deviceLogicId_);
    if (handler != nullptr) {
        (void)handler->UnRegister(reinterpret_cast<u64>(this));
    }

    CHK_PRT(HcclBinaryUnLoad());

    // 兜底释放所有team的syncMem本地内存
    HcclTeamMgr::GetInstance().ClearByCollComm(this);
    HCCL_INFO("[CollComm][~CollComm] collComm deinit");
    (void)DestroyAicpuComm();
    HCCL_RUN_INFO("[CollComm][~CollComm] cclBuffer free, commId[%s].", commId_.c_str());
}

HcclResult CollComm::Init(void* rankGraph, aclrtBinHandle binHandle, HcclMem cclBuffer, uint32_t opExpansionMode)
{
    if (IsFullMode()) { // A5和下一代
        return InitFullMode(rankGraph, binHandle, cclBuffer, opExpansionMode);
    } else { // A2/A3使用简化版CollComm
        return InitSimpleMode(rankGraph, binHandle, cclBuffer, opExpansionMode);
    }
}

HcclResult
CollComm::InitSimpleMode(void* rankGraph, aclrtBinHandle binHandle, HcclMem cclBuffer, uint32_t opExpansionMode)
{
    CHK_PTR_NULL(rankGraph);

    EXCEPTION_HANDLE_BEGIN

    CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());

    // SimpleMode: A2/A3的RankGraph是保存在hccl::Communicator中的静态对象裸指针，CollComm不负责释放
    rankgraph_ = static_cast<RankGraph*>(rankGraph);

    uint32_t rankNum = 0;
    CHK_PTR_NULL(rankgraph_);
    CHK_RET(rankgraph_->GetRankSize(&rankNum));

    EXCEPTION_CATCH(
        myRank_ = std::make_shared<MyRank>(binHandle, rankId_, config_, callbacks_, rankgraph_, rankIpPortMap_),
        return HCCL_E_PTR);

    CHK_RET(myRank_->Init(cclBuffer, opExpansionMode, rankNum));

    commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_READY;

    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult
CollComm::InitFullMode(void* rankGraph, aclrtBinHandle binHandle, HcclMem cclBuffer, uint32_t opExpansionMode)
{
    CHK_PTR_NULL(rankGraph);

    EXCEPTION_HANDLE_BEGIN

    CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());
    rankGraphOwner_ = std::make_unique<RankGraphV2>(rankGraph);
    rankgraph_ = rankGraphOwner_.get();
    uint32_t rankNum = 0;
    CHK_PTR_NULL(rankgraph_);
    CHK_RET(rankgraph_->GetRankSize(&rankNum));
    CHK_RET(GetRankIpPortMap());

    u32 threadNum = 0xffffffff;
    u32 notifyNumPerThread = 0xffffffff;
    if (!commEngineResMgr_) {
        EXCEPTION_CATCH(commEngineResMgr_ = std::make_unique<CommEngineResMgr>(), return HCCL_E_PTR);
        CHK_PRT(commEngineResMgr_->Init(threadNum, notifyNumPerThread, commId_, binHandle, callbacks_));
    }

    if (!contextMgr_) {
        EXCEPTION_CATCH(contextMgr_ = std::make_unique<ContextManager>(), return HCCL_E_PTR);
    }

    EXCEPTION_CATCH(
        myRank_ = std::make_shared<MyRank>(binHandle, rankId_, config_, callbacks_, rankgraph_, rankIpPortMap_),
        return HCCL_E_PTR);
    CHK_RET(myRank_->Init(cclBuffer, opExpansionMode, rankNum));
    CHK_RET(hrtGetDevice(&deviceLogicId_));
    CHK_RET(InitSymmetricMemory());

    CHK_RET(InitHDCommunicate());

    if (!hcclCommDfx_) {
        EXCEPTION_CATCH(hcclCommDfx_ = std::make_unique<HcclCommDfx>(), return HCCL_E_PTR);
    }
    CHK_RET(hcclCommDfx_->Init(deviceLogicId_, commId_, rankId_));
    CHK_RET(InitTaskExceptionHandler());

    CHK_RET(InitKfcAndRegisterCollComm());

    Hccl::HcclCommunicator* comV2 = static_cast<Hccl::HcclCommunicator*>(comm_);
    CHK_PTR_NULL(comV2);
    CHK_RET(comV2->GetCclBufferSharedPtr(cclBuffer_));

    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult CollComm::InitSymmetricMemory()
{
    uint32_t rankSize = GetRankSize();
    HCCL_RUN_INFO(
        "[CollComm][InitSymmetricMemory] commId[%s], rank[%u], rankSize[%u].", commId_.c_str(), rankId_, rankSize);

    EXCEPTION_CATCH(
        symmetricMemory_.reset(new SymmetricMemory(rankId_, rankSize, 0, SymmetricMemoryMode::URMA)),
        return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(symmetricMemory_);
    return HCCL_SUCCESS;
}

HcclResult CollComm::RegisterSymmetricMemoryResource(void* ptr, size_t size, SymmetricMemoryResource& resource)
{
    CHK_PTR_NULL(ptr);
    CHK_PRT_RET(
        size == 0, HCCL_ERROR("[CollComm][RegisterSymmetricMemoryResource] invalid symmetric memory size 0."),
        HCCL_E_PARA);
    CHK_SMART_PTR_NULL(myRank_);

    CommMems* commMems = myRank_->GetCommMems();
    CHK_PTR_NULL(commMems);

    CommMem commMem{};
    commMem.type = COMM_MEM_TYPE_DEVICE;
    commMem.addr = ptr;
    commMem.size = static_cast<uint64_t>(size);
    resource.memTag = std::string(HCCL_SYMMETRIC_MEMORY_TAG_PREFIX) + commId_ + "_addr_"
                      + std::to_string(reinterpret_cast<uintptr_t>(ptr)) + "_size_" + std::to_string(size);
    HcclResult ret = commMems->CommRegMem(resource.memTag, commMem, &resource.memHandle);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollComm][RegisterSymmetricMemoryResource] CommRegMem failed, tag[%s], ptr[%p], "
            "size[%zu], ret[%d].",
            resource.memTag.c_str(), ptr, size, ret),
        ret);

    HCCL_RUN_INFO(
        "[CollComm][RegisterSymmetricMemoryResource] register symmetric memory success, group[%s], "
        "tag[%s], ptr[%p], size[%zu], memHandle[%p].",
        commId_.c_str(), resource.memTag.c_str(), ptr, size, resource.memHandle);
    return HCCL_SUCCESS;
}

void CollComm::UnregisterSymmetricMemoryResource(const SymmetricMemoryResource& resource)
{
    if (resource.memHandle == nullptr || resource.memTag.empty()) {
        HCCL_WARNING(
            "[CollComm][UnregisterSymmetricMemoryResource] invalid resource, tag[%s], memHandle[%p].",
            resource.memTag.c_str(), resource.memHandle);
        return;
    }
    if (myRank_ == nullptr) {
        HCCL_WARNING(
            "[CollComm][UnregisterSymmetricMemoryResource] myRank is null, skip CommUnregMem, "
            "tag[%s], memHandle[%p].",
            resource.memTag.c_str(), resource.memHandle);
        return;
    }
    CommMems* commMems = myRank_->GetCommMems();
    if (commMems == nullptr) {
        HCCL_WARNING(
            "[CollComm][UnregisterSymmetricMemoryResource] commMems is null, skip CommUnregMem, "
            "tag[%s], memHandle[%p].",
            resource.memTag.c_str(), resource.memHandle);
        return;
    }
    HcclResult ret = commMems->CommUnregMem(resource.memTag, resource.memHandle);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING(
            "[CollComm][UnregisterSymmetricMemoryResource] CommUnregMem failed, tag[%s], "
            "memHandle[%p], ret[%d].",
            resource.memTag.c_str(), resource.memHandle, ret);
    }
    ret = myRank_->UnregMemByTag(resource.memTag);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[CollComm][UnregisterSymmetricMemoryResource] UnregMemByTag failed, tag[%s], ret[%d].",
            resource.memTag.c_str(), ret);
        return;
    }
    HCCL_INFO(
        "[CollComm][UnregisterSymmetricMemoryResource] unregister symmetric memory success, "
        "tag[%s], memHandle[%p].",
        resource.memTag.c_str(), resource.memHandle);
}

HcclResult CollComm::RegisterWindow(void* ptr, size_t size, HcclCommSymWindow* winHandle)
{
    CHK_SMART_PTR_NULL(symmetricMemory_);
    return symmetricMemory_->RegisterUrmaSymmetricMem(ptr, size, winHandle);
}

HcclResult CollComm::DeregisterWindow(HcclCommSymWindow winHandle)
{
    CHK_SMART_PTR_NULL(symmetricMemory_);
    SymmetricMemoryResource resource;
    HcclResult getResourceRet = symmetricMemory_->GetRegisteredMemoryResource(winHandle, resource);
    CHK_PRT_RET(
        getResourceRet != HCCL_SUCCESS && getResourceRet != HCCL_E_NOT_FOUND,
        HCCL_ERROR(
            "[CollComm][DeregisterWindow] get registered symmetric memory resource failed, "
            "winHandle[%p], ret[%d].",
            winHandle, getResourceRet),
        getResourceRet);

    HcclResult ret = symmetricMemory_->DeregisterUrmaSymmetricMem(winHandle);
    if (ret == HCCL_SUCCESS && getResourceRet == HCCL_SUCCESS) {
        UnregisterSymmetricMemoryResource(resource);
    }
    return ret;
}

HcclResult CollComm::GetCommSymWin(void* ptr, size_t size, HcclCommSymWindow* winHandle, size_t* offset)
{
    CHK_SMART_PTR_NULL(symmetricMemory_);
    return symmetricMemory_->FindUrmaSymmetricWindow(ptr, size, winHandle, offset);
}

HcclResult CollComm::RegisterPendingSymmetricMemHandles(std::vector<HcclMemHandle>& memHandles)
{
    memHandles.clear();
    if (symmetricMemory_ == nullptr) {
        return HCCL_SUCCESS;
    }

    std::vector<SymmetricMemoryRegisterInfo> registerInfos;
    // HcclCommSymWinRegister只记录窗口，真正CommRegMem延迟到ChannelAcquire阶段执行。
    CHK_RET(symmetricMemory_->GetPendingRegisterInfos(registerInfos));
    if (registerInfos.empty()) {
        return HCCL_SUCCESS;
    }

    std::vector<std::pair<void*, SymmetricMemoryResource>> registeredResources;
    for (const SymmetricMemoryRegisterInfo& registerInfo : registerInfos) {
        SymmetricMemoryResource resource;
        HcclResult ret = RegisterSymmetricMemoryResource(registerInfo.userVa, registerInfo.userSize, resource);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[CollComm][RegisterPendingSymmetricMemHandles] register symmetric memory failed, "
                "win[%p], userVa[%p], size[%zu], ret[%d].",
                registerInfo.devWin, registerInfo.userVa, registerInfo.userSize, ret);
            for (const auto& registeredResource : registeredResources) {
                symmetricMemory_->RemoveRegisteredMemoryResource(registeredResource.first);
                UnregisterSymmetricMemoryResource(registeredResource.second);
            }
            return ret;
        }

        ret = symmetricMemory_->SetRegisteredMemoryResource(registerInfo.devWin, resource);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[CollComm][RegisterPendingSymmetricMemHandles] save symmetric memory resource failed, "
                "win[%p], userVa[%p], size[%zu], ret[%d].",
                registerInfo.devWin, registerInfo.userVa, registerInfo.userSize, ret);
            UnregisterSymmetricMemoryResource(resource);
            for (const auto& registeredResource : registeredResources) {
                symmetricMemory_->RemoveRegisteredMemoryResource(registeredResource.first);
                UnregisterSymmetricMemoryResource(registeredResource.second);
            }
            return ret;
        }
        registeredResources.emplace_back(registerInfo.devWin, resource);
        // 仅返回本次新注册的memHandle，避免普通URMA重复携带历史对称内存句柄。
        memHandles.emplace_back(static_cast<HcclMemHandle>(resource.memHandle));
    }

    return HCCL_SUCCESS;
}

HcclResult CollComm::UpdateSymmetricRemoteMem(
    uint32_t remoteRank, const CommMem* remoteMems, const std::vector<std::string>& memTags)
{
    if (symmetricMemory_ == nullptr) {
        return HCCL_SUCCESS;
    }
    return symmetricMemory_->UpdateRemoteMem(remoteRank, remoteMems, memTags);
}

HcclResult CollComm::InitKfcAndRegisterCollComm()
{
    myRank_->SetKfcControlTransfer(kfcControlTransferH2D_, kfcStatusTransferD2H_);
    commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_READY;
    return HCCL_SUCCESS;
}

HcclResult CollComm::DestroyAicpuComm()
{
    CHK_PTR_NULL(callbacks_.getAicpuCommState);
    if (callbacks_.getAicpuCommState()) {
        CHK_SMART_PTR_NULL(kfcControlTransferH2D_);
        CHK_SMART_PTR_NULL(kfcStatusTransferD2H_);

        Hccl::KfcCommand opCmd = Hccl::KfcCommand::DESTROY_AICPU_COMM;
        CHK_RET(kfcControlTransferH2D_->Put(0, sizeof(Hccl::KfcCommand), reinterpret_cast<uint8_t*>(&opCmd)));
        HCCL_RUN_INFO(
            "[%s]group[%s] send Hccl::KfcCommand[%d] success", __func__, commId_.c_str(), static_cast<int>(opCmd));

        Hccl::KfcExecStatus opInfo;
        constexpr u32 WAIT_CMD_TIMEOUT = 10 * 1000; // 最大等待10秒
        auto timeout = std::chrono::milliseconds(WAIT_CMD_TIMEOUT);
        auto startTime = std::chrono::steady_clock::now();

        while (true) {
            CHK_RET(kfcStatusTransferD2H_->Get(0, sizeof(Hccl::KfcExecStatus), reinterpret_cast<uint8_t*>(&opInfo)));
            if (opInfo.kfcStatus == Hccl::KfcStatus::DESTROY_AICPU_COMM_DONE) {
                HCCL_RUN_INFO("[%s]get Hccl::KfcStatus[%d] success", __func__, static_cast<int>(opInfo.kfcStatus));
                return HCCL_SUCCESS;
            } else if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
                HCCL_ERROR(
                    "[%s]timeout, maxTime[%u ms] and get the opExecStatus is [%s].", __func__, WAIT_CMD_TIMEOUT,
                    opInfo.kfcStatus.Describe().c_str());
                return HCCL_E_TIMEOUT;
            }
            usleep(TEN_MILLISECOND_OF_USLEEP);
        }
    }
    return HCCL_SUCCESS;
}

uint32_t CollComm::GetMyRankId() const { return rankId_; }

HcclResult CollComm::GetParentRankId(u32& parentRankId) const
{
    Hccl::HcclCommunicator* comV2 = static_cast<Hccl::HcclCommunicator*>(comm_);
    CHK_PTR_NULL(comV2);
    parentRankId = comV2->GetRankInParentComm();
    return HCCL_SUCCESS;
}

HcclResult CollComm::InitHDCommunicate()
{
    // 初始化aicpu进程 host-device 共享内存
    EXCEPTION_CATCH(
        (kfcControlTransferH2D_
         = std::make_shared<hccl::HDCommunicate>(deviceLogicId_, HCCL_HDC_TYPE_H2D, sizeof(Hccl::KfcCommand))),
        return HCCL_E_PTR);
    CHK_RET(kfcControlTransferH2D_->InitHost());

    EXCEPTION_CATCH(
        (kfcStatusTransferD2H_
         = std::make_shared<hccl::HDCommunicate>(deviceLogicId_, HCCL_HDC_TYPE_D2H, sizeof(Hccl::KfcExecStatus))),
        return HCCL_E_PTR);
    CHK_RET(kfcStatusTransferD2H_->InitHost());

    return HCCL_SUCCESS;
}

HcclResult CollComm::GetHDCommunicate(
    HDCommunicateParams& kfcControlTransferH2DParams, HDCommunicateParams& kfcStatusTransferD2HParams)
{
    CHK_SMART_PTR_NULL(kfcControlTransferH2D_);
    CHK_SMART_PTR_NULL(kfcStatusTransferD2H_);
    kfcControlTransferH2DParams = kfcControlTransferH2D_->GetCommunicateParams();
    kfcStatusTransferD2HParams = kfcStatusTransferD2H_->GetCommunicateParams();
    HCCL_INFO("%s success, group[%s]", __func__, commId_.c_str());
    return HCCL_SUCCESS;
}

HcclCommStatus CollComm::GetCommStatus() const
{
    std::lock_guard<std::mutex> lock(commMutex_);
    return commStatus_;
}

HcclResult CollComm::Suspend()
{
    HCCL_RUN_INFO("[CollComm][Suspend] commId[%s] start to suspend.", commId_.c_str());
    {
        std::lock_guard<std::mutex> lock(commMutex_);
        if (commStatus_ == HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING) {
            HCCL_WARNING("[CollComm][Suspend] The current communication has been suspended, no need to suspend again.");
            return HcclResult::HCCL_SUCCESS;
        }

        CHK_SMART_PTR_NULL(myRank_);

        commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING;
    }

    return myRank_->StopLaunch();
}

HcclResult CollComm::Clean()
{
    HCCL_RUN_INFO("[CollComm][Clean] commId[%s] start to clean.", commId_.c_str());
    {
        std::lock_guard<std::mutex> lock(commMutex_);
        if (commStatus_ != HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING) {
            HCCL_ERROR(
                "[CollComm][Clean] The current communication is not suspended, cannot clean, status is [%u]",
                static_cast<uint32_t>(commStatus_));
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }
        if (isCleaned_) {
            HCCL_WARNING("[CollComm][Clean] The current communication has been cleaned, no need to clean again.");
            return HcclResult::HCCL_SUCCESS;
        }

        CHK_SMART_PTR_NULL(myRank_);

        isCleaned_ = true;
    }

    // 先清理Host
    return myRank_->Clean();
}

HcclResult CollComm::Resume()
{
    {
        std::lock_guard<std::mutex> lock(commMutex_);
        if (commStatus_ == HcclCommStatus::HCCL_COMM_STATUS_INVALID) {
            HCCL_ERROR("[CollComm][Resume] Comm has been error, can not resume now!");
            return HcclResult::HCCL_E_INTERNAL;
        }
        if (commStatus_ != HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING) {
            HCCL_WARNING(
                "[CollComm][Resume] The current communication is normal, no need to resume, status is [%u]",
                static_cast<uint32_t>(commStatus_));
            return HcclResult::HCCL_SUCCESS;
        }

        HCCL_INFO("[CollComm][Resume] start to Resume.");
        CHK_SMART_PTR_NULL(myRank_);
        auto ret = myRank_->Resume();
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR("[CollComm][Resume] %s failed, ret = 0x%016llx", __func__, HCCL_ERROR_CODE(ret));
            return ret;
        }

        if (commEngineResMgr_ != nullptr) {
            auto notifyRet = commEngineResMgr_->ResetCommLocalNotifies();
            if (notifyRet != HcclResult::HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[CollComm][Resume] ResetCommLocalNotifies failed, ret = 0x%016llx", HCCL_ERROR_CODE(notifyRet));
                return notifyRet;
            }
        }

        commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_READY;
        isCleaned_ = false;
    }
    HCCL_INFO("[CollComm][Resume] commId[%s] resume success.", commId_.c_str());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CollComm::InitTaskExceptionHandler()
{
    hcomm::TaskExceptionHost* handler = hcomm::TaskExceptionHost::GetInstance(deviceLogicId_);
    CHK_PTR_NULL(handler);
    CHK_RET(handler->Register(reinterpret_cast<u64>(this)));
    return HCCL_SUCCESS;
}

Hccl::ErrorMessageReport CollComm::GetAicpuTaskException()
{
    Hccl::ErrorMessageReport errorMessage;
    CHK_PRT_RET(kfcStatusTransferD2H_ == nullptr, HCCL_ERROR("[%s]fail, d2h is nullptr", __func__), errorMessage);

    HcclResult ret = kfcStatusTransferD2H_->Get(
        sizeof(Hccl::KfcStatus) + sizeof(Hccl::KfcErrType), sizeof(errorMessage),
        reinterpret_cast<uint8_t*>(&errorMessage));

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]fail, group [%s], ret[%d]", __func__, commId_.c_str(), static_cast<int>(ret)), errorMessage);
    HCCL_INFO("[%s]group[%s] success", __func__, commId_.c_str());
    return errorMessage;
}

uint32_t CollComm::UpdateIndex() { return index_ += 1; }

HcclResult CollComm::GetRankIpPortMap()
{
    Hccl::HcclCommunicator* commV2 = static_cast<Hccl::HcclCommunicator*>(comm_);
    CHK_PTR_NULL(commV2);
    CHK_RET(commV2->GetRankIpPortMap(rankIpPortMap_));
    CHK_PTR_NULL(rankIpPortMap_);
    // rankIpPortMap_ 在单卡多进程场景下，用于保证端口不冲突
    // 该映射表记录了：Rank ID -> (IP地址 -> 已占用的端口号)
    return HCCL_SUCCESS;
}

HcclResult CollComm::GetHcclBinHandle(aclrtBinHandle& binHcclHandle)
{
    std::lock_guard<std::mutex> lock(binHcclmutex_);
    HCCL_DEBUG("[%s] GetHcclBinHandle", __func__);
    if (binHcclHandle_ == nullptr) {
        std::string hcclJsonPath;
        CHK_RET(GetKernelFilePath(hcclJsonPath));
        hcclJsonPath += "libscatter_aicpu_kernel.json";
        HcclResult ret
            = LoadBinaryFromFile(hcclJsonPath.c_str(), ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE, 0, binHcclHandle_);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s]errNo[0x%016llx]load aicpu file fail, path[%s] optionType[%u] cpuKernelMode[%u].", __func__,
                HCCL_ERROR_CODE(ret), hcclJsonPath.c_str(), ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE, 0),
            ret);

        HCCL_INFO(
            "[%s]load aicpu file success, path[%s] optionType[%u] cpuKernelMode[%u].", __func__, hcclJsonPath.c_str(),
            ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE, 0);
    }
    binHcclHandle = binHcclHandle_;
    return HCCL_SUCCESS;
}

HcclResult CollComm::HcclBinaryUnLoad()
{
    std::lock_guard<std::mutex> lock(binHcclmutex_);
    if (binHcclHandle_ == nullptr) {
        HCCL_RUN_WARNING("[%s] binHcclHandle is nullptr", __func__);
        return HCCL_SUCCESS;
    }

    HCCL_DEBUG("[%s]aclrtBinaryUnLoad binHcclHandle", __func__);
    aclError ret = aclrtBinaryUnLoad(binHcclHandle_);
    binHcclHandle_ = nullptr;
    if (ret != 0) {
        HCCL_RUN_WARNING("[%s]aclrtBinaryUnLoad failed, aclRet[%d]", __func__, ret);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

} // namespace hccl
