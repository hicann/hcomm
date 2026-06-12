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
// #include "rank_graphs/rank_graph.h"
#include "exception_handler.h"
#include "rank_graph_v2.h"
#include "coll_comm_mgr.h"
#include "kfc.h"
#include "dlhal_function.h"
#include "hcclCommTaskException.h"

constexpr uint32_t MULTIPLE = 4;               // 用于A5判断TC是否为4的倍数
constexpr uint32_t TC_MAX = 255;               // TC的最大值（不区分芯片类型）
constexpr uint32_t SL_MAX = 7u;                // sl范围的最大值，sl即serviceLevel（不区分芯片类型）
constexpr uint32_t TC_DEFAULT = 0xFFFFFFFFu;   // TC的默认值（不区分芯片类型）
constexpr uint32_t SL_DEFAULT = 0xFFFFFFFFu;   // SL的默认值（不区分芯片类型）

namespace hccl {
CollComm::CollComm(void * comm, uint32_t rankId, const std::string &commName, const ManagerCallbacks& callbacks)
    : comm_(comm), rankId_(rankId), commId_ (commName), callbacks_(callbacks)
{
}

CollComm::~CollComm()
{
    CollCommMgr::GetInstance()->UnRegisteCollComm(this); 
    HCCL_INFO("[CollComm][~CollComm] collComm deinit");
    // dpu的兜底上报 - 异常退出时捕获异常避免二次崩溃
    if (hcclCommDfx_ != nullptr) {
        // 析构属于最终阶段，不再存储数据，直接上报
        DECTOR_TRY_CATCH("CollComm", hcclCommDfx_->ReportAllTasks(false));
    }
    for (auto streamId : aicpuStreamIds_) {
        hcomm::TaskExceptionHostManager::UnregisterGetAicpuTaskExceptionCallBack(streamId, deviceLogicId_);
    }
    aicpuStreamIds_.clear();
    (void)DestroyAicpuComm();
}

HcclResult CollComm::Init(void * rankGraph, aclrtBinHandle binHandle, HcclMem cclBuffer, HcclCommConfig *config)
{
    CHK_PTR_NULL(rankGraph);

    EXCEPTION_HANDLE_BEGIN

    CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());
    EXECEPTION_CATCH(rankgraph_ = std::make_unique<RankGraphV2>(rankGraph), return HCCL_E_PTR);
    uint32_t rankNum = 0;
    CHK_PTR_NULL(rankgraph_);
    CHK_RET(rankgraph_->GetRankSize(&rankNum));
    CHK_RET(GetRankIpPortMap());

    u32 threadNum = 0xffffffff;
    u32 notifyNumPerThread = 0xffffffff;
    if (!commEngineResMgr_) {
        EXECEPTION_CATCH(commEngineResMgr_ = std::make_unique<CommEngineResMgr>(),
            return HCCL_E_PTR);
        CHK_PRT(commEngineResMgr_->Init(threadNum, notifyNumPerThread, commId_, binHandle, callbacks_));
    }

    if (!contextMgr_) {
        EXECEPTION_CATCH(contextMgr_ = std::make_unique<ContextManager>(), return HCCL_E_PTR);
    }

    uint32_t opExpansionMode = 0;
    CHK_RET(ApplyUserCommConfig(config, opExpansionMode));

    EXECEPTION_CATCH(
        myRank_ = std::make_shared<MyRank>(binHandle, rankId_, config_, callbacks_, rankgraph_.get(), rankIpPortMap_),
        return HCCL_E_PTR);
    CHK_RET(myRank_->Init(cclBuffer, opExpansionMode, rankNum));
    CHK_RET(hrtGetDevice(&deviceLogicId_));

    CHK_RET(InitHDCommunicate());

 	if (!hcclCommDfx_) {
        EXECEPTION_CATCH(hcclCommDfx_ = std::make_unique<HcclCommDfx>(), return HCCL_E_PTR);
 	}
 	CHK_RET(hcclCommDfx_->Init(deviceLogicId_, commId_, rankId_));
    CHK_RET(InitTaskExceptionHandler());

    CHK_RET(InitKfcAndRegisterCollComm());

    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult CollComm::ApplyUserCommConfig(HcclCommConfig *config, uint32_t &opExpansionMode)
{
    if (!config) {
        return HCCL_SUCCESS;
    }

    opExpansionMode = config->hcclOpExpansionMode;
    u32 tc = config->hcclRdmaTrafficClass;
    CHK_PRT_RET((tc != TC_DEFAULT) && (tc > TC_MAX || (tc % MULTIPLE != 0)),
        HCCL_ERROR("[InitCollComm]errNo[0x%016llx] invalid hcclRdmaTrafficClass[%u], must be 0xFFFFFFFF or in [0,255] and a multiple of 4",
            HCCL_ERROR_CODE(HCCL_E_PARA), tc),
        HCCL_E_PARA);
    CHK_RET(config_.SetConfigTrafficClass(tc));

    u32 sl = config->hcclRdmaServiceLevel;
    CHK_PRT_RET((sl != SL_DEFAULT) && (sl > SL_MAX),
        HCCL_ERROR("[InitCollComm]errNo[0x%016llx] invalid hcclRdmaServiceLevel[%u], must be 0xFFFFFFFF or in [0,7]",
            HCCL_ERROR_CODE(HCCL_E_PARA), sl),
        HCCL_E_PARA);
    CHK_RET(config_.SetConfigServiceLevel(sl));

    u32 qos = config->hcclQos;
    CHK_PRT_RET((qos != HCCL_COMM_QOS_CONFIG_NOT_SET) && (qos > SL_MAX),
        HCCL_ERROR("[InitCollComm]errNo[0x%016llx] invalid hcclQos[%u], must be 0xFFFFFFFF or in [0,7]",
            HCCL_ERROR_CODE(HCCL_E_PARA), qos),
        HCCL_E_PARA);
    CHK_RET(config_.SetConfigHcclQos(qos));
    return HCCL_SUCCESS;
}

HcclResult CollComm::InitKfcAndRegisterCollComm()
{
    myRank_->SetKfcControlTransfer(kfcControlTransferH2D_, kfcStatusTransferD2H_);
    CollCommMgr::GetInstance()->RegisteCollComm(this); 
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
        CHK_RET(kfcControlTransferH2D_->Put(0, sizeof(Hccl::KfcCommand), reinterpret_cast<uint8_t *>(&opCmd)));
        HCCL_RUN_INFO("[%s]group[%s] send Hccl::KfcCommand[%d] success", __func__, commId_.c_str(), opCmd);

        Hccl::KfcExecStatus opInfo;
        constexpr u32 WAIT_CMD_TIMEOUT = 10 * 1000; // 最大等待10秒
        auto timeout = std::chrono::milliseconds(WAIT_CMD_TIMEOUT);
        auto startTime = std::chrono::steady_clock::now();

        while (true) {
            CHK_RET(kfcStatusTransferD2H_->Get(0, sizeof(Hccl::KfcExecStatus), reinterpret_cast<uint8_t *>(&opInfo)));
            if (opInfo.kfcStatus == Hccl::KfcStatus::DESTROY_AICPU_COMM_DONE) {
                HCCL_RUN_INFO("[%s]get Hccl::KfcStatus[%d] success", __func__, opInfo.kfcStatus);
                return HCCL_SUCCESS;
            } else if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
                HCCL_ERROR("[%s]timeout, maxTime[%u ms] and get the opExecStatus is [%u].",
                    __func__, WAIT_CMD_TIMEOUT, opInfo.kfcStatus);
                return HCCL_E_TIMEOUT;
            }
            usleep(TEN_MILLISECOND_OF_USLEEP);
        }
    }
    return HCCL_SUCCESS;
}

uint32_t CollComm::GetMyRankId() const
{
    return rankId_;
}

HcclResult CollComm::InitHDCommunicate()
{
    // 初始化aicpu进程 host-device 共享内存
    EXECEPTION_CATCH((kfcControlTransferH2D_ = 
        std::make_shared<hccl::HDCommunicate>(deviceLogicId_, HCCL_HDC_TYPE_H2D, sizeof(Hccl::KfcCommand))),
        return HCCL_E_PTR);
    CHK_RET(kfcControlTransferH2D_->InitHost());

    EXECEPTION_CATCH((kfcStatusTransferD2H_ = 
        std::make_shared<hccl::HDCommunicate>(deviceLogicId_, HCCL_HDC_TYPE_D2H, sizeof(Hccl::KfcExecStatus))),
        return HCCL_E_PTR);
    CHK_RET(kfcStatusTransferD2H_->InitHost());

    return HCCL_SUCCESS;
}

HcclResult CollComm::GetHDCommunicate(
    HDCommunicateParams &kfcControlTransferH2DParams, HDCommunicateParams &kfcStatusTransferD2HParams)
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
    return commStatus_;
}

HcclResult CollComm::Suspend()
{
    HCCL_RUN_INFO("[CollComm][Suspend] commId[%s] start to suspend.", commId_.c_str());
    if (commStatus_ == HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING) {
        HCCL_WARNING("[CollComm][Suspend] The current communication has been suspended, no need to suspend again.");
        return HcclResult::HCCL_SUCCESS;
    }
    commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING;

    return myRank_->StopLaunch();
}

HcclResult CollComm::Clean()
{
    HCCL_RUN_INFO("[CollComm][Clean] commId[%s] start to clean.", commId_.c_str());
    if (commStatus_ != HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING) {
        HCCL_ERROR("[CollComm][Clean] The current communication is not suspended, cannot clean, status is [%u]", 
            static_cast<uint32_t>(commStatus_));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    if (isCleaned_) {
        HCCL_WARNING("[CollComm][Clean] The current communication has been cleaned, no need to clean again.");
        return HcclResult::HCCL_SUCCESS;
    }
    isCleaned_ = true;

    // 先清理Host
    return myRank_->Clean();
}

HcclResult CollComm::Resume()
{
    if (commStatus_ == HcclCommStatus::HCCL_COMM_STATUS_INVALID) {
        HCCL_ERROR("[CollComm][Resume] Comm has been error, can not resume now!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    if (commStatus_ != HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING) {
        HCCL_WARNING("[CollComm][Resume] The current communication is normal, no need to resume, status is [%u]",
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

    commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_READY;
    isCleaned_ = false;
    HCCL_INFO("[CollComm][Resume] Resume success.");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CollComm::InitTaskExceptionHandler()
{
    hcomm::TaskExceptionHost* handler = hcomm::TaskExceptionHostManager::GetHandler(static_cast<size_t>(deviceLogicId_));
    CHK_PTR_NULL(handler);
    CHK_RET(handler->Register());
    return HCCL_SUCCESS;
}

void CollComm::RegisterAicpuTaskExceptionCallback(u32 streamId)
{
    HCCL_INFO("[%s] start, commId[%s], streamId[%u]", __func__, commId_.c_str(), streamId);
    auto getAicpuTaskExceptionCallBack = [this]() {return this->GetAicpuTaskException();};
    hcomm::TaskExceptionHostManager::RegisterGetAicpuTaskExceptionCallBack(streamId, deviceLogicId_,
        getAicpuTaskExceptionCallBack);
    aicpuStreamIds_.insert(static_cast<s32>(streamId));
    return ;
}

Hccl::ErrorMessageReport CollComm::GetAicpuTaskException()
{
    Hccl::ErrorMessageReport errorMessage;
    CHK_PRT_RET(kfcStatusTransferD2H_ == nullptr, HCCL_ERROR("[%s]fail, d2h is nullptr", __func__), errorMessage);
    
    HcclResult ret = kfcStatusTransferD2H_->Get(sizeof(Hccl::KfcStatus) + sizeof(Hccl::KfcErrType),
       sizeof(errorMessage),reinterpret_cast<uint8_t *>(&errorMessage));
   
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]fail, group [%s], ret[%u]", __func__, commId_.c_str() ,ret), errorMessage);
    HCCL_INFO("[%s]group[%s] success", __func__, commId_.c_str());
   return errorMessage;
}

uint32_t CollComm::UpdateIndex()
{
    return index_ += 1;
}

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

}  // namespace hccl
