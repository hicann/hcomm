/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_comm_aicpu.h"
#include "aicpu_operator_pub.h"
#include "adapter_hal_pub.h"
#include "aicpu_ts_thread.h"
#include "aicpu_res_package_helper.h"
#include "ub_transport_lite_impl.h"
#include "notify_manager.h"
#include "aicpu_hccl_def.h"
#include "ns_recovery/aicpu/ns_recovery_func_lite.h"
#include "dlhal_function_v2.h"
#include "profiling_command_handle_lite.h"
#include "aicpu_daemon_service.h"
#include "hcclCommTaskExceptionLite.h"
#include "coll_comm_aicpu_destroy_func.h"
#include "aicpu_indop_env.h"

constexpr u32 NOTIFY_SIZE_EIGHT = 8;

HcclResult CollCommAicpu::InitAicpuIndOp(CommAicpuParam *commAicpuParam)
{
    if (commStatus_ == HcclCommStatus::HCCL_COMM_STATUS_READY) {
        HCCL_RUN_INFO("[CollCommAicpu][%s]Group[%s] already initialized, skip reinit", __func__,
            identifier_.c_str());
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(commAicpuParam);
    topoInfo_.deviceLogicId = commAicpuParam->deviceLogicId;
    topoInfo_.devicePhyId = commAicpuParam->devicePhyId;
    topoInfo_.deviceType = static_cast<DevType>(commAicpuParam->deviceType);
    identifier_ = std::string(commAicpuParam->hcomId);
    topoInfo_.userRankSize = commAicpuParam->userRankSize;
    topoInfo_.userRank = commAicpuParam->userRank; 
    notifys_.reserve(hccl::HCCL_THREAD_NOTIFY_MAX_NUM);

    CHK_RET(hrtSetWorkModeAicpu(true));
    CHK_RET(hrtSetlocalDevice(topoInfo_.deviceLogicId));
    CHK_RET(hrtSetlocalDeviceType(topoInfo_.deviceType));
    CHK_RET(hrtDrvGetLocalDevIDByHostDevID(topoInfo_.devicePhyId, &devId_));
    CHK_RET(dfx_.Init(devId_, identifier_));
    CHK_RET(RegisterProfCallBack());
    CHK_RET(InitHDCommunicate(commAicpuParam));

    EXECEPTION_CATCH(nsRecoveryLitePtr_ = std::make_shared<NsRecoveryLite>(), return HCCL_E_PTR);
    nsRecoveryLitePtr_->Init(kfcControlTransferH2D_, kfcStatusTransferD2H_);

    CHK_RET(Hccl::DlHalFunctionV2::GetInstance().DlHalFunctionInit());

    commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_READY;

    static std::once_flag initBackGround;
    std::call_once(initBackGround, [this]() { this->InitBackGroundThread();} );

    static std::once_flag initEnv;
    std::call_once(initEnv, [this, commAicpuParam]() { this->InitIndopEnv(commAicpuParam);} );
    HCCL_RUN_INFO("[%s]success, group[%s], deviceLogicId[%u], devicePhyId[%u], deviceType[%u], rankSize[%u] "\
        "userRank[%u], devId[%u]", __func__, identifier_.c_str(), topoInfo_.deviceLogicId, topoInfo_.devicePhyId,
        topoInfo_.deviceType, topoInfo_.userRankSize, topoInfo_.userRank, devId_);
    return HCCL_SUCCESS;
}

HcclResult CollCommAicpu::InitHDCommunicate(CommAicpuParam *commAicpuParam)
{
    if (commAicpuParam->kfcControlTransferH2DParams.buffLen != 0 && kfcControlTransferH2D_ == nullptr) {
        EXECEPTION_CATCH((kfcControlTransferH2D_ = std::make_shared<hccl::HDCommunicate>()), return HCCL_E_PTR);
        CHK_SMART_PTR_NULL(kfcControlTransferH2D_);
        CHK_RET(kfcControlTransferH2D_->InitDevice(commAicpuParam->kfcControlTransferH2DParams));
    }
    if (commAicpuParam->kfcStatusTransferD2HParams.buffLen != 0 && kfcStatusTransferD2H_ == nullptr) {
        EXECEPTION_CATCH((kfcStatusTransferD2H_ = std::make_shared<hccl::HDCommunicate>()), return HCCL_E_PTR);
        CHK_SMART_PTR_NULL(kfcStatusTransferD2H_);
        CHK_RET(kfcStatusTransferD2H_->InitDevice(commAicpuParam->kfcStatusTransferD2HParams));
    }
    return HCCL_SUCCESS;
}

void CollCommAicpu::InitIndopEnv(CommAicpuParam *commAicpuParam)
{
    hcomm::SetTaskExceptionEnable(commAicpuParam->envConfig.taskExceptionEnable);
    hcomm::SetNotifyWaitTimeout(commAicpuParam->envConfig.notifyWaitTimeout);
    HCCL_RUN_INFO("[%s]Env: taskExceptionEnable[%d], notifyWaitTimeout[%u]",
        __func__, commAicpuParam->envConfig.taskExceptionEnable, commAicpuParam->envConfig.notifyWaitTimeout);
}

void CollCommAicpu::SetCommmStatus(HcclCommStatus status)
{
    HCCL_INFO("[%s]group[%s], flag[%d]", __func__, identifier_.c_str(), static_cast<int>(status));
    commStatus_ = status;
}

HcclResult CollCommAicpu::InitThreads(ThreadMgrAicpuParam *param)
{
    u32 threadNum = param->threadNum;
    std::vector<std::shared_ptr<Thread>> outThreads;
    outThreads.reserve(threadNum);
    std::string hcomId(param->hcomId);
    for (u32 i = 0; i < threadNum; ++i) {
        std::string thdUniqueId(param->threadParam[i], THREAD_UNIQUE_ID_MAX_SIZE);
        if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
            std::ostringstream oss;
            oss << "threadParam[" << i << "] raw bytes: ";
            for (u32 j = 0; j < THREAD_UNIQUE_ID_MAX_SIZE; ++j) {
                oss << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<unsigned int>(static_cast<unsigned char>(param->threadParam[i][j])) << " ";
            }
            HCCL_INFO("[CollCommAicpu][%s] %s", __func__, oss.str().c_str());
        }
        std::shared_ptr<AicpuTsThread> thread;
        EXECEPTION_CATCH((thread = std::make_shared<AicpuTsThread>(thdUniqueId)), return HCCL_E_PTR);
        HcclResult ret = thread->Init();
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[CollCommAicpu][%s] comm identifier[%s], init threads num[%u] failed at index %u",
                __func__, hcomId.c_str(), param->threadNum, i);
            return ret;
        }
        outThreads.emplace_back(thread);
    }

    ThreadHandle *threadArray = static_cast<ThreadHandle*>(param->deviceHandle);
    // 空指针校验
    CHK_PTR_NULL(threadArray);
    for (size_t i = 0; i < outThreads.size(); ++i) {
        threadArray[i] = reinterpret_cast<ThreadHandle>(outThreads[i].get());  // 拷贝裸指针
        HCCL_INFO("[CollCommAicpu][%s] threadArray[%u] = [%lu]", __func__, i, threadArray[i]);
        CHK_RET(RegisterThreadAddDfxTaskInfo(threadArray[i]));
    }
    ReadWriteLock rwLock(threadMutex_);
    rwLock.writeLock();
    threads_.insert(threads_.end(), std::make_move_iterator(outThreads.begin()),
        std::make_move_iterator(outThreads.end()));
    rwLock.writeUnlock();
    HCCL_INFO("[CollCommAicpu][%s] comm identifier[%s], init threads num[%u] success",
        __func__, hcomId.c_str(), threadNum);
    return HCCL_SUCCESS;
}

HcclResult CollCommAicpu::RegisterThreadAddDfxTaskInfo(ThreadHandle thread) 
{
    int32_t ret = HcommThreadRegisterDfx(thread, dfx_.GetCallback());
    if (ret != 0) {
        HCCL_ERROR("[CollCommAicpu][RegisterThreadAddDfxTaskInfo] HcommThreadRegisterDfx failed, ret[%d]", ret);
        return HCCL_E_PTR;
    }
 	return HCCL_SUCCESS;
} 

HcclResult CollCommAicpu::AllocChannelResource(HcclChannelUrmaRes *commParam)
{
    HCCL_INFO("[CollCommAicpu][%s] deviceLogicId[%d], devicePhyId[%u], deviceType[%d], commParam->channelList[%p], "
              "commParam->listNum[%u], commParam->uniqueIdAddr[%p], commParam->uniqueIdSize[%u]",
              __func__, topoInfo_.deviceLogicId, topoInfo_.devicePhyId, topoInfo_.deviceType, commParam->channelList,
              commParam->listNum, commParam->uniqueIdAddr, commParam->uniqueIdSize);
    CHK_RET(InitUrmaChannel(commParam));
    return HCCL_SUCCESS;
}

HcclResult CollCommAicpu::ProcessUrmaRes(HcclChannelUrmaRes *commParam, bool isInit)
{
    HCCL_INFO("[CollCommAicpu][%s] commParam->uniqueIdAddr[%p], commParam->uniqueIdSize[%u]",
        __func__, commParam->uniqueIdAddr, commParam->uniqueIdSize);
    ChannelHandle* channelList = reinterpret_cast<ChannelHandle*>(commParam->channelList);
    u8* currentSrcAddr = reinterpret_cast<u8*>(commParam->uniqueIdAddr);
    u32* addSize = reinterpret_cast<u32*>(commParam->channelSizeAddr);
    CHK_PTR_NULL(channelList);
    CHK_PTR_NULL(currentSrcAddr);
    CHK_PTR_NULL(addSize);

    for (u32 index = 0; index < commParam->listNum; index++) {
        std::vector<char> data(*addSize);

        // 计算地址块的偏移
        CHK_SAFETY_FUNC_RET(memcpy_s(data.data(), data.size(), currentSrcAddr, *addSize));
        currentSrcAddr += *addSize;
        addSize++;
        // 反序列化得到device侧transport对象
        Hccl::AicpuResPackageHelper helper;
        auto dataVec = helper.ParsePackedData(data);

        Hccl::AicpuResMgrType resType = Hccl::AicpuResMgrType::STREAM; // 待修改
        if (static_cast<u32>(resType) >= dataVec.size()) {
            HCCL_ERROR("[CollCommAicpu][%s] fail, resType[%d], dataVec size[%u]", __func__, resType, dataVec.size());
            return HCCL_E_PARA;
        }

        ChannelHandle channelHandle{0};
        if (isInit) {
            CHK_RET(ParsePackData(dataVec[resType].data, channelHandle));
            // 恢复出的channelHandle回填到commParam中
            channelList[index] = channelHandle;
            CHK_RET(RegisterChannelAddDfxTaskInfo(channelHandle));
            dfx_.AddChannelRemoteRankId(channelHandle, commParam->remoteRankList[index]);
        } else {
            channelHandle = channelList[index];
            if (!ubTransportMap_.count(channelHandle)) {
                HCCL_ERROR("[CollCommAicpu][%s] fail, resType[%d], current ChannelHandle nullptr", __func__, resType);
                return HCCL_E_PARA;
            }
            CHK_RET(ResumePackData(dataVec[resType].data, channelHandle));
        }
        
        // 打印
        HCCL_INFO("[CollCommAicpu][%s] index[%u], currentSrcAddr[%p], channelSizeAddr[%p], channelHandle[0x%llx]",
            __func__, index, currentSrcAddr, commParam->channelSizeAddr, channelHandle);
    }

    return HCCL_SUCCESS;
}

HcclResult CollCommAicpu::InitUrmaChannel(HcclChannelUrmaRes *commParam)
{
    return ProcessUrmaRes(commParam, true);
}

HcclResult CollCommAicpu::ParsePackData(std::vector<char> &data, ChannelHandle &handle)
{
    HCCL_DEBUG("[CollCommAicpu][%s] data: ptr[%p], size[%u]", __func__, data.data(), data.size());
    Hccl::BinaryStream binaryStream(data);

    std::vector<char> transpUniqueId;
    binaryStream >> transpUniqueId;

    Hccl::BinaryStream binaryStreamForType(transpUniqueId);
    u32 transType;
    binaryStreamForType >> transType;
    HCCL_INFO("[CollCommAicpu][ParsePackData] transType[%u]", transType);
    if (transType == Hccl::TransportType::UB) {
        std::unique_ptr<Hccl::UbTransportLiteImpl> ubTransportLiteImpl;
        EXECEPTION_CATCH((ubTransportLiteImpl = std::make_unique<Hccl::UbTransportLiteImpl>(transpUniqueId)),
            return HCCL_E_PTR);
        CHK_SMART_PTR_NULL(ubTransportLiteImpl);
        ubTransportLiteImpl->SetTaskExceptionEnable(hcomm::GetTaskExceptionEnable());
        handle = reinterpret_cast<uint64_t>(ubTransportLiteImpl.get());
        ubTransportMap_.insert({handle, std::move(ubTransportLiteImpl)});
    } else if (transType == Hccl::TransportType::P2P) {
        std::unique_ptr<Hccl::P2PTransportLiteImpl> p2pTransportLiteImpl;
        EXECEPTION_CATCH((p2pTransportLiteImpl = std::make_unique<Hccl::P2PTransportLiteImpl>(transpUniqueId)),
            return HCCL_E_PTR);
        CHK_SMART_PTR_NULL(p2pTransportLiteImpl);
        handle = reinterpret_cast<uint64_t>(p2pTransportLiteImpl.get());
        p2pTransportMap_.insert({handle, std::move(p2pTransportLiteImpl)});
    } else {
        HCCL_ERROR("[CollCommAicpu][ParsePackData] unsupported transportType[%u]", transType);
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult CollCommAicpu::RegisterChannelAddDfxTaskInfo(ChannelHandle channel) {
    int hert = HcommChannelRegisterDfx(channel, dfx_.GetCallback());
    return static_cast<HcclResult>(hert);
}

HcclResult CollCommAicpu::NotifyFree(NotifyMgrAicpuParam *param)
{
    u32 notifyNum = param->notifyNum;
    NotifyHandle *notifyArray = static_cast<NotifyHandle*>(param->deviceHandle);
    std::string hcomId(param->hcomId);
    // 空指针校验
    CHK_PTR_NULL(notifyArray);
    for (size_t i = 0; i < notifyNum; ++i) {
        LocalNotify* notify = reinterpret_cast<LocalNotify*>(notifyArray[i]);
        HCCL_INFO("[CollCommAicpu][%s] notifyArray[%u]=[%lu]", __func__, i, notifyArray[i]);
        auto it = std::find_if(notifys_.begin(), notifys_.end(),
            [notify](const std::unique_ptr<LocalNotify>& ptr) {
            return ptr.get() == notify;
        });
        if (it != notifys_.end()) {
            HCCL_INFO("[CollCommAicpu][%s] comm identifier[%s], free notifys[%u] success",
                __func__, hcomId.c_str(), notifyArray[i]);
            notifys_.erase(it);
        } else {
            HCCL_RUN_WARNING("[CollCommAicpu][%s] localNotify[%u] not found in notifys_", __func__, i);
        }
    }

    HCCL_INFO("[CollCommAicpu][%s] comm identifier[%s], free notifys num[%u] success",
            __func__, hcomId.c_str(), notifyNum);
    return HCCL_SUCCESS;
}

HcclResult CollCommAicpu::NotifyAlloc(NotifyMgrAicpuParam *param)
{
    u32 notifyNum = param->notifyNum;
    std::string notifysStr = std::string(param->notifyParam, NOTIFY_UNIQUE_ID_MAX_SIZE);
    std::string hcomId(param->hcomId);
    size_t notifySize = notifys_.size();
    HCCL_INFO("[CollCommAicpu][%s] comm identifier[%s], alloc notifys num[%u] begin, before notifySize[%u]",
        __func__, hcomId.c_str(), notifyNum, notifySize);
    if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
        std::ostringstream oss;
        oss << "notifyParam" << " raw bytes: ";
        for (u32 i = 0; i < NOTIFY_UNIQUE_ID_MAX_SIZE; ++i) {
            oss << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<unsigned int>(static_cast<unsigned char>(param->notifyParam[i])) << " ";
        }
        HCCL_INFO("[CollCommAicpu][%s] %s", __func__, oss.str().c_str());
    }
    HcclResult ret = NotifyManager::ParseBinNotifys(notifysStr, notifys_);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CollCommAicpu][%s] comm identifier[%s], alloc notifys num[%u] failed %u",
            __func__, hcomId.c_str(), notifyNum, ret);
        return ret;
    }
    HCCL_INFO("[CollCommAicpu][%s] comm identifier[%s], alloc notifys num[%u] end, after notifySize[%u]",
        __func__, hcomId.c_str(), notifyNum, notifys_.size());
    NotifyHandle *notifyArray = static_cast<NotifyHandle*>(param->deviceHandle);
    CHK_PTR_NULL(notifyArray);
    // 空指针校验
    for (size_t i = 0; i < notifyNum; ++i) {
        notifyArray[i] = reinterpret_cast<NotifyHandle>(notifys_[i + notifySize].get());  // 拷贝裸指针
        HCCL_INFO("[CollCommAicpu][%s] notifyArray[%u] = [%lu]", __func__, i + notifySize, notifyArray[i]);
    }

    HCCL_INFO("[CollCommAicpu][%s] comm identifier[%s], alloc notifys num[%u] success",
        __func__, hcomId.c_str(), notifyNum);
    return HCCL_SUCCESS;
}

hccl::NsRecoveryLitePtr CollCommAicpu::GetNsRecoveryLitePtr()
{
    return nsRecoveryLitePtr_;
}

HcclResult CollCommAicpu::Clean()
{
    for (auto& transPort : ubTransportMap_) {
        CHK_RET(transPort.second->Clean());
    }
    HCCL_INFO("CollCommAicpu::Clean() finished");
    
    return HCCL_SUCCESS;
}

HcclResult CollCommAicpu::ResumePackData(std::vector<char> &data, ChannelHandle &handle)
{
    Hccl::BinaryStream binaryStream(data);
    std::vector<char> transpUniqueId;
    binaryStream >> transpUniqueId;

    auto& transPortPtr = ubTransportMap_[handle];
    CHK_RET(transPortPtr->Resume(transpUniqueId));
    return HCCL_SUCCESS;
}

HcclResult CollCommAicpu::Resume(HcclChannelUrmaRes *commParam)
{
    CHK_PTR_NULL(commParam);
    CHK_RET(ProcessUrmaRes(commParam, false));
    nsRecoveryLitePtr_->SetNeedClean(false);
    nsRecoveryLitePtr_->ResetErrorReported();

    commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_READY;
    
    return HCCL_SUCCESS;
}

void CollCommAicpu::InitBackGroundThread()
{
    static auto commandToBackGroud = Hccl::CommandToBackGroud::Default;
    static auto daemonServiceRun = [](void *info) {
        Hccl::AicpuDaemonService::GetInstance().ServiceRun(info);
    };
    static auto daemonServiceStop = [](void *info) {
        Hccl::AicpuDaemonService::GetInstance().ServiceStop(info);
    };

    // 注册守护进程函数
    hcomm::HcclCommTaskExceptionLite::GetInstance().Init(devId_);
    Hccl::AicpuDaemonService::GetInstance().Register(&hcomm::HcclCommTaskExceptionLite::GetInstance());
    Hccl::AicpuDaemonService::GetInstance().Register(&hccl::CollCommAicpuDestroyFunc::GetInstance());
    Hccl::AicpuDaemonService::GetInstance().Register(&NsRecoveryFuncLite::GetInstance());

    // 启动背景线程
    if (Hccl::StartMC2MaintenanceThread != nullptr) {
        Hccl::StartMC2MaintenanceThread(daemonServiceRun, &commandToBackGroud, daemonServiceStop, &commandToBackGroud);
        HCCL_RUN_INFO("[%s]start BackGround thread success.", __func__);
    } else {
        HCCL_WARNING("[%s]StartMC2MaintenanceThread func is nullptr", __func__);
    }
}

HcclResult CollCommAicpu::BackGroundGetCmd(Hccl::KfcCommand &cmd)
{
    CHK_SMART_PTR_NULL(kfcControlTransferH2D_);
    HcclResult ret = kfcControlTransferH2D_->Get(0, sizeof(Hccl::KfcCommand), reinterpret_cast<uint8_t *>(&cmd));
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s]fail, group[%s]", __func__, identifier_.c_str()), ret);
    return HCCL_SUCCESS;
}

HcclResult CollCommAicpu::BackGroundSetStatus(Hccl::KfcStatus state)
{
    Hccl::KfcExecStatus status;
    status.kfcStatus = state;
    HCCL_INFO("[%s]group[%s], state[%u]", __func__, identifier_.c_str(), state);
    HcclResult ret = kfcStatusTransferD2H_->Put(0, sizeof(status.kfcStatus), reinterpret_cast<uint8_t *>(&status));
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s]fail, group[%s]", __func__, identifier_.c_str()), ret);
    return HCCL_SUCCESS;
}

HcclResult CollCommAicpu::SendErrorMessageReportToHost(Hccl::ErrorMessageReport& errMsgInfo)
{
    CHK_SMART_PTR_NULL(kfcStatusTransferD2H_);
    CHK_RET(kfcStatusTransferD2H_->Put(sizeof(Hccl::KfcStatus) + sizeof(Hccl::KfcErrType), sizeof(errMsgInfo),
        reinterpret_cast<uint8_t *>(&errMsgInfo)));
    return HCCL_SUCCESS;
}

HcclResult CollCommAicpu::RegisterProfCallBack()
{
    if (MsprofRegisterCallback != nullptr) {
        HCCL_INFO("RegisterProfCallBack not null");
        int32_t ret = MsprofRegisterCallback(AICPU, &Hccl::DeviceCommandHandle);
        CHK_PRT_RET((ret != 0), HCCL_ERROR("[%s] failed. ret = [%d]", __func__, ret), HCCL_E_PARA);
    } else {
        HCCL_INFO("RegisterProfCallBack is null");
    }
    return HCCL_SUCCESS;
}

u32 CollCommAicpu::UpdateIndex()
{
    return index_+=1;
}
