/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef COLL_COMM_H
#define COLL_COMM_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "my_rank.h"
#include "rank_graph.h"
#include "comm_config_pub.h"
#include "comm_engine_res_manager.h"
#include "independent_op_context_manager.h"
#include "comm_mem_manager.h"
#include "channel_manager.h"
#include "hcclCommDfx.h"
#include "rank_graph_v2.h"
#include "error_message_v2.h"
#include "manager_common.h"
#include "include/hccl_communicator.h"
#include "hccl/hccl_res.h"
namespace Hccl {
class DevBuffer;
}  // namespace Hccl
namespace hccl {
class SymmetricMemory;
struct SymmetricMemoryResource;
struct SymmetricMemoryDeleter {
    void operator()(SymmetricMemory *ptr) const;
};
/**
 * @note 职责：集合通信通信域上下文管理，包括RankGraph和本rank信息资源等内容。
 * 当前需包含原有的91092/91093的通信域、原有的91095的通信域void
 * *指针、及新独立算子架构的通信域（支持91092/91093/91095...）。
 */
enum class CollCommInitMode {
    fullMode,    // 全功能模式：给A5及后续新架构使用，完整的CollComm初始化和资源管理
    simpleMode   // 简化模式：给A2/A3老芯片使用，由于架构限制，仅将RankGraph、MyRank等放入CollComm管理
};

class CollComm {
public:
    CollComm(void *comm, uint32_t rankId, const std::string &commName, const ManagerCallbacks& callbacks,
             CollCommInitMode initMode = CollCommInitMode::fullMode);
    ~CollComm();
    
    // 初始化通信域
    HcclResult Init(void *rankGraph, aclrtBinHandle binHandle, HcclMem cclBuffer, uint32_t opExpansionMode = 0);

    inline CommConfig& GetCommConfig() { return config_;}
    inline RankGraph* GetRankGraph() { return rankgraph_; }
    inline CommEngineResMgr* GetCommEngineResMgr() { return commEngineResMgr_.get(); }
    inline ContextManager* GetContextManager() { return contextMgr_.get(); }
    inline CommMemMgr* GetCommMemMgr() { return commMemMgr_.get(); }
    inline ChannelManager* GetChannelManager() { return channelMgr_.get(); }
    void *GetCommunicatorV2() { return comm_; }

    // 获取MyRank
    MyRank* GetMyRank() const { return myRank_.get(); }
    
    // 获取Rank ID
    uint32_t GetMyRankId() const;

    // 获取devicelogicId
    s32 GetDeviceLogicId() const { return deviceLogicId_; }

    // 获取Rank数量
    uint32_t GetRankSize() const {
        if (rankgraph_ == nullptr) {
            HCCL_ERROR("[CollComm]get ranksize failed");
            return 0;
        }
        uint32_t rankSize{0};
        HcclResult ret = rankgraph_->GetRankSize(&rankSize);
        if (ret != 0) {
            HCCL_ERROR("[CollComm]get ranksize failed");
            return 0;
        }
        return rankSize;
    }

    // 获取HcclCommDfx
    HcclCommDfx* GetHcclCommDfx() { return hcclCommDfx_.get(); }
    std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> GetDfxCallback() {
        if (hcclCommDfx_ == nullptr) {
            HCCL_ERROR("[CollComm]CollComm DfxCallBack failed. hcclCommDfx is nullptr");
            return nullptr;
        }
        return hcclCommDfx_->GetCallback();
    }
    const std::string& GetCommId() const {return commId_;}
    HcclResult GetHDCommunicate(
        HDCommunicateParams &kfcControlTransferH2DParams, HDCommunicateParams &kfcStatusTransferD2HParams);
    Hccl::ErrorMessageReport GetAicpuTaskException();
    HcclResult GetParentRankId(u32& parentRankId) const;
    uint32_t UpdateIndex();
    
    // Todo:在这里做N秒快恢
    HcclCommStatus GetCommStatus() const;
    HcclResult Suspend();
    HcclResult Clean();
    HcclResult Resume();
    HcclResult RegisterWindow(void* ptr, size_t size, HcclCommSymWindow *winHandle);
    HcclResult DeregisterWindow(HcclCommSymWindow winHandle);
    HcclResult GetCommSymWin(void* ptr, size_t size, HcclCommSymWindow *winHandle, size_t *offset);
    HcclResult RegisterPendingSymmetricMemHandles(std::vector<HcclMemHandle> &memHandles);
    HcclResult UpdateSymmetricRemoteMem(uint32_t remoteRank, const CommMem *remoteMems,
        const std::vector<std::string> &memTags);
    HcclResult GetHcclBinHandle(aclrtBinHandle &binHcclHandle);
    std::shared_ptr<class GroupScheduleMgr> groupScheduleMgr {nullptr}; //for group

private:
    HcclResult DestroyAicpuComm();
    HcclResult InitHDCommunicate();   
    HcclResult InitTaskExceptionHandler();
    HcclResult InitKfcAndRegisterCollComm();
    HcclResult GetRankIpPortMap();
    HcclResult InitSymmetricMemory();
    HcclResult RegisterSymmetricMemoryResource(void* ptr, size_t size, SymmetricMemoryResource &resource);
    void UnregisterSymmetricMemoryResource(const SymmetricMemoryResource &resource);

    /* 
     * CollComm初始化方式：
     *      fullMode：给A5及后续新架构使用，完整的CollComm初始化和资源管理
     *      SimpleMode：给A2/A3老芯片使用，由于架构限制，仅将RankGraph、MyRank等放入CollComm管理，简化CollComm实现
     */
    HcclResult InitFullMode(void* rankGraph, aclrtBinHandle binHandle, HcclMem cclBuffer, uint32_t opExpansionMode);
    HcclResult InitSimpleMode(void* rankGraph, aclrtBinHandle binHandle, HcclMem cclBuffer, uint32_t opExpansionMode);

    /* A2/A3：使用simpleMode兼容模式没有CommV2，使用简化版的CollComm代理rankgraph、myrank对象，其他功能暂不实现
     * A5&&下一代：使用fullMode全功能collComm模式
     */
    bool IsFullMode() const { return initMode_ == CollCommInitMode::fullMode; }
    HcclResult HcclBinaryUnLoad();

    void* comm_{nullptr};
    uint32_t rankId_{};
    std::string commId_;
    CommConfig config_{};
    HcclCommStatus commStatus_{HcclCommStatus::HCCL_COMM_STATUS_INVALID};
    
    ManagerCallbacks callbacks_; 
    s32 deviceLogicId_{0};
    uint32_t index_{0};

    // 持有CommunicatorImpl中cclBuffer的shared_ptr拷贝，延长其生命周期
    std::shared_ptr<Hccl::DevBuffer> cclBuffer_{nullptr};

    RankGraph* rankgraph_{nullptr};
    std::unique_ptr<CommEngineResMgr> commEngineResMgr_{nullptr};
    std::unique_ptr<ContextManager>  contextMgr_{nullptr};
    std::unique_ptr<CommMemMgr> commMemMgr_{nullptr};
    std::unique_ptr<ChannelManager> channelMgr_{nullptr};
    std::shared_ptr<MyRank> myRank_{};
    std::unique_ptr<HcclCommDfx> hcclCommDfx_{nullptr};
    uintptr_t   addr_{0};
    std::size_t size_{0};
    HcclMemType memType_{HcclMemType::HCCL_MEM_TYPE_DEVICE};

    // NS recover
    bool isCleaned_{false};

    std::shared_ptr<HDCommunicate> kfcControlTransferH2D_{nullptr};
    std::shared_ptr<HDCommunicate> kfcStatusTransferD2H_{nullptr};
    Hccl::RankIpPortMapPtr rankIpPortMap_;

    CollCommInitMode initMode_{CollCommInitMode::fullMode};  // 初始化模式
    std::unique_ptr<SymmetricMemory, SymmetricMemoryDeleter> symmetricMemory_{nullptr};
    aclrtBinHandle binHcclHandle_{nullptr};
    std::mutex binHcclmutex_;
};
}  // namespace hccl

#endif  // COLL_COMM_H
