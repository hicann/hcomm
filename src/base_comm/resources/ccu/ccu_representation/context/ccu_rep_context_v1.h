/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu context header file
 * Create: 2025-02-18
 */

#ifndef CCU_REP_CTX_H
#define CCU_REP_CTX_H

#include <set>
#include <string>
#include <unordered_map>

#include "hcomm_primitives.h"
#include "ccu_rep_base_v1.h"
#include "ccu_rep_block_v1.h"

#include "ccu_common.h"
#include "task_param.h"

namespace hcomm {
constexpr uint16_t  CCU_MAX_CHANNEL_NUM     = 16;     // 最多16条link
constexpr uint16_t  INVALID_CKE_ID          = 0xFFFF; // CKE ID非法值
constexpr uint16_t  INVALID_VALUE_CHANNELID = 0xFFFF; // channel id非法值
constexpr uint64_t  INVALID_VALUE_NOTIFYID  = 0xFFFFFFFFFFFFFFFF; // NOTIFY id非法值

enum class CcuProfilinType { CCU_TASK_PROFILING, CCU_WAITCKE_PROFILING, CCU_LOOPGROUP_PROFILING, CCU_MAP_PROFILING };

using CcuProfilingInfo = Hccl::CcuProfilingInfo;
namespace CcuRep {

struct LoopGroupProfilingInfo {
 	     std::vector<CcuProfilingInfo> ccuProfilingInfos;
 	     std::unordered_map<std::shared_ptr<CcuRep::CcuRepBase>, uint32_t> loadRep2ArgIdxMap; // loadArg rep -> argIdx
 	     std::vector<std::shared_ptr<CcuRepBase>> assignProfilingReps;  // assign rep
 	     std::vector<std::shared_ptr<CcuRepBase>> lgProfilingReps;  // loopgroup rep
 	 };

class CcuRepContext {
public:
    explicit CcuRepContext();
    virtual ~CcuRepContext();

    // 平台层内部使用
    std::shared_ptr<CcuRep::CcuRepBlock> CurrentBlock();
    void                                 SetCurrentBlock(std::shared_ptr<CcuRep::CcuRepBlock> repBlock);
    virtual void                         Append(std::shared_ptr<CcuRep::CcuRepBase> rep);
    const std::vector<std::shared_ptr<CcuRep::CcuRepBase>> &GetRepSequence();
    std::shared_ptr<CcuRep::CcuRepBase> GetRepByInstrId(uint16_t instrId);
    void DumpReprestation();

    void     SetDieId(uint32_t dieId);
    uint32_t GetDieId() const;
    void     SetMissionId(uint32_t missionId);
    uint32_t GetMissionId() const;
    void     SetMissionKey(uint32_t missionKey);
    uint32_t GetMissionKey() const;

    // ccu profiling相关接口
    std::vector<CcuProfilingInfo> &GetProfilingInfo();
    CcuRep::LoopGroupProfilingInfo &GetLGProfilingInfo();
    const std::vector<std::shared_ptr<CcuRep::CcuRepBase>> &GetWaiteCkeProfilingReps() const;
    void CollectProfilingReps(std::shared_ptr<CcuRep::CcuRepBase> rep);

    void AddSqeProfiling(const std::string &kernelName);
    int32_t AddProfiling(const std::string &name, uint32_t mask);
    int32_t AddProfiling(const ChannelHandle channel, const std::string &name, uint32_t signalIndex, uint32_t mask);
    int32_t AddProfiling(const ChannelHandle *channels, uint32_t channelNum);
    int32_t AddProfiling(const ChannelHandle *channels, uint32_t channelNum, HcommDataType hcommDataType,
        HcommDataType hcommOutputDataType, HcommReduceOp hcommOpType);
public:
    // CCU Profiling相关数据
    CcuProfilingInfo ccuProfilingInfoCache;
    std::vector<std::shared_ptr<CcuRepBase>> allLgProfilingReps;  // 当前所有的loopGroup Rep
    LoopGroupProfilingInfo lgProfilingInfo; // LoopGroup相关profiling缓存信息
    std::vector<std::shared_ptr<CcuRepBase>> waitCkeProfilingReps; // waitCKE相关REP缓存
    std::vector<CcuProfilingInfo> profilingInfo; // context全部profiling缓存信息
protected:
    std::set<std::string> registeredLoop;

private:
    std::shared_ptr<CcuRep::CcuRepBlock> activeBlock{nullptr};
    std::shared_ptr<CcuRep::CcuRepBlock> mainBlock{nullptr};

    uint32_t             dieId{CCU_MAX_IODIE_NUM};
    uint32_t             missionId{UINT32_MAX};
    uint32_t             missionKey{0};
};

}; // namespace CcuRep
}; // namespace hcomm

#endif // _CCU_REP_CTX_H