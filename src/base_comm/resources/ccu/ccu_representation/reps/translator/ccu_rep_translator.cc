/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu rep translator implement file
 * Create: 2025-02-20
 */

#include "ccu_rep_translator_v1.h"

#include <algorithm>

#include "exception_util.h"
#include "ccu_api_exception.h"
#include "ccu_rep_loopcall_v1.h"
#include "ccu_rep_funccall_v1.h"
#include "ccu_rep_type_v1.h"
#include "ccu_rep_loop_v1.h"
#include "ccu_rep_loadarg_v1.h"
#include "ccu_assist_v1.h"

#include "ccu_dev_mgr_imp.h"

namespace hcomm {
namespace CcuRep {

template <typename T> bool CheckType(const std::shared_ptr<CcuRepBlock> &refer)
{
    if (refer == nullptr) {
        HCCL_ERROR("input refer is nullptr");
        return false;
    }
    HCCL_INFO("[ChechType] refer->Type() = %d", refer->Type());
    return false;
}

template <> bool CheckType<CcuRepFuncBlock>(const std::shared_ptr<CcuRepBlock> &refer)
{
    if (refer == nullptr) {
        HCCL_ERROR("input refer is nullptr");
        return false;
    }
    return refer->Type() == CcuRepType::FUNC_BLOCK ? true : false;
}

template <> bool CheckType<CcuRepLoopBlock>(const std::shared_ptr<CcuRepBlock> &refer)
{
    if (refer == nullptr) {
        HCCL_ERROR("input refer is nullptr");
        return false;
    }
    return refer->Type() == CcuRepType::LOOP_BLOCK ? true : false;
}

template <typename T1, typename T2> void CcuRepTranslator::BuildReference(const std::shared_ptr<CcuRepBase> &rep)
{
    auto caller = std::static_pointer_cast<T1>(rep);
    auto label  = caller->GetLabel();
    // 特例：针对函数地址调用，不需要依靠函数名索引
    if (label == "") {
        return;
    }
    auto refer = refManager->GetRefBlock(label);
    if (CheckType<T2>(refer)) {
        caller->Reference(std::static_pointer_cast<T2>(refer));
    } else {
        Hccl::THROW<Hccl::CcuApiException>("Invalid Reference: %s", label.c_str());
    }
}

CcuRepTranslator::CcuRepTranslator(int32_t deviceLogicId, uint8_t dieId,
                                   std::shared_ptr<CcuRepReferenceManager> refManager,
                                   std::array<uint16_t, CCU_MAX_IODIE_NUM>& reserverChannalId,
                                   std::pair<uint64_t, uint64_t>& ccuTokenInfo, uint64_t hbmTokenInfo)
    : refManager(refManager)
{
    transDep.logicalId = deviceLogicId;
    transDep.dieId     = dieId;
    s32 result = memcpy_s(transDep.reserveChannalId, sizeof(transDep.reserveChannalId), reserverChannalId.data(), sizeof(reserverChannalId));
    if (result != 0) {
        Hccl::THROW<Hccl::InternalException>(Hccl::StringFormat("[NsRecovery] CcuRepTranslator::CcuRepTranslator: memcpy_s failed, ret = %d", result));
    }
    // 获取xn起始地址
    HcclResult ret = CcuDevMgrImp::GetXnBaseAddr(deviceLogicId, dieId, transDep.xnBaseAddr);
    if (ret != HcclResult::HCCL_SUCCESS) {
        Hccl::THROW<Hccl::CcuApiException>("Failed to get xn base address. deviceLogicId = %d, dieId = %u, ret = %d",
                               deviceLogicId, dieId, ret);
    }
    transDep.ccuResSpaceTokenInfo = CcuRep::GetToken(ccuTokenInfo.first, ccuTokenInfo.second, 1);
    // 获取hbm token信息
    transDep.memTokenInfo = hbmTokenInfo;
}

CcuRepTranslator::CcuRepTranslator(std::shared_ptr<CcuRepReferenceManager> refManager, const TransDep &transDep)
    : refManager(refManager), transDep(transDep)
{
}

uint32_t CcuRepTranslator::GetInstrNum()
{
    return 4; // 4:翻译器翻译过程中额外需要的指令空间大小(插入3条通用操作指令+1条终止指令)
}

CcuResReq CcuRepTranslator::GetResReq(uint8_t dieId)
{
    // xn 资源统一从 continuousXn 池子申请，离散 xn 帐户已废弃
    CcuResReq resReq;
    resReq.continuousXnReq[dieId] = XN_NUM; // 4个Xn资源
    resReq.gsaReq[dieId]          = GSA_NUM; // 3个GSA资源
    resReq.ckeReq[dieId]          = CKE_NUM; // 2个CKE资源
    return resReq;
}

void CcuRepTranslator::GetRes(CcuRepResource &res)
{
    for (int i = 0; i < XN_NUM; i++) {
        res.continuousVariable[transDep.dieId].push_back(var[i]);
    }
    for (int i = 0; i < GSA_NUM; i++) {
        res.address[transDep.dieId].push_back(addr[i]);
    }
    for (int i = 0; i < CKE_NUM; i++) {
        res.localNotify[transDep.dieId].push_back(signal[i]);
    }
}

void CcuRepTranslator::PreProcess(std::shared_ptr<CcuRepBase> rep)
{
    auto repType = rep->Type();
    if (repType == CcuRepType::FUNC_BLOCK) {
        auto funcBlock = std::static_pointer_cast<CcuRepFuncBlock>(rep);
        refManager->SetRefBlock(funcBlock->GetLabel(), funcBlock);
        funcBlock->SetFuncManager(refManager.get());
    } else if (repType == CcuRepType::LOOP_BLOCK) {
        auto loopBlock = std::static_pointer_cast<CcuRepLoopBlock>(rep);
        refManager->SetRefBlock(loopBlock->GetLabel(), loopBlock);
    } else if (repType == CcuRepType::FUNC_CALL) {
        BuildReference<CcuRepFuncCall, CcuRepFuncBlock>(rep);
        auto funcCall = std::static_pointer_cast<CcuRepFuncCall>(rep);
        funcCall->SetFuncManager(refManager.get());
    } else if (repType == CcuRepType::LOOP_CALL) {
        BuildReference<CcuRepLoopCall, CcuRepLoopBlock>(rep);
    } else if (repType == CcuRepType::LOOP) {
        BuildReference<CcuRepLoop, CcuRepLoopBlock>(rep);
    }
}

void CcuRepTranslator::Translate(const std::vector<std::shared_ptr<CcuRepBase>> &repVec, CcuInstr *&instr,
                                 uint16_t &instrId, std::function<bool(std::shared_ptr<CcuRepBase>)> filter)
{
    constexpr uint32_t maxTryCount = 10; // 最大尝试次数10
    uint32_t           tryCount    = 0;
    uint32_t           restCount   = 0;

    auto funcInVar = refManager.get()->GetFuncIn();
    int funcArgIndex = 0;

    do {
        restCount = 0;
        for (uint32_t index = 0; index < repVec.size(); index++) {
            if (!filter(repVec[index])) {
                continue;
            }

            if (repVec[index]->Translated()) {
                continue;
            }

            if (repVec[index]->Type() == CcuRepType::LOAD_ARG && transDep.isFuncBlock) {
                transDep.loadXnId = funcInVar[funcArgIndex++].Id();
            }

            PreProcess(repVec[index]);
            bool flag = repVec[index]->Translate(instr, instrId, transDep);
            if (!flag) {
                restCount++;
            }
        }
        tryCount++;
        HCCL_INFO("tryCount = %u, remaining representation = %u", tryCount, restCount);
    } while (restCount > 0 && tryCount < maxTryCount);

    if (tryCount == maxTryCount && restCount > 0) {
        HCCL_ERROR("After translation, remaining representation: tryCount = %u, restCount = %u ", tryCount, restCount);
        for (uint32_t index = 0; index < repVec.size(); index++) {
            if (!repVec[index]->Translated()) {
                HCCL_ERROR("index[%u], %s", index, repVec[index]->Describe().c_str());
            }
        }
        Hccl::THROW<Hccl::CcuApiException>("Translation Failed");
    }
}

CcuInstrInfo CcuRepTranslator::Translate(const std::vector<std::shared_ptr<CcuRepBase>> &repVec, uint16_t startInstrId, bool isFuncBlock)
{
    constexpr uint32_t defaultInstrCapacity = 32 * 1024; // 默认最大容量32 * 1024条
    CcuInstrInfo       instrInfo;
    instrInfo.instrVec.resize(defaultInstrCapacity);
    CcuInstr *instr      = instrInfo.instrVec.data();
    uint16_t  curInstrId = startInstrId;

    BindResource(isFuncBlock);

    // 翻译LoopBlock
    Translate(repVec, instr, curInstrId, [](std::shared_ptr<CcuRepBase> rep) -> bool {
        return rep->Type() == CcuRepType::LOOP_BLOCK;
    });

    // 翻译funcBlock
    Translate(repVec, instr, curInstrId, [](std::shared_ptr<CcuRepBase> rep) -> bool {
        return rep->Type() == CcuRepType::FUNC_BLOCK;
    });

    uint16_t missionStartInstrId = curInstrId;

    // 翻译Load:按全局 argId 升序排序后再翻译,确保 LoadSqeArgs 指令在 mission 切分时
    // 落入与其 slot id 匹配的 mission(避免用户乱序 LoadArg 导致取参错位)
    std::vector<std::shared_ptr<CcuRepBase>> sortedLoadArgReps;
    sortedLoadArgReps.reserve(repVec.size());
    for (const auto &rep : repVec) {
        if (rep->Type() == CcuRepType::LOAD_ARG) {
            sortedLoadArgReps.push_back(rep);
        }
    }
    std::stable_sort(sortedLoadArgReps.begin(), sortedLoadArgReps.end(),
        [](const std::shared_ptr<CcuRepBase> &a, const std::shared_ptr<CcuRepBase> &b) {
            return std::static_pointer_cast<CcuRepLoadArg>(a)->GetFullArgId()
                 < std::static_pointer_cast<CcuRepLoadArg>(b)->GetFullArgId();
        });
    Translate(sortedLoadArgReps, instr, curInstrId,
        [](std::shared_ptr<CcuRepBase> rep) -> bool { return rep->Type() == CcuRepType::LOAD_ARG; });

    // 插入通用操作
    CommonProcess(instr, curInstrId);

    // 翻译主体
    Translate(repVec, instr, curInstrId, [](std::shared_ptr<CcuRepBase> rep) -> bool {
        return true;
    });

    FinishMainBlock(instr, curInstrId);

    instrInfo.startInstrId        = startInstrId;
    instrInfo.instrCount          = curInstrId - startInstrId;
    instrInfo.missionStartInstrId = missionStartInstrId;
    instrInfo.missionInstrCount   = curInstrId - missionStartInstrId;
    instrInfo.instrVec.resize(instrInfo.instrCount);

    DumpInstruction(instrInfo);
    DumpRep(repVec, instrInfo);

    return instrInfo;
}

void CcuRepTranslator::CommonProcess(CcuInstr *&instr, uint16_t &instrId)
{
    LoadImdToXnInstr(instr++, var[0].Id(), 0);
    LoadImdToGSAInstr(instr++, addr[0].Id(), 0);
    SetCKEInstr(instr++, signal[0].Id(), 0xffff, 0, 0, 1);
    u32 instrNum = 3;
    if (instrId > UINT16_MAX - instrNum) {
        Hccl::THROW<Hccl::InternalException>("integer overflow occurs");
    }
    instrId += instrNum;  // 插入3条指令
}

void CcuRepTranslator::FinishMainBlock(CcuInstr *&instr, uint16_t &instrId)
{
    if (transDep.isFuncBlock) {
        JumpInstr(instr++, refManager.get()->GetFuncRet(FUNC_NEST_MAX).Id(), transDep.reserveXnId, 1);
    } else {
        LoadImdToXnInstr(instr++, var[0].Id(), 0);
    }
    if (instrId > UINT16_MAX - 1) {
        Hccl::THROW<Hccl::InternalException>("integer overflow occurs");
    }
    instrId++;
}

void CcuRepTranslator::DumpInstruction(const CcuInstrInfo &instrInfo) const
{
    HCCL_INFO("CcuInstrInfo: startInstrId = %u, instrCount = %u, missionStartInstrId = %u, missionInstrCount = %u",
              instrInfo.startInstrId, instrInfo.instrCount, instrInfo.missionStartInstrId, instrInfo.missionInstrCount);
    for (uint16_t index = 0; index < instrInfo.instrVec.size(); index++) {
        HCCL_INFO("%d: %s", instrInfo.startInstrId + index, ParseInstr(instrInfo.instrVec.data() + index).c_str());
    }
}

void CcuRepTranslator::DumpRep(const std::vector<std::shared_ptr<CcuRepBase>> &repVec,
                               const CcuInstrInfo                             &instrInfo) const
{
    HCCL_INFO("Translated Ccu Rep:");
    for (uint32_t index = 0; index < repVec.size(); index++) {
        HCCL_INFO("rep[%u]: %s", index, repVec[index]->Describe().c_str());
        uint16_t startInstrId = repVec[index]->StartInstrId();
        uint32_t sum = static_cast<uint32_t>(startInstrId) + repVec[index]->InstrCount();
        if (sum > UINT16_MAX) {
            HCCL_ERROR("instrId overflow: startInstrId[%u] + InstrCount[%u] = %u exceeds UINT16_MAX",
                       startInstrId, repVec[index]->InstrCount(), sum);
            continue;
        }
        uint16_t endInstrId = static_cast<uint16_t>(sum);
        for (uint16_t instrId = startInstrId; instrId < endInstrId; instrId++) {
            if (instrId < instrInfo.startInstrId) {
                HCCL_ERROR("instrId[%u] less than startInstrId[%u]", instrId, instrInfo.startInstrId);
                continue;
            }
            HCCL_INFO("microcode[%u]: %s", instrId,
                      ParseInstr(instrInfo.instrVec.data() + (instrId - instrInfo.startInstrId)).c_str());
        }
    }
}

void CcuRepTranslator::BindResource(bool isFuncBlock)
{
    transDep.reserveXnId  = var[0].Id();
    transDep.reserveGsaId = addr[0].Id();
    transDep.reserveCkeId = signal[0].Id();
    for (int i = 0; i < XN_NUM - 1; i++) {
        transDep.commXn[i] = var[i + 1].Id();
    }
    for (int i = 0; i < GSA_NUM - 1; i++) {
        transDep.commGsa[i] = addr[i + 1].Id();
    }
    transDep.commSignal = signal[1].Id();
    transDep.isFuncBlock = isFuncBlock;
    HCCL_INFO("TransDep info: logicalId = %d, dieId = %u, reserveXnId = %u, reserveGsaId = %u, reserveCkeId = %u, "
              "innerDieChannelId = %u, interDieChannelId = %u",
              transDep.logicalId, transDep.dieId, transDep.reserveXnId, transDep.reserveGsaId, transDep.reserveCkeId,
              transDep.reserveChannalId[0], transDep.reserveChannalId[1]);
}
}; // namespace CcuRep
}; // namespace hcomm
