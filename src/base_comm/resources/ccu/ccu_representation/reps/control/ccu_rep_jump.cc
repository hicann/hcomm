/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ccu representation implementation file
 * Author: sunzhepeng
 * Create: 2024-06-17
 */

#include "ccu_rep_v1.h"
#include "exception_util.h"
#include "ccu_api_exception.h"
#include "ccu_ins_generater_base.h"
#include "ccu_ins_generater_v1.h"

namespace hcomm {
namespace CcuRep {

using namespace Hccl;

// jump基类
CcuRepJumpBase::CcuRepJumpBase(CcuInsGeneraterBase* insGenPtr, const std::string &label, const Variable &targetInstrId):
    insGeneratorPtr_(insGenPtr), label(label), targetInstrId(targetInstrId)
{
}

CcuRepJumpBase::CcuRepJumpBase(CcuInsGeneraterBase* insGenPtr, const std::string &label, const Variable &targetInstrId,
                               const Variable& expectedVar, const Variable& condition):
    insGeneratorPtr_(insGenPtr), label(label), targetInstrId(targetInstrId), expectedVar(expectedVar), condition(condition)
{
}

void CcuRepJumpBase::Reference(std::shared_ptr<CcuRepJumpLabel> refRep)
{
    jumpLabel = refRep;
}

void CcuRepJumpBase::ValidateInsGeneratorForJump()
{
    CcuInsGeneraterV1* tmpPtrV1 = dynamic_cast<CcuInsGeneraterV1*>(insGeneratorPtr_);
    if (tmpPtrV1 && !supportCcuV1) {
        // 当右值只传入var没有立即数时，无法在A5上翻译
        Hccl::THROW<Hccl::CcuApiException>("Cannot translate %s for A5 when supportCcuV1 is false!",
            this->Describe().c_str());
    }
}

CcuResult CcuRepJumpBase::InitInstr(CcuInstr *&instr, uint16_t &instrId)
{
    CCU_CHK_PTR_NULL(instr);
    if (this->instr == nullptr) {
        this->instrId = instrId;
        this->instr   = instr;
        instr += instrCount;
        instrId += instrCount;
    }
    return CcuResult::CCU_SUCCESS;
}

// direct jump
CcuRepJump::CcuRepJump(CcuInsGeneraterBase* insGenPtr, const std::string &label, const Variable &targetInstrId) :
    CcuRepJumpBase(insGenPtr, label, targetInstrId)
{
    type       = CcuRepType::JUMP;
    instrCount = insGeneratorPtr_->GetInstrCount(type);
}

bool CcuRepJump::Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    if (InitInstr(instr, instrId) != CcuResult::CCU_SUCCESS) {
        Hccl::THROW<Hccl::CcuApiException>("instr is empty!");
    }

    if (jumpLabel->Translated()) {
        CHK_RET_THROW(Hccl::CcuApiException,
            Hccl::StringFormat("[CcuRepJump][%s] failed to translate repJump for instrId[%u] ", __func__, instrId),
                insGeneratorPtr_->CcuRepJumpTranslate(ccuKernel, instr, instrId, this, dep));
        translated = true;
    }

    return translated;
}

std::string CcuRepJump::Describe()
{
    return Hccl::StringFormat("Jump To Label[%s]", label.c_str());
}

// jumpNE
CcuRepJumpNE::CcuRepJumpNE(CcuInsGeneraterBase* insGenPtr, const std::string &label, const Variable &targetInstrId,
    const Variable &expectedVar, const Variable &condition, uint64_t expected)
    : CcuRepJumpBase(insGenPtr, label, targetInstrId, expectedVar, condition)
{
    this->expected = expected;
    type       = CcuRepType::JUMP_NE;
    instrCount = insGeneratorPtr_->GetInstrCount(type);
    comp2Immed = true;  // A6翻译时插入一条加载立即数指令，A5无关
    supportCcuV1 = true;
}

CcuRepJumpNE::CcuRepJumpNE(CcuInsGeneraterBase* insGenPtr, const std::string &label, const Variable &targetInstrId,
    const Variable &condition, const Variable &expectedVar) 
    : CcuRepJumpBase(insGenPtr, label, targetInstrId, expectedVar, condition)
{
    // 仅用于A6翻译
    type       = CcuRepType::JUMP_NE;
    instrCount = 2;  // 2条指令，暂直接填充指令数，insGenerator中未记录这种使用方式对应的指令数
    comp2Immed = false;
    supportCcuV1 = false;
}

bool CcuRepJumpNE::Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    ValidateInsGeneratorForJump();

    if (InitInstr(instr, instrId) != CcuResult::CCU_SUCCESS) {
        Hccl::THROW<Hccl::CcuApiException>("instr is empty!");
    }

    if (jumpLabel->Translated()) {
        CHK_RET_THROW(Hccl::CcuApiException,
            Hccl::StringFormat("[CcuRepJumpNE][%s] failed to translate repJumpNE for instrId[%u] ", __func__, instrId),
                insGeneratorPtr_->CcuRepJumpNETranslate(ccuKernel, instr, instrId, this, dep));
        translated = true;
    }

    return translated;
}

std::string CcuRepJumpNE::Describe()
{
    return Hccl::StringFormat("Jump To Label[%s], When Condition[%u] Not equal to Expected[%lu]", label.c_str(), condition.Id(),
                        expected);
}

// jumpEQ
CcuRepJumpEQ::CcuRepJumpEQ(CcuInsGeneraterBase* insGenPtr, const std::string &label, const Variable &targetInstrId,
    const Variable &expectedVar, const Variable &condition, uint64_t expected)
    : CcuRepJumpBase(insGenPtr, label, targetInstrId, expectedVar, condition)
{
    this->expected = expected;
    type       = CcuRepType::JUMP_EQ;
    instrCount = insGeneratorPtr_->GetInstrCount(type);
    comp2Immed = true;  // A6翻译时插入一条加载立即数指令，A5无关
    supportCcuV1 = true;
}

CcuRepJumpEQ::CcuRepJumpEQ(CcuInsGeneraterBase* insGenPtr, const std::string &label, const Variable &targetInstrId,
    const Variable &condition, const Variable &expectedVar)
    : CcuRepJumpBase(insGenPtr, label, targetInstrId, expectedVar, condition)
{
    type       = CcuRepType::JUMP_EQ;
    instrCount = 2; // 2条指令，暂直接填充指令数，insGenerator中未记录这种使用方式对应的指令数
    comp2Immed = false;
    supportCcuV1 = false;
}

bool CcuRepJumpEQ::Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    ValidateInsGeneratorForJump();
    if (InitInstr(instr, instrId) != CcuResult::CCU_SUCCESS) {
        Hccl::THROW<Hccl::CcuApiException>("instr is empty!");
    }

    if (jumpLabel->Translated()) {
        CHK_RET_THROW(Hccl::CcuApiException,
            Hccl::StringFormat("[CcuRepJumpEQ][%s] failed to translate repJumpEQ for instrId[%u] ", __func__, instrId),
                insGeneratorPtr_->CcuRepJumpEQTranslate(ccuKernel, instr, instrId, this, dep));
        translated = true;
    }

    return translated;
}

std::string CcuRepJumpEQ::Describe()
{
    return Hccl::StringFormat("Jump To Label[%s], When Condition[%u] Be equal to Expected[%lu]", label.c_str(), condition.Id(),
                        expected);
}

// jumpLE
CcuRepJumpLE::CcuRepJumpLE(CcuInsGeneraterBase* insGenPtr, const std::string &label, const Variable &targetInstrId,
    const Variable &condition, const Variable &expectedVar)
    : CcuRepJumpBase(insGenPtr, label, targetInstrId, expectedVar, condition)
{
    type                             = CcuRepType::JUMP_LE;
    instrCount                       = 2;  // 2条指令
    comp2Immed                       = false;
    supportCcuV1                     = false;
}
 
CcuRepJumpLE::CcuRepJumpLE(CcuInsGeneraterBase* insGenPtr, const std::string &label, const Variable &targetInstrId,
    const Variable &expectedVar, const Variable &condition, uint64_t expected)
    : CcuRepJumpBase(insGenPtr, label, targetInstrId, expectedVar, condition)
{
    this->expected = expected;
    type                             = CcuRepType::JUMP_LE;
    instrCount                       = 3;  // 3条指令
    comp2Immed                       = true;
    supportCcuV1                     = false;
}
 
bool CcuRepJumpLE::Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    ValidateInsGeneratorForJump();
 
    if (InitInstr(instr, instrId) != CcuResult::CCU_SUCCESS) {
        Hccl::THROW<Hccl::CcuApiException>("instr is empty!");
    }
 
    if (jumpLabel->Translated()) {
        CHK_RET_THROW(Hccl::CcuApiException,
            Hccl::StringFormat("[CcuRepJumpLE][%s] failed to translate repJumpLE for instrId[%u] ", __func__, instrId),
                insGeneratorPtr_->CcuRepJumpLETranslate(ccuKernel, instr, instrId, this, dep));
        translated = true;
    }
 
    return translated;
}
 
std::string CcuRepJumpLE::Describe()
{
    return Hccl::StringFormat("Jump To Label[%s], When Condition[%u] <= Expected[%lu]", label.c_str(), condition.Id(),
                        expected);
}
 
// jumpGE
CcuRepJumpGE::CcuRepJumpGE(CcuInsGeneraterBase* insGenPtr, const std::string &label, const Variable &targetInstrId,
    const Variable &condition, const Variable &expectedVar)
    : CcuRepJumpBase(insGenPtr, label, targetInstrId, expectedVar, condition)
{
    type                             = CcuRepType::JUMP_GE;
    instrCount                       = 2;  // 2条指令
    comp2Immed                       = false;
    supportCcuV1                     = false;
}
 
CcuRepJumpGE::CcuRepJumpGE(CcuInsGeneraterBase* insGenPtr, const std::string &label, const Variable &targetInstrId,
    const Variable &expectedVar, const Variable &condition, uint64_t expected)
    : CcuRepJumpBase(insGenPtr, label, targetInstrId, expectedVar, condition)
{
    this->expected = expected;
    type                             = CcuRepType::JUMP_GE;
    instrCount                       = 3;  // 3条指令
    comp2Immed                       = true;
    supportCcuV1                     = false;
}
 
bool CcuRepJumpGE::Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    ValidateInsGeneratorForJump();
 
    if (InitInstr(instr, instrId) != CcuResult::CCU_SUCCESS) {
        Hccl::THROW<Hccl::CcuApiException>("instr is empty!");
    }
 
    if (jumpLabel->Translated()) {
        CHK_RET_THROW(Hccl::CcuApiException,
            Hccl::StringFormat("[CcuRepJumpGE][%s] failed to translate repJumpGE for instrId[%u] ", __func__, instrId),
                insGeneratorPtr_->CcuRepJumpGETranslate(ccuKernel, instr, instrId, this, dep));
 
        translated = true;
    }
 
    return translated;
}
 
std::string CcuRepJumpGE::Describe()
{
    return Hccl::StringFormat("Jump To Label[%s], When Condition[%u] >= Expected[%lu]", label.c_str(), condition.Id(),
                        expected);
}
 
// jumpGT
CcuRepJumpGT::CcuRepJumpGT(CcuInsGeneraterBase* insGenPtr, const std::string &label, const Variable &targetInstrId,
    const Variable &condition, const Variable &expectedVar)
    : CcuRepJumpBase(insGenPtr, label, targetInstrId, expectedVar, condition)
{
    type                             = CcuRepType::JUMP_GT;
    instrCount                       = 2;  // 2条指令
    comp2Immed                       = false;
    supportCcuV1                     = false;
}
 
CcuRepJumpGT::CcuRepJumpGT(CcuInsGeneraterBase* insGenPtr, const std::string &label, const Variable &targetInstrId,
    const Variable &expectedVar, const Variable &condition, uint64_t expected)
    : CcuRepJumpBase(insGenPtr, label, targetInstrId, expectedVar, condition)
{
    this->expected = expected;
    type                             = CcuRepType::JUMP_GT;
    instrCount                       = 3;  // 3条指令
    comp2Immed                       = true;
    supportCcuV1                     = false;
}
 
bool CcuRepJumpGT::Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    ValidateInsGeneratorForJump();
    if (InitInstr(instr, instrId) != CcuResult::CCU_SUCCESS) {
        Hccl::THROW<Hccl::CcuApiException>("instr is empty!");
    }
    if (jumpLabel->Translated()) {
        CHK_RET_THROW(Hccl::CcuApiException,
            Hccl::StringFormat("[CcuRepJumpGT][%s] failed to translate repJumpGT for instrId[%u] ", __func__, instrId),
                insGeneratorPtr_->CcuRepJumpGTTranslate(ccuKernel, instr, instrId, this, dep));
 
        translated = true;
    }
 
    return translated;
}
 
std::string CcuRepJumpGT::Describe()
{
    return Hccl::StringFormat("Jump To Label[%s], When Condition[%u] > Expected[%lu]", label.c_str(), condition.Id(),
                        expected);
}
 
// jumpLT
CcuRepJumpLT::CcuRepJumpLT(CcuInsGeneraterBase* insGenPtr, const std::string &label, const Variable &targetInstrId,
    const Variable &condition, const Variable &expectedVar)
    : CcuRepJumpBase(insGenPtr, label, targetInstrId, expectedVar, condition)
{
    type                             = CcuRepType::JUMP_LT;
    instrCount                       = 2;  // 2条指令
    comp2Immed                       = false;
    supportCcuV1                     = false;
}
 
CcuRepJumpLT::CcuRepJumpLT(CcuInsGeneraterBase* insGenPtr, const std::string &label, const Variable &targetInstrId,
    const Variable &expectedVar, const Variable &condition, uint64_t expected)
    : CcuRepJumpBase(insGenPtr, label, targetInstrId, expectedVar, condition)
{
    this->expected = expected;
    type                             = CcuRepType::JUMP_LT;
    instrCount                       = 3;  // 3条指令
    comp2Immed                       = true;
    supportCcuV1                     = false;
}
 
bool CcuRepJumpLT::Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    ValidateInsGeneratorForJump();
    if (InitInstr(instr, instrId) != CcuResult::CCU_SUCCESS) {
        Hccl::THROW<Hccl::CcuApiException>("instr is empty!");
    }
 
    if (jumpLabel->Translated()) {
        CHK_RET_THROW(Hccl::CcuApiException,
            Hccl::StringFormat("[CcuRepJumpLT][%s] failed to translate repJumpLT for instrId[%u] ", __func__, instrId),
                insGeneratorPtr_->CcuRepJumpLTTranslate(ccuKernel, instr, instrId, this, dep));
 
        translated = true;
    }
 
    return translated;
}
 
std::string CcuRepJumpLT::Describe()
{
    return Hccl::StringFormat("Jump To Label[%s], When Condition[%u] < Expected[%lu]", label.c_str(), condition.Id(),
                        expected);
}
}; // namespace CcuRep
}; // namespace hcomm