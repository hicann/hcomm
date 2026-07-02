/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: ccu representation implementation file
 * Author: sunzhepeng
 * Create: 2024-06-17
 */

#include "ccu_rep_v1.h"

#include "string_util.h"

namespace hcomm {
namespace CcuRep {

CcuRepJumpBase::CcuRepJumpBase(const std::string &label, const Variable &targetInstrId) : label(label), targetInstrId(targetInstrId)
{
}

void CcuRepJumpBase::Reference(std::shared_ptr<CcuRepJumpLabel> refRep)
{
    jumpLabel = refRep;
}

CcuRepJump::CcuRepJump(const std::string &label, const Variable &targetInstrId) : CcuRepJumpBase(label, targetInstrId)
{
    type       = CcuRepType::JUMP;
    instrCount = 2; // jump翻译需要2条指令
}

bool CcuRepJump::Translate(CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    if (this->instr == nullptr) {
        this->instrId = instrId;
        this->instr   = instr;
        instr += instrCount;
        instrId += instrCount;
    }

    if (jumpLabel->Translated()) {
        LoadImdToXnInstr(this->instr + 0, targetInstrId.Id(), jumpLabel->StartInstrId());
        JumpInstr(this->instr + 1, targetInstrId.Id(), dep.reserveXnId, 1);

        translated = true;
    }

    return translated;
}

std::string CcuRepJump::Describe()
{
    return Hccl::StringFormat("Jump To Label[%s]", label.c_str());
}

CcuRepJumpNE::CcuRepJumpNE(const std::string &label, const Variable &targetInstrId, const Variable &condition, uint64_t expected)
    : CcuRepJumpBase(label, targetInstrId), condition(condition), expected(expected)
{
    type       = CcuRepType::JUMP_NE;
    instrCount = 2; // jumpNE翻译需要2条指令
}

bool CcuRepJumpNE::Translate(CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    if (this->instr == nullptr) {
        this->instrId = instrId;
        this->instr   = instr;
        instr += instrCount;
        instrId += instrCount;
    }

    if (jumpLabel->Translated()) {
        LoadImdToXnInstr(this->instr + 0, targetInstrId.Id(), jumpLabel->StartInstrId());
        JumpInstr(this->instr + 1, targetInstrId.Id(), condition.Id(), expected);

        translated = true;
    }

    return translated;
}

std::string CcuRepJumpNE::Describe()
{
    return Hccl::StringFormat("Jump To Label[%s], When Condition[%u] Not equal to Expected[%lu]", label.c_str(), condition.Id(),
                        expected);
}

CcuRepJumpEQ::CcuRepJumpEQ(const std::string &label, const Variable &targetInstrId, const Variable &condition, uint64_t expected)
    : CcuRepJumpBase(label, targetInstrId), condition(condition), expected(expected)
{
    type       = CcuRepType::JUMP_EQ;
    instrCount = 5; // jumpEQ翻译需要5条指令
}

bool CcuRepJumpEQ::Translate(CcuInstr *&instr, uint16_t &instrId, const TransDep &dep)
{
    if (instr == nullptr) {
        return false;
    }
    if (this->instr == nullptr) {
        this->instrId = instrId;
        this->instr   = instr;
        instr += instrCount;
        instrId += instrCount;
    }

    if (jumpLabel->Translated()) {
        uint32_t localInstrIndex = 0;
        LoadImdToXnInstr(this->instr + localInstrIndex++, targetInstrId.Id(),
                         this->instrId + 4); // 需要指向NOP位置，为输入指令Id + 4
        JumpInstr(this->instr + localInstrIndex++, targetInstrId.Id(), condition.Id(), expected);
        LoadImdToXnInstr(this->instr + localInstrIndex++, targetInstrId.Id(), jumpLabel->StartInstrId());
        JumpInstr(this->instr + localInstrIndex++, targetInstrId.Id(), dep.reserveXnId, 1);
        LoadImdToXnInstr(this->instr + localInstrIndex++, dep.reserveXnId, 0);

        translated = true;
    }

    return translated;
}

std::string CcuRepJumpEQ::Describe()
{
    return Hccl::StringFormat("Jump To Label[%s], When Condition[%u] Be equal to Expected[%lu]", label.c_str(), condition.Id(),
                        expected);
}

}; // namespace CcuRep
}; // namespace hcomm