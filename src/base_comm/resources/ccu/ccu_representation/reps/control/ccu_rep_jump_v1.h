/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef CCU_REPRESENTATION_JUMP_H
#define CCU_REPRESENTATION_JUMP_H

#include <memory>

#include "ccu_datatype_v1.h"
#include "ccu_rep_base_v1.h"
#include "ccu_rep_jumplabel_v1.h"
#include "ccu_types.h"

namespace hcomm {
namespace CcuRep {

enum class ConditionType { EQUAL, NOT_EQUAL, GREATER_THAN, GREATER_EQUAL, LESS_THAN, LESS_EQUAL, DEFAULT, INVALID };

class CcuRepJumpBase : public CcuRepBase {
public:
    explicit CcuRepJumpBase(CcuInsGeneratorBase* insGenPtr, const std::string &label, const Variable &targetInstrId);
    explicit CcuRepJumpBase(CcuInsGeneratorBase* insGenPtr, const std::string &label, const Variable &targetInstrId,
                            const Variable& expectedVar, const Variable& condition);
    void Reference(std::shared_ptr<CcuRepJumpLabel> refRep);
    void ValidateInsGeneratorForJump();

    std::shared_ptr<CcuRepJumpLabel> GetJumpLabel() {
        return jumpLabel;
    }

    Variable& GetTargetInstrId() {
        return targetInstrId;
    }

    Variable& GetCondition() {
        return condition;
    }

    Variable& GetExpectedVar() {
        return expectedVar;
    }

    uint64_t GetExpectedNum() {
        return expected;
    }

    CcuInstr* GetInstr() {
        return instr;
    }

    bool IsComparedWithImmd() {
        return comp2Immed;
    }

protected:
    CcuResult InitInstr(CcuInstr *&instr, uint16_t &instrId);

    CcuInsGeneratorBase*             insGeneratorPtr_;
    std::string                      label;
    std::shared_ptr<CcuRepJumpLabel> jumpLabel{nullptr};
    Variable                         targetInstrId;
    CcuInstr                        *instr{nullptr};

    bool                             comp2Immed{false};
    bool                             supportCcuV1{true};  // 暂定用于识别 使用特定的构造方法时是否支持A5的翻译流程
 
    Variable expectedVar;
    Variable condition;
    uint64_t expected{0};
};

class CcuRepJump : public CcuRepJumpBase {
public:
    explicit CcuRepJump(CcuInsGeneratorBase* insGenPtr, const std::string &label, const Variable &targetInstrId);
    bool        Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep) override;
    std::string Describe() override;
};

class CcuRepJumpNE : public CcuRepJumpBase {
public:
    CcuRepJumpNE(CcuInsGeneratorBase* insGenPtr, const std::string &label, const Variable &targetInstrId, const Variable &expectedVar, const Variable &condition, uint64_t expected);
    CcuRepJumpNE(CcuInsGeneratorBase* insGenPtr, const std::string &label, const Variable &targetInstrId, const Variable &condition, const Variable &expectedVar);  // 仅用于A6翻译
    bool        Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep) override;
    std::string Describe() override;
};

class CcuRepJumpEQ : public CcuRepJumpBase {
public:
    CcuRepJumpEQ(CcuInsGeneratorBase* insGenPtr, const std::string &label, const Variable &targetInstrId, const Variable &expectedVar, const Variable &condition, uint64_t expected);
    CcuRepJumpEQ(CcuInsGeneratorBase* insGenPtr, const std::string &label, const Variable &targetInstrId, const Variable &condition, const Variable &expectedVar);  // 仅用于A6翻译
    bool        Translate(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId, const TransDep &dep) override;
    std::string Describe() override;
};

class CcuRepJumpLE : public CcuRepJumpBase {
public:
    CcuRepJumpLE(CcuInsGeneratorBase* insGenPtr, const std::string &label, const Variable &targetInstrId, const Variable &condition,
                 const Variable &expectedVar);
    CcuRepJumpLE(CcuInsGeneratorBase* insGenPtr, const std::string &label, const Variable &targetInstrId, const Variable &expectedVar,
                 const Variable &condition, uint64_t expected);
    bool        Translate(CcuKernel* ccuKernel, CcuInstr *&curInstr, uint16_t &curInstrId, const TransDep &dep) override;
    std::string Describe() override;
};
 
class CcuRepJumpGE : public CcuRepJumpBase {
public:
    CcuRepJumpGE(CcuInsGeneratorBase* insGenPtr, const std::string &label, const Variable &targetInstrId, const Variable &condition,
                 const Variable &expectedVar);
    CcuRepJumpGE(CcuInsGeneratorBase* insGenPtr, const std::string &label, const Variable &targetInstrId, const Variable &expectedVar,
                 const Variable &condition, uint64_t expected);
    bool        Translate(CcuKernel* ccuKernel, CcuInstr *&curInstr, uint16_t &curInstrId, const TransDep &dep) override;
    std::string Describe() override;
};
 
class CcuRepJumpGT : public CcuRepJumpBase {
public:
    CcuRepJumpGT(CcuInsGeneratorBase* insGenPtr, const std::string &label, const Variable &targetInstrId, const Variable &condition,
                 const Variable &expectedVar);
    CcuRepJumpGT(CcuInsGeneratorBase* insGenPtr, const std::string &label, const Variable &targetInstrId, const Variable &expectedVar,
                 const Variable &condition, uint64_t expected);
    bool        Translate(CcuKernel* ccuKernel, CcuInstr *&curInstr, uint16_t &curInstrId, const TransDep &dep) override;
    std::string Describe() override;
};
 
class CcuRepJumpLT : public CcuRepJumpBase {
public:
    CcuRepJumpLT(CcuInsGeneratorBase* insGenPtr, const std::string &label, const Variable &targetInstrId, const Variable &condition,
                 const Variable &expectedVar);
    CcuRepJumpLT(CcuInsGeneratorBase* insGenPtr, const std::string &label, const Variable &targetInstrId, const Variable &expectedVar,
                 const Variable &condition, uint64_t expected);
    bool        Translate(CcuKernel* ccuKernel, CcuInstr *&curInstr, uint16_t &curInstrId, const TransDep &dep) override;
    std::string Describe() override;
};
}; // namespace CcuRep
}; // namespace hcomm
#endif // _CCU_REPRESENTATION_JUMP_H