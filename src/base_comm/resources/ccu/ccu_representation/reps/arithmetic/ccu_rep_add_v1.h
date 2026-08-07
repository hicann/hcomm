/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu representation base header file
 * Create: 2025-02-18
 */

#ifndef HCOMM_CCU_REPRESENTATION_ADD_H
#define HCOMM_CCU_REPRESENTATION_ADD_H

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepAdd : public CcuRepBase {
    public:
        // support ccuV1 & ccuV2
        explicit CcuRepAdd(
            CcuInsGeneratorBase* insGenPtr, const Address& addrC, const Address& addrA, const Variable& varB);
        explicit CcuRepAdd(
            CcuInsGeneratorBase* insGenPtr, const Address& addrC, const Address& addrA, const Address& addrB);
        explicit CcuRepAdd(
            CcuInsGeneratorBase* insGenPtr, const Variable& varC, const Variable& varA, const Variable& varB);
        explicit CcuRepAdd(CcuInsGeneratorBase* insGenPtr, const Address& addrA, const Variable& offset);
        explicit CcuRepAdd(CcuInsGeneratorBase* insGenPtr, const Variable& varA, const Variable& offset);

        // only support ccuV2
        explicit CcuRepAdd(
            CcuInsGeneratorBase* insGenPtr, const Variable& varC, const Variable& varA, const uint16_t immedB);
        explicit CcuRepAdd(CcuInsGeneratorBase* insGenPtr, const Variable& varA, const uint16_t immedB);
        explicit CcuRepAdd(
            CcuInsGeneratorBase* insGenPtr, const Address& addrC, const Address& addrA, const uint16_t immedB);
        explicit CcuRepAdd(CcuInsGeneratorBase* insGenPtr, const Address& addrA, const uint16_t immedB);
        explicit CcuRepAdd(
            CcuInsGeneratorBase* insGenPtr, const Address& addrC, const Variable& varA, const uint16_t immedB);
        explicit CcuRepAdd(
            CcuInsGeneratorBase* insGenPtr, const Variable& varC, const Address& addrA, const uint16_t immedB);
        explicit CcuRepAdd(
            CcuInsGeneratorBase* insGenPtr, const Variable& varC, const Address& addrA, const Address& addrB);
        explicit CcuRepAdd(
            CcuInsGeneratorBase* insGenPtr, const Address& addrC, const Variable& varA, const Variable& varB);

        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;

        Address GetAddrA();
        Address GetAddrB();
        Address GetAddrC();
        Variable GetVarA();
        Variable GetVarB();
        Variable GetVarC();
        uint16_t GetImmedB();
        AddSubType GetSubType();

    private:
        void ValidateInsGenPtrForAdd();
        void SetCommonInfo();

        CcuInsGeneratorBase* insGenPtr{nullptr};
        AddSubType subType{AddSubType::INVALID};

        Address addrA;
        Address addrB;
        Address addrC;

        Variable varA;
        Variable varB;
        Variable varC;

        uint16_t immedB{0};

        bool supportCcuV1{false};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCCL_CCU_REPRESENTATION_ADD_H
