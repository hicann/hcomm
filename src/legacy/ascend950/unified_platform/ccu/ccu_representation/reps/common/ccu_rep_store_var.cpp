/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: ccu representation store implementation file
 * Create: 2026-01-12
 */

#include "ccu_rep.h"
#include "string_util.h"

namespace Hccl {
namespace CcuRep {

    CcuRepStoreVar::CcuRepStoreVar(const Variable& src, const Variable& var) : src(src), var(var)
    {
        type = CcuRepType::STORE;
        instrCount = 7; // 7: Store包含7条指令
    }

    bool CcuRepStoreVar::Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;
        uint64_t varAddr = dep.xnBaseAddr + CCU_RESOURCE_XN_PER_SIZE * var.Id();

        LoadGSAXnInstr(instr++, dep.commGsa[0], dep.reserveGsaId, src.Id());
        LoadImdToGSAInstr(instr++, dep.commGsa[1], varAddr);
        LoadImdToXnInstr(instr++, dep.commXn[0], dep.memTokenInfo, CCU_LOAD_TO_XN_SEC_INFO);
        LoadImdToXnInstr(instr++, dep.commXn[1], dep.ccuResSpaceTokenInfo, CCU_LOAD_TO_XN_SEC_INFO);
        LoadImdToXnInstr(instr++, dep.commXn[2], CCU_RESOURCE_XN_PER_SIZE);
        TransLocMemToLocMemInstr(
            instr++, dep.commGsa[0], dep.commXn[0], dep.commGsa[1], dep.commXn[1], dep.commXn[2],
            dep.reserveChannalId[0], dep.commSignal, mask, 0, 0, 1, 1);
        SetCKEInstr(instr++, 0, 0, dep.commSignal, mask, 1);
        instrId += instrCount;

        return translated;
    }

    std::string CcuRepStoreVar::Describe() { return StringFormat("Store Var([%llu], [%u])", src.Id(), var.Id()); }

}; // namespace CcuRep
}; // namespace Hccl
