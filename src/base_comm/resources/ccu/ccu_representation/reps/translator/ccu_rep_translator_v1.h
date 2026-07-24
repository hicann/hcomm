/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu rep translator header file
 * Create: 2025-02-20
 */

#ifndef HCOMM_CCU_REP_TRANSLATOR_H
#define HCOMM_CCU_REP_TRANSLATOR_H

#include <memory>
#include <vector>
#include <functional>

#include "ccu_instr_info_v1.h"
#include "ccu_rep_block_v1.h"
#include "ccu_rep_reference_manager_v1.h"
#include "ccu_kernel_resource.h"
#include "ccu_rep_base_v1.h"
#include "ccu_kernel.h"
#include "ccu_dev_mgr_imp.h"

namespace hcomm {
namespace CcuRep {

class CcuRepTranslator {
public:
    CcuRepTranslator(int32_t deviceLogicId, uint8_t dieId,
        std::shared_ptr<CcuRepReferenceManager> refManager,
        std::array<uint16_t, CCU_MAX_IODIE_NUM> &reserverChannalId,
        std::pair<uint64_t, uint64_t> &ccuTokenInfo, uint64_t hbmTokenInfo);

    CcuRepTranslator(std::shared_ptr<CcuRepReferenceManager> refManager, const TransDep &transDep);
    static uint32_t             GetInstrNum();
    static CcuResReq GetResReq(uint8_t dieId);
    void                        GetRes(CcuRepResource &res);
    CcuInstrInfo Translate(CcuKernel* ccuKernel, const std::vector<std::shared_ptr<CcuRepBase>> &repVec,
                            uint16_t startInstrId, bool isFuncBlock=false);
    void         Translate(CcuKernel* ccuKernel, const std::vector<std::shared_ptr<CcuRepBase>> &repVec,
                            CcuInstr *&instr, uint16_t &instrId, std::function<bool(std::shared_ptr<CcuRepBase>)> filter);
    void         DumpInstruction(const CcuInstrInfo &instrInfo) const;
    void         SetTransDep(TransDep transDepIn) {transDep = transDepIn;}
    TransDep&    GetTransDep() {return transDep;}

private:
    template <typename T1, typename T2> void BuildReference(const std::shared_ptr<CcuRepBase> &rep);
    void                                     PreProcess(std::shared_ptr<CcuRepBase> rep);
    void                                     CommonProcess(CcuKernel* ccuKernel, CcuInstr *&instr, uint16_t &instrId);
    void                                     FinishMainBlock(CcuInstr *&instr, uint16_t &instrId);
    void DumpRep(const std::vector<std::shared_ptr<CcuRepBase>> &repVec, const CcuInstrInfo &instrInfo) const;
    void BindResource(bool isFuncBlock);

private:
    static const int XN_NUM = 4; // 4: Xn资源个数
    static const int GSA_NUM = 3; // 3: GSA资源个数
    static const int CKE_NUM = 2; // 2: CKE资源个数
    std::shared_ptr<CcuRepReferenceManager> refManager{nullptr};
    Variable                             var[XN_NUM];
    Address                              addr[GSA_NUM];
    LocalNotify                          signal[CKE_NUM];
    TransDep                             transDep{0};
    static CcuVersion                    ccuVersion;
};
}; // namespace CcuRep
}; // namespace hcomm

#endif // HCCL_CCU_REP_TRANSLATOR_H