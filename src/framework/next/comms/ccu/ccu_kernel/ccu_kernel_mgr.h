/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_KERNEL_MGR_H
#define HCOMM_CCU_KERNEL_MGR_H

#include <mutex>
#include <unordered_map>

#include "ccu_kernel.h"
#include "ccu_res_pack.h"

#include "ccu_dev_mgr_imp.h"
#include "../ccu_representation/reps/translator/ccu_rep_translator_v1.h"

namespace hcomm {

using namespace CcuRep;

class CcuKernelMgr {
public:
    static CcuKernelMgr &GetInstance(const s32 deviceLogicId);

    HcclResult Init();
    HcclResult Deinit();

    HcclResult Register(std::unique_ptr<CcuKernel> kernel,
        CcuResPack &resPack, CcuKernelHandle &kernelHandle);
    HcclResult Translate(const std::vector<CcuKernelHandle> &kernelHandles);
    HcclResult UnRegister(const CcuKernelHandle kernelHandle);
    CcuKernel *GetKernel(const CcuKernelHandle kernelHandle);

private:
    explicit CcuKernelMgr() = default;
    ~CcuKernelMgr();

    CcuKernelMgr(const CcuKernelMgr &that) = delete;
    CcuKernelMgr &operator=(const CcuKernelMgr &that) = delete;

private:
    struct CcuTranslatResPack {
        std::vector<CcuResHandle> handles{};
    };

private:
    HcclResult AllocRes(std::unique_ptr<CcuKernel> &kernel, CcuResPack &resPack);

    HcclResult InstantiationTranslator(const uint16_t dieId);
    HcclResult TransRepSequenceToMicrocode(const std::vector<CcuKernel *> &kernels,
        bool isFuncBlock);
    HcclResult LoadInstruction(const CcuRep::CcuInstrInfo &instrInfo, const uint32_t dieId);

    HcclResult GetResPackTotalResRepository(const CcuTranslatResPack &resPack,
        CcuResRepository &totalRes) const;

private:
    bool initializedFlag_{false};
    int32_t devLogicId_{-1};
    std::mutex kernelMapMutex_;
    std::mutex translateMutex_;
    CcuKernelHandle kernelId_ = 0;
    std::unordered_map<CcuKernelHandle, std::unique_ptr<CcuKernel>> kernelMap_{};
    void *instructionLoadDevMem_{nullptr};

    std::unordered_map<uint16_t, std::unordered_map<uint16_t, std::shared_ptr<CcuRepTranslator>>> translators;
    std::unordered_map<uint16_t, std::unordered_map<uint16_t, std::shared_ptr<CcuRepReferenceManager>>> referenceMgrs;
    CcuTranslatResPack translatorResPack{};
};
}; // namespace hcomm

#endif // HCOMM_CCU_KERNEL_MGR_IMP_H