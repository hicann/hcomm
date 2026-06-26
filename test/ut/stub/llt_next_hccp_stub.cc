/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstring>

#include "hccp.h"
#include "hccp_ctx.h"
#include "hccp_async.h"
#include "hccp_async_ctx.h"
#include "hccp_tp.h"
#include "orion_adapter_hccp.h"
#include "hccp_nda.h"
 
int RaCtxQpCreate(void *ctx_handle, struct QpCreateAttr *attr, struct QpCreateInfo *info,
    void **qp_handle)
{
    return 0;
}
 
int RaCtxQpDestroy(void *qp_handle)
{
    return 0;
}
 
int RaCtxQpImport(void *ctx_handle, struct QpImportInfoT *qp_info, void **rem_qp_handle)
{
    return 0;
}
 
int RaCtxQpUnimport(void *ctx_handle, void *rem_qp_handle)
{
    return 0;
}
 
int RaGetAsyncReqResult(void *reqHandle, int *reqResult)
{
    *reqResult = 0;
    return 0;
}
 
int RaCtxQpCreateAsync(void *ctx_handle, struct QpCreateAttr *attr,
    struct QpCreateInfo *info, void **qp_handle, void **req_handle)
{
    int a = 12378;
    *req_handle = &a;
    return 0;
}
 
int RaCtxQpImportAsync(void *ctx_handle, struct QpImportInfoT *info, void **rem_qp_handle,
    void **req_handle)
{
    int a = 12378;
    *req_handle = &a;
    return 0;
}
 
int RaGetTpInfoListAsync(void *ctx_handle, struct GetTpCfg *cfg, struct HccpTpInfo info_list[],
    unsigned int *num, void **req_handle)
{
    (void)ctx_handle;
    (void)cfg;
    if (info_list != nullptr) {
        (void)std::memset(info_list, 0, sizeof(struct HccpTpInfo));
        info_list[0].tpHandle = 1U;
    }
    if (num != nullptr) {
        *num = 1U;
    }
    static int kStubRaTpInfoListReq = 12378;
    if (req_handle != nullptr) {
        *req_handle = &kStubRaTpInfoListReq;
    }
    return 0;
}

extern "C" {

int RaGetTpAttrAsync(void *ctxHandle, uint64_t tpHandle, uint32_t *attrBitmap, struct TpAttr *attr, void **reqHandle)
{
    static char kStubRaTpAttrReq{};
    (void)ctxHandle;
    (void)tpHandle;
    (void)attrBitmap;
    if (attr != nullptr) {
        (void)std::memset(attr, 0, sizeof(struct TpAttr));
        attr->slBitmap = 0x7U;
        attr->dscpConfigMode = 1U;
    }
    if (reqHandle != nullptr) {
        *reqHandle = &kStubRaTpAttrReq;
    }
    return 0;
}

int RaCtxGetTpAttr(void *ctxHandle, uint64_t tpHandle, uint32_t *attrBitmap, struct TpAttr *attr)
{
    (void)ctxHandle;
    (void)tpHandle;
    if (attrBitmap != nullptr) {
        *attrBitmap = 0U;
    }
    if (attr != nullptr) {
        (void)std::memset(attr, 0, sizeof(struct TpAttr));
        attr->slBitmap = 0x7U;
        attr->dscpConfigMode = 1U;
    }
    return 0;
}

int RaCtxSetTpAttr(void *ctxHandle, uint64_t tpHandle, uint32_t attrBitmap, struct TpAttr *attr)
{
    (void)ctxHandle;
    (void)tpHandle;
    (void)attrBitmap;
    (void)attr;
    return 0;
}

int RaGetHccnCfg(struct RaInfo *info, enum HccnCfgKey key, char *value, unsigned int *valueLen)
{
    (void)info;
    (void)key;
    if (value != nullptr && valueLen != nullptr && *valueLen > 0U) {
        value[0] = '\0';
    }
    if (valueLen != nullptr) {
        *valueLen = 0U;
    }
    return -1;
}

} // extern "C"
 
int RaCustomChannel(struct RaInfo info, struct CustomChanInfoIn *in,
    struct CustomChanInfoOut *out)
{
    return 0;
}
 
int RaGetDevEidInfoNum(struct RaInfo info, unsigned int *num)
{
    *num = 2;
    return 0;
}
 
int RaGetDevEidInfoList(struct RaInfo info, struct HccpDevEidInfo info_list[],
    unsigned int *num)
{
    if (info.phyId == 0) {
        info_list[0].eid.in4.addr = 167772383;
    } else {
        info_list[0].eid.in4.addr = 469762271;
    }
    
    info_list[0].dieId = 0;
    info_list[0].chipId = 0;
    info_list[0].funcId = 2;
 
    info_list[1].eid.in4.addr = 12346;
    info_list[1].dieId = 1;
    info_list[1].chipId = 0;
    info_list[1].funcId = 3;
 
    return 0;
}

int RaGetSecRandom(struct RaInfo *info, uint32_t *value)
{
    return 0;
}

int RaCtxGetAuxInfo(void *ctx_handle, struct HccpAuxInfoIn *in, struct HccpAuxInfoOut *out) {
    return 0;
}

int RaCtxQpQueryBatch(void *qp_handle[], struct JettyAttr attr[], unsigned int *num) {
    return 0;
}

int RaNdaGetDirectFlag(void *rdmaHandle, int *directFlag)
{
    if (directFlag != nullptr) {
        *directFlag = 1;
    }
    return 0;
}

int RaCtxGetAsyncEvents(void *ctxHandle, struct AsyncEvent events[], unsigned int *num)
{
    *num = 0;
    return 0;
}

namespace Hccl {
HcclResult HrtRaGetTlsStatus(struct RaInfo *info, TlsStatus &tlsStatus)
{
    (void)info;
    tlsStatus = TlsStatus::DISABLE;
    return HCCL_SUCCESS;
}

void HrtRaCustomChannel(const HRaInfo &raInfo, void *customIn, void *customOut)
{
    return;
}

void HrtDeviceAbortRegCallBack(aclrtDeviceTaskAbortCallback callback, void *args, const std::string& name)
{
    return;
}

void HrtRaSocketWhiteListDel(SocketHandle socketHandle, vector<RaSocketWhitelist> &wlists)
{
    return;
}

HcclResult HrtRaGetTpAttrAsync(u32 phyId, RdmaHandle handle, uint64_t tpHandle, uint32_t& attrBitmap, TpAttr& attr, RequestHandle& reqHandle)
{
    (void)phyId;
    (void)handle;
    (void)tpHandle;
    (void)attrBitmap;
    (void)attr;
    (void)reqHandle;
    return HCCL_SUCCESS;
}

HcclResult HrtRaSetTpAttrAsync(RdmaHandle handle, uint64_t tpHandle, uint32_t attrBitmap, TpAttr& attr, RequestHandle& reqHandle)
{
    (void)handle;
    (void)tpHandle;
    (void)attrBitmap;
    (void)attr;
    reqHandle = 0;
    return HCCL_SUCCESS;
}

int RaCtxGetTpInfoList(void *ctxHandle, struct GetTpCfg *cfg, struct HccpTpInfo infoList[],
    unsigned int *num)
{
    (void)ctxHandle;
    (void)cfg;
    if (infoList != nullptr) {
        (void)std::memset(infoList, 0, sizeof(struct HccpTpInfo));
        infoList[0].tpHandle = 1U;
    }
    if (num != nullptr) {
        *num = 1U;
    }
    return 0;
}

} // namespace Hccl

int RaGetDevBaseAttr(void *ctxHandle, struct DevBaseAttr *attr)
{
    return 0;
}
