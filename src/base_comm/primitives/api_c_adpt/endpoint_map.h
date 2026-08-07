/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENDPOINT_MAP_H
#define ENDPOINT_MAP_H

#include "hcomm_res_defs.h"
#include "endpoint.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

namespace hcomm {

    class HcommEndpointMap {
    public:
        void AddEndpoint(EndpointHandle handle, std::unique_ptr<Endpoint> endpoint);

        bool RemoveEndpoint(EndpointHandle handle);

        bool UpdateEndpoint(EndpointHandle handle, std::unique_ptr<Endpoint> newEndpoint);

        Endpoint* GetEndpoint(EndpointHandle handle);
    };
} // namespace hcomm

#ifdef __cplusplus
}
#endif // __cplusplus

#endif
