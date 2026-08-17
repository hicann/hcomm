/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "hcomm_nic_plugin.h"
#include "nic_plugin_manager.h"

namespace {

int32_t DummyCreateEndpoint(const EndpointDesc*, void**, HcommNicEndpointOps**) { return HCCL_SUCCESS; }

int32_t DummyCreateChannel(void*, const HcommChannelDesc*, void**, HcommNicChannelOps**) { return HCCL_SUCCESS; }

TEST(NicPluginManagerValidatePluginInfo, RejectsProtocolBetweenUbgAndCustomBase)
{
    // 覆盖 nic_plugin_manager.cc 中协议区间校验:
    // (protocol < COMM_PROTOCOL_HCCS || protocol > COMM_PROTOCOL_UBG) && protocol < COMM_PROTOCOL_CUSTOM_BASE
    HcommNicPluginInfo info{};
    info.header.version = HCOMM_NIC_PLUGIN_INFO_VERSION;
    info.header.magicWord = HCOMM_NIC_PLUGIN_INFO_MAGIC_WORD;
    info.header.size = sizeof(HcommNicPluginInfo);
    info.protocolCount = 1U;
    info.protocols[0] = static_cast<CommProtocol>(COMM_PROTOCOL_UBG + 1); // 10, 在内置区间之外且低于 CUSTOM_BASE
    EXPECT_FALSE(hcomm::ValidatePluginInfo("ut_nic_plugin.so", &info, DummyCreateEndpoint, DummyCreateChannel));
}

} // namespace
