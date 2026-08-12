/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_api_base_test.h"
#include "coll_comm_mgr.h"
extern thread_local s32 g_hcclDeviceId;

class HcclGetRootInfoTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        // 将建链超时时间设置为1s，减少测试用例运行时间
        MOCKER(GetExternalInputHcclLinkTimeOut).stubs().with(mockcpp::any()).will(returnValue(1));
    }
    void TearDown() override
    {
        // 删除所有拓扑建链的线程
        HcclOpInfoCtx& opBaseInfo = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx(g_hcclDeviceId);
        opBaseInfo.hcclCommTopoInfoDetectServer.clear();
        opBaseInfo.hcclCommTopoInfoDetectAgent.clear();

        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(HcclGetRootInfoTest, Ut_HcclGetRootInfo_When_RootInfoIsNull_Expect_ReturnIsHCCL_E_PTR)
{
    Ut_Device_Set(0);

    HcclResult ret = HcclGetRootInfo(nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclGetRootInfoTest, Ut_HcclGetRootInfo_When_NotSetEnv_Expect_ReturnIsHCCL_SUCCESS)
{
    unsetenv("HCCL_IF_IP");
    unsetenv("HCCL_WHITELIST_DISABLE");
    unsetenv("HCCL_WHITELIST_FILE");
    Ut_Device_Set(0);
    HcclRootInfo id;
    memset(id.internal, 0, HCCL_ROOT_INFO_BYTES);

    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(strncmp(id.internal, "127.0.0.1", strlen("127.0.0.1")), 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclGetRootInfoTest, Ut_HcclGetRootInfo_When_NotSetWHITELISTFILE_Expect_ReturnIsHCCL_E_PARA)
{
    const char* IpConfig = "127.0.0.2";
    setenv("HCCL_IF_IP", IpConfig, 1);
    setenv("HCCL_WHITELIST_DISABLE", "0", 1);
    unsetenv("HCCL_WHITELIST_FILE");
    Ut_Device_Set(0);
    HcclRootInfo id;
    memset(id.internal, 0, HCCL_ROOT_INFO_BYTES);

    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_NE(strncmp(id.internal, IpConfig, strlen(IpConfig)), 0);
    EXPECT_EQ(ret, HCCL_E_PARA);

    unsetenv("HCCL_WHITELIST_DISABLE");
    unsetenv("HCCL_IF_IP");
}

TEST_F(HcclGetRootInfoTest, Ut_HcclGetRootInfo_When_IPNotInWhiteList_Expect_IdNotEqWithHCCL_IF_IP)
{
    const char* IpConfig = "127.0.0.2";
    const char* WhitelistFilePath = "./whitelist.json";
    setenv("HCCL_IF_IP", IpConfig, 1);
    setenv("HCCL_WHITELIST_DISABLE", "0", 1);
    setenv("HCCL_WHITELIST_FILE", WhitelistFilePath, 1);
    nlohmann::json WhiteList = {{"host_ip", {"1.1.1.1"}}, {"device_ip", {}}};
    std::ofstream outfile(WhitelistFilePath, std::ios::out | std::ios::trunc | std::ios::binary);
    if (outfile.is_open()) {
        outfile << std::setw(1) << WhiteList << std::endl;
        HCCL_INFO("open %s success", WhitelistFilePath);
    } else {
        HCCL_ERROR("open %s failed", WhitelistFilePath);
    }
    outfile.close();
    Ut_Device_Set(0);
    HcclRootInfo id;
    memset(id.internal, 0, HCCL_ROOT_INFO_BYTES);

    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_NE(strncmp(id.internal, IpConfig, strlen(IpConfig)), 0);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);

    remove(WhitelistFilePath);
    unsetenv("HCCL_WHITELIST_FILE");
    unsetenv("HCCL_WHITELIST_DISABLE");
    unsetenv("HCCL_IF_IP");
}

TEST_F(HcclGetRootInfoTest, Ut_HcclGetRootInfo_When_IPInWhiteList_Expect_IdEqWithHCCL_IF_IP)
{
    const char* IpConfig = "127.0.0.2";
    const char* WhitelistFilePath = "./whitelist.json";
    setenv("HCCL_IF_IP", IpConfig, 1);
    setenv("HCCL_WHITELIST_DISABLE", "0", 1);
    setenv("HCCL_WHITELIST_FILE", WhitelistFilePath, 1);
    nlohmann::json WhiteList = {{"host_ip", {IpConfig}}, {"device_ip", {}}};
    std::ofstream outfile(WhitelistFilePath, std::ios::out | std::ios::trunc | std::ios::binary);
    if (outfile.is_open()) {
        outfile << std::setw(1) << WhiteList << std::endl;
        HCCL_INFO("open %s success", WhitelistFilePath);
    } else {
        HCCL_ERROR("open %s failed", WhitelistFilePath);
    }
    outfile.close();
    Ut_Device_Set(0);
    HcclRootInfo id;
    memset(id.internal, 0, HCCL_ROOT_INFO_BYTES);

    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(strncmp(id.internal, IpConfig, strlen(IpConfig)), 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    remove(WhitelistFilePath);
    unsetenv("HCCL_WHITELIST_FILE");
    unsetenv("HCCL_WHITELIST_DISABLE");
    unsetenv("HCCL_IF_IP");
}

TEST_F(HcclGetRootInfoTest, Ut_HcclGetRootInfo_When_SetWhiteListDisable_Expect_IdEqWithHCCL_IF_IP)
{
    const char* IpConfig = "127.0.0.2";
    setenv("HCCL_IF_IP", IpConfig, 1);
    setenv("HCCL_WHITELIST_DISABLE", "1", 1);
    Ut_Device_Set(0);
    HcclRootInfo id;
    memset(id.internal, 0, HCCL_ROOT_INFO_BYTES);

    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(strncmp(id.internal, IpConfig, strlen(IpConfig)), 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    unsetenv("HCCL_WHITELIST_DISABLE");
    unsetenv("HCCL_IF_IP");
}

TEST_F(HcclGetRootInfoTest, Ut_HcclGetRootInfo_When_SetHOSTNAME_Expect_IdEqWithHCCL_SOCKET_IFNAME)
{
    const char* HostnameConfig = "=eth0";
    setenv("HCCL_SOCKET_IFNAME", HostnameConfig, 1);
    setenv("HCCL_WHITELIST_DISABLE", "1", 1);
    Ut_Device_Set(0);
    HcclRootInfo id;
    memset(id.internal, 0, HCCL_ROOT_INFO_BYTES);

    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(strncmp(id.internal, "127.0.0.1", strlen("127.0.0.1")), 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    unsetenv("HCCL_WHITELIST_DISABLE");
    unsetenv("HCCL_SOCKET_IFNAME");
    unsetenv("HCCL_IF_IP");
}

TEST_F(HcclGetRootInfoTest, Ut_HcclGetRootInfo_When_SetIPAndHOSTNAME_Expect_IdEqWithHCCL_IF_IP)
{
    const char* IpConfig1 = "127.0.0.2";
    const char* HostnameConfig2 = "=eth2";
    setenv("HCCL_IF_IP", IpConfig1, 1);
    setenv("HCCL_SOCKET_IFNAME", HostnameConfig2, 1);
    setenv("HCCL_WHITELIST_DISABLE", "1", 1);
    Ut_Device_Set(0);
    HcclRootInfo id;
    memset(id.internal, 0, HCCL_ROOT_INFO_BYTES);

    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(strncmp(id.internal, IpConfig1, strlen(IpConfig1)), 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    unsetenv("HCCL_WHITELIST_DISABLE");
    unsetenv("HCCL_SOCKET_IFNAME");
    unsetenv("HCCL_IF_IP");
}

TEST_F(HcclGetRootInfoTest, Ut_HcclGetRootInfo_When_SetHOSTNAMEAsEth_Expect_IdEqWithFirstIP)
{
    const char* HostnameConfig = "eth";
    unsetenv("HCCL_IF_IP");
    setenv("HCCL_SOCKET_IFNAME", HostnameConfig, 1);
    setenv("HCCL_WHITELIST_DISABLE", "1", 1);
    Ut_Device_Set(0);
    HcclRootInfo id;
    memset(id.internal, 0, HCCL_ROOT_INFO_BYTES);

    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(strncmp(id.internal, "127.0.0.1", strlen("127.0.0.1")), 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    unsetenv("HCCL_WHITELIST_DISABLE");
    unsetenv("HCCL_SOCKET_IFNAME");
}
