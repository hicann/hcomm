/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <string>

#include "gtest/gtest.h"
#include "base_config.h"
#include "hcomm_res_mgr.h"

using namespace hcomm;

namespace {

/// 保存/恢复单个环境变量，RAII 保证用例间隔离
class EnvGuard {
public:
    explicit EnvGuard(const char* name) : name_(name)
    {
        const char* val = std::getenv(name);
        if (val != nullptr) {
            savedValue_ = val;
            hadValue_ = true;
        }
    }
    ~EnvGuard()
    {
        if (hadValue_) {
            (void)setenv(name_, savedValue_.c_str(), 1);
        } else {
            (void)unsetenv(name_);
        }
    }
    void Set(const char* value) { (void)setenv(name_, value, 1); }
    void Unset() { (void)unsetenv(name_); }

private:
    const char* name_;
    std::string savedValue_;
    bool hadValue_{false};
};

} // namespace

// ==================== HCOMM_TA_CTP_UB_TIMEOUT ====================

TEST(ConfigMgrTest, Ut_TaCtpUbTimeOut_NotSet_Expect_Default8)
{
    EnvGuard guard("HCOMM_TA_CTP_UB_TIMEOUT");
    guard.Unset();
    HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_EQ(HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaCtpUbTimeOut(value), HCCL_SUCCESS);
    EXPECT_EQ(value, 8U);
}

TEST(ConfigMgrTest, Ut_TaCtpUbTimeOut_ValidValue0_Expect_0)
{
    EnvGuard guard("HCOMM_TA_CTP_UB_TIMEOUT");
    guard.Set("0");
    HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_EQ(HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaCtpUbTimeOut(value), HCCL_SUCCESS);
    EXPECT_EQ(value, 0U);
}

TEST(ConfigMgrTest, Ut_TaCtpUbTimeOut_ValidValue31_Expect_31)
{
    EnvGuard guard("HCOMM_TA_CTP_UB_TIMEOUT");
    guard.Set("31");
    HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_EQ(HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaCtpUbTimeOut(value), HCCL_SUCCESS);
    EXPECT_EQ(value, 31U);
}

TEST(ConfigMgrTest, Ut_TaCtpUbTimeOut_InvalidValue32_Expect_Error)
{
    EnvGuard guard("HCOMM_TA_CTP_UB_TIMEOUT");
    guard.Set("32"); // 超过 UB_TIMEOUT_MAX=31，校验失败返回错误码
    HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_NE(HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaCtpUbTimeOut(value), HCCL_SUCCESS);
}

TEST(ConfigMgrTest, Ut_TaCtpUbTimeOut_InvalidString_Expect_Error)
{
    EnvGuard guard("HCOMM_TA_CTP_UB_TIMEOUT");
    guard.Set("abc"); // 非数字，解析失败返回错误码
    HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_NE(HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaCtpUbTimeOut(value), HCCL_SUCCESS);
}

TEST(ConfigMgrTest, Ut_TaCtpUbTimeOut_NegativeValue_Expect_Error)
{
    EnvGuard guard("HCOMM_TA_CTP_UB_TIMEOUT");
    guard.Set("-1"); // StrToNum 检查全数字，负号被拒绝，解析失败返回错误码
    HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_NE(HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaCtpUbTimeOut(value), HCCL_SUCCESS);
}

TEST(ConfigMgrTest, Ut_TaCtpUbTimeOut_EmptyString_Expect_Default8)
{
    EnvGuard guard("HCOMM_TA_CTP_UB_TIMEOUT");
    guard.Set(""); // 空字符串视为未设置
    HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_EQ(HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaCtpUbTimeOut(value), HCCL_SUCCESS);
    EXPECT_EQ(value, 8U);
}

// ==================== HCOMM_TA_RTP_UB_TIMEOUT ====================

TEST(ConfigMgrTest, Ut_TaRtpUbTimeOut_NotSet_Expect_Default16)
{
    EnvGuard guard("HCOMM_TA_RTP_UB_TIMEOUT");
    guard.Unset();
    HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_EQ(HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaRtpUbTimeOut(value), HCCL_SUCCESS);
    EXPECT_EQ(value, 16U);
}

TEST(ConfigMgrTest, Ut_TaRtpUbTimeOut_ValidValue0_Expect_0)
{
    EnvGuard guard("HCOMM_TA_RTP_UB_TIMEOUT");
    guard.Set("0");
    HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_EQ(HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaRtpUbTimeOut(value), HCCL_SUCCESS);
    EXPECT_EQ(value, 0U);
}

TEST(ConfigMgrTest, Ut_TaRtpUbTimeOut_ValidValue31_Expect_31)
{
    EnvGuard guard("HCOMM_TA_RTP_UB_TIMEOUT");
    guard.Set("31");
    HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_EQ(HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaRtpUbTimeOut(value), HCCL_SUCCESS);
    EXPECT_EQ(value, 31U);
}

TEST(ConfigMgrTest, Ut_TaRtpUbTimeOut_InvalidValue32_Expect_Error)
{
    EnvGuard guard("HCOMM_TA_RTP_UB_TIMEOUT");
    guard.Set("32"); // 超过 UB_TIMEOUT_MAX=31，校验失败返回错误码
    HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_NE(HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaRtpUbTimeOut(value), HCCL_SUCCESS);
}

TEST(ConfigMgrTest, Ut_TaRtpUbTimeOut_NegativeValue_Expect_Error)
{
    EnvGuard guard("HCOMM_TA_RTP_UB_TIMEOUT");
    guard.Set("-1"); // StrToNum 检查全数字，负号被拒绝，解析失败返回错误码
    HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_NE(HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaRtpUbTimeOut(value), HCCL_SUCCESS);
}

// ==================== HCOMM_TA_RTP_UBOE_TIMEOUT ====================

TEST(ConfigMgrTest, Ut_TaRtpUboeTimeOut_NotSet_Expect_Default16)
{
    EnvGuard guard("HCOMM_TA_RTP_UBOE_TIMEOUT");
    guard.Unset();
    HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_EQ(HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaRtpUboeTimeOut(value), HCCL_SUCCESS);
    EXPECT_EQ(value, 16U);
}

TEST(ConfigMgrTest, Ut_TaRtpUboeTimeOut_ValidValue0_Expect_0)
{
    EnvGuard guard("HCOMM_TA_RTP_UBOE_TIMEOUT");
    guard.Set("0");
    HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_EQ(HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaRtpUboeTimeOut(value), HCCL_SUCCESS);
    EXPECT_EQ(value, 0U);
}

TEST(ConfigMgrTest, Ut_TaRtpUboeTimeOut_ValidValue31_Expect_31)
{
    EnvGuard guard("HCOMM_TA_RTP_UBOE_TIMEOUT");
    guard.Set("31");
    HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_EQ(HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaRtpUboeTimeOut(value), HCCL_SUCCESS);
    EXPECT_EQ(value, 31U);
}

TEST(ConfigMgrTest, Ut_TaRtpUboeTimeOut_InvalidValue100_Expect_Error)
{
    EnvGuard guard("HCOMM_TA_RTP_UBOE_TIMEOUT");
    guard.Set("100"); // 超过 UB_TIMEOUT_MAX=31，校验失败返回错误码
    HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_NE(HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaRtpUboeTimeOut(value), HCCL_SUCCESS);
}

TEST(ConfigMgrTest, Ut_TaRtpUboeTimeOut_InvalidString_Expect_Error)
{
    EnvGuard guard("HCOMM_TA_RTP_UBOE_TIMEOUT");
    guard.Set("xyz"); // 非数字，解析失败返回错误码
    HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_NE(HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaRtpUboeTimeOut(value), HCCL_SUCCESS);
}

TEST(ConfigMgrTest, Ut_TaRtpUboeTimeOut_NegativeValue_Expect_Error)
{
    EnvGuard guard("HCOMM_TA_RTP_UBOE_TIMEOUT");
    guard.Set("-1"); // StrToNum 检查全数字，负号被拒绝，解析失败返回错误码
    HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_NE(HcommResMgr::GetInstance().GetConfigMgr().GetRdmaConfig().GetTaRtpUboeTimeOut(value), HCCL_SUCCESS);
}

// ==================== EnvField 无 parser 函数 ====================

TEST(ConfigMgrTest, Ut_EnvField_NoParser_Expect_Error)
{
    EnvGuard guard("HCOMM_TEST_NO_PARSER");
    guard.Set("123");
    EnvField<uint32_t> field("HCOMM_TEST_NO_PARSER", 0U, nullptr);
    EXPECT_NE(field.Parse(), HCCL_SUCCESS);
}
