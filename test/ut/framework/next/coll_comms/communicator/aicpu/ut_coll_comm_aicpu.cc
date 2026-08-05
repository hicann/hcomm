#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#include "coll_comm_aicpu.h"
#include "ns_recovery/aicpu/ns_recovery_lite.h"

#define private public
using namespace hccl;

class CollCommAicpuTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(CollCommAicpuTest, Ut_DefaultStatus_IsInvalid_And_SetGet_Works) {
    CollCommAicpu coll;
    // default constructed status should be INVALID as in header init
    EXPECT_EQ(coll.GetCommmStatus(), HcclCommStatus::HCCL_COMM_STATUS_INVALID);

    coll.SetCommmStatus(HcclCommStatus::HCCL_COMM_STATUS_READY);
    EXPECT_EQ(coll.GetCommmStatus(), HcclCommStatus::HCCL_COMM_STATUS_READY);
}

TEST_F(CollCommAicpuTest, Ut_Clean_EmptyUbTransportMap_Returns_SUCCESS) {
    CollCommAicpu coll;
    // with empty ubTransportMap_ Clean should return success
    auto ret = coll.Clean();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CollCommAicpuTest, Ut_GetNsRecoveryLitePtr_DefaultNull_And_Settable) {
    CollCommAicpu coll;
    // default should be nullptr
    EXPECT_EQ(coll.GetNsRecoveryLitePtr(), nullptr);

    // manually assign a NsRecoveryLite and verify getter
    auto nsPtr = std::make_shared<NsRecoveryLite>();
    coll.nsRecoveryLitePtr_ = nsPtr; // private exposed by define
    EXPECT_NE(coll.GetNsRecoveryLitePtr(), nullptr);
}

TEST_F(CollCommAicpuTest, Ut_Resume_CallsProcessUrmaRes_And_ResetsNsRecoveryFlags) {
    CollCommAicpu coll;

    // ensure nsRecoveryLitePtr_ exists so Resume can call SetNeedClean/ResetErrorReported
    coll.nsRecoveryLitePtr_ = std::make_shared<NsRecoveryLite>();
    coll.SetCommmStatus(HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING);

    // Mock ProcessUrmaRes to avoid running heavy logic; return success
    MOCKER_CPP(&CollCommAicpu::ProcessUrmaRes, HcclResult(CollCommAicpu::*)(HcclChannelUrmaRes*, bool))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    HcclChannelUrmaRes commParam{};
    // call Resume with mocked ProcessUrmaRes
    auto ret = coll.Resume(&commParam);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // after Resume commStatus_ should be READY
    EXPECT_EQ(coll.GetCommmStatus(), HcclCommStatus::HCCL_COMM_STATUS_READY);
    // nsRecovery flags should have been reset
    EXPECT_FALSE(coll.GetNsRecoveryLitePtr()->IsNeedClean());
}

TEST_F(CollCommAicpuTest, Ut_CheckIndOpExecStatus_Return_HCCL_E_SUSPENDING) {
    CollCommAicpu coll;
    coll.SetCommmStatus(HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING);
    auto ret = coll.CheckIndOpExecStatus(0);
    EXPECT_EQ(ret, HCCL_E_SUSPENDING);
}

TEST_F(CollCommAicpuTest, Ut_InitIndopEnv_Expect_Success) {
    CollCommAicpu coll;
    CommAicpuParam param{};
    param.commConfig.taskExceptionEnable = true;
    param.commConfig.notifyWaitTimeout = 1836;
    param.commConfig.plfDebugConfig = 0;
    coll.InitIndopEnv(&param);
}
