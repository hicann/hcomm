#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "cpu_roce_endpoint.h"
#include "hcomm_res.h"
#include "hcomm_c_adpt.h"
#include "rdma_handle_manager.h"
#include "buffer/local_rdma_rma_buffer.h"
#include "ip_address.h"
#include "hccp.h"
#include "buffer.h"
#include "network_api_exception.h"
#include "endpoint.h"

class CpuRoceEndpointTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CpuRoceEndpointTest tests set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "CpuRoceEndpointTest tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        std::cout << "A Test case in CpuRoceEndpointTest SetUP" << std::endl;
        Hccl::IpAddress   localIp("1.0.0.0");
        Hccl::IpAddress   remoteIp("2.0.0.0");
        fakeSocket = new Hccl::Socket(nullptr, localIp, listenPort, remoteIp, tag, Hccl::SocketRole::SERVER, Hccl::NicType::HOST_NIC_TYPE);
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        delete fakeSocket;
        std::cout << "A Test case in HostRdmaConnection TearDown" << std::endl;
    }
    Hccl::Socket     *fakeSocket;
    
    u32         listenPort = 100;
    std::string tag        = "test";
    RdmaHandle   rdmaHandle = (void *)0x1000000;
};

// HcommEndpointCreate
TEST_F(CpuRoceEndpointTest, Ut_When_Normal_EXPECT_Return_HCCL_SUCCESS)
{
    Hccl::IpAddress   localIp("1.0.0.0");
    EndpointDesc endpointDesc;
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    void* endpointHandle{nullptr};
    MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle));
    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// HcommEndpointCreate fail
TEST_F(CpuRoceEndpointTest, Ut_When_wrongIp_EXPECT_Return_128003)
{
    Hccl::IpAddress   localIp("223.0.0.1");
    EndpointDesc endpointDesc;
    endpointDesc.protocol = COMM_PROTOCOL_UBC_CTP;
    endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    void* endpointHandle{nullptr};
    MOCKER(&Hccl::RdmaHandleManager::GetByIp).stubs().will(throws(Hccl::NetworkApiException("error")));
    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, 1);
}

// RdmaHandle初始化失败
TEST_F(CpuRoceEndpointTest, Ut_When_RdmaHandle_Init_Fail_Expect_Return_HCCL_E_PTR)
{
    Hccl::IpAddress   localIp("1.0.0.0");
    EndpointDesc endpointDesc;
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    void* endpointHandle{nullptr};
    RdmaHandle rdmaHandle2{nullptr};
    MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle2));
    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// HcommEndpointStartListen
TEST_F(CpuRoceEndpointTest, Ut_When_HcommEndpointStartListen_EXPECT_Return_HCCL_SUCCESS)
{
    Hccl::IpAddress   localIp("1.0.0.0");
    EndpointDesc endpointDesc;
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    void* endpointHandle{nullptr};
    MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle));
    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcommEndpointStartListen(endpointHandle, 60001, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// Ip重复监听
TEST_F(CpuRoceEndpointTest, Ut_When_Listen_Repeat_Ip_EXPECT_Return_HCCL_SUCCESS)
{
    Hccl::IpAddress   localIp("1.0.0.0");
    EndpointDesc endpointDesc;
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    void* endpointHandle{nullptr};
    MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle));
    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcommEndpointStartListen(endpointHandle, 60001, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcommEndpointStartListen(endpointHandle, 60001, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 停止监听
TEST_F(CpuRoceEndpointTest, Ut_When_Stop_Listen_EXPECT_Return_HCCL_SUCCESS)
{
    Hccl::IpAddress   localIp("1.0.0.0");
    EndpointDesc endpointDesc;
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    void* endpointHandle{nullptr};
    MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle));
    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcommEndpointStopListen(endpointHandle, 60001);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcommEndpointStartListen(endpointHandle, 60001, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcommEndpointStopListen(endpointHandle, 60001);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 内存注册失败
TEST_F(CpuRoceEndpointTest, Ut_When_Register_Memory_Fail_Expect_Return_HCCL_E_PTR)
{
    Hccl::IpAddress   localIp("1.0.0.0");
    EndpointDesc endpointDesc;
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    void* endpointHandle{nullptr};
    MOCKER_CPP(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle));
    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    hcomm::CpuRoceEndpoint* endpoint = static_cast<hcomm::CpuRoceEndpoint*>(endpointHandle);
    HcommMem mem;
    mem.type = COMM_MEM_TYPE_DEVICE;
    mem.addr = malloc(10);
    mem.size = 10;
    ret = endpoint->RegisterMemory(mem, "HcclBuffer", nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
    free(mem.addr);
}

TEST_F(CpuRoceEndpointTest, ut_HcommResMgrInit_When_Normal_Expect_ReturnSuccess)
{
    HcommResult ret = HcommResMgrInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CpuRoceEndpointTest, ut_HcommEndpointGet_When_EndpointNotFound_Expect_ReturnHCCL_E_NOT_FOUND)
{
    void *endpoint = nullptr;
    HcommResult ret = HcommEndpointGet(reinterpret_cast<EndpointHandle>(0x12345678), &endpoint);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(CpuRoceEndpointTest, Ut_When_Unregister_Memory_Fail_Expect_Return_HCCL_E_PTR)
{
    Hccl::IpAddress   localIp("1.0.0.0");
    EndpointDesc endpointDesc;
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    void* endpointHandle{nullptr};
    MOCKER_CPP(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle));
    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    hcomm::CpuRoceEndpoint* endpoint = static_cast<hcomm::CpuRoceEndpoint*>(endpointHandle);
    HcommMem mem;
    mem.type = COMM_MEM_TYPE_DEVICE;
    mem.addr = malloc(10);
    mem.size = 10;
    void* memHandle{nullptr};
    void* mrHandle{nullptr};
    ret = endpoint->UnregisterMemory(memHandle);
    EXPECT_EQ(ret, HCCL_E_PTR);
    auto localBufferPtr = std::make_shared<Hccl::Buffer>(666);
    auto localRdmaRmaBuffer = std::make_shared<Hccl::LocalRdmaRmaBuffer>(localBufferPtr, rdmaHandle);
    memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    ret = endpoint->UnregisterMemory(memHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    free(mem.addr);
}

TEST_F(CpuRoceEndpointTest, ut_HcommEndpointGet_When_EndpointPtrIsNull_Expect_ReturnHCCL_E_PTR)
{
    EndpointHandle handle = reinterpret_cast<EndpointHandle>(0x12345678);
    HcommResult ret = HcommEndpointGet(handle, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(CpuRoceEndpointTest, ut_HcommEndpointDestroy_When_EndpointNotFound_Expect_ReturnHCCL_E_NOT_FOUND)
{
    EndpointHandle handle = reinterpret_cast<EndpointHandle>(0x12345678);
    HcommResult ret = HcommEndpointDestroy(handle);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(CpuRoceEndpointTest, ut_HcommEndpointStartListen_When_EndpointIsNull_Expect_ReturnHCCL_E_NOT_FOUND)
{
    HcommResult ret = HcommEndpointStartListen(nullptr, 100, nullptr);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(CpuRoceEndpointTest, ut_HcommEndpointStopListen_When_EndpointIsNull_Expect_ReturnHCCL_E_NOT_FOUND)
{
    HcommResult ret = HcommEndpointStopListen(nullptr, 100);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(CpuRoceEndpointTest, ut_HcommMemReg_When_MemIsNull_Expect_ReturnHCCL_E_PTR)
{
    HcommMemHandle memHandle;
    HcommResult ret = HcommMemReg(reinterpret_cast<EndpointHandle>(0x12345678), "tag", nullptr, &memHandle);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(CpuRoceEndpointTest, ut_HcommMemReg_When_MemHandleIsNull_Expect_ReturnHCCL_E_PTR)
{
    HcommMem mem;
    mem.type = COMM_MEM_TYPE_DEVICE;
    mem.addr = malloc(10);
    mem.size = 10;
    HcommResult ret = HcommMemReg(nullptr, "tag", &mem, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
    free(mem.addr);
}

TEST_F(CpuRoceEndpointTest, ut_HcommMemUnreg_When_EndpointIsNull_Expect_ReturnHCCL_E_NOT_FOUND)
{
    HcommMemHandle memHandle = reinterpret_cast<HcommMemHandle>(0x12345678);
    HcommResult ret = HcommMemUnreg(nullptr, memHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(CpuRoceEndpointTest, ut_HcommMemExport_When_EndpointIsNull_Expect_ReturnHCCL_E_NOT_FOUND)
{
    void *memDesc = nullptr;
    uint32_t memDescLen = 0;
    HcommResult ret = HcommMemExport(nullptr, reinterpret_cast<HcommMemHandle>(0x12345678), &memDesc, &memDescLen);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

// GetCapabilities 正常调用，返回成功且maxMsgSize为1GB
TEST_F(CpuRoceEndpointTest, ut_GetCapabilities_When_Normal_Expect_ReturnSuccess_And_MaxMsgSz1GB_lbMax0)
{
    EndpointDesc endpointDesc{};
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    hcomm::CpuRoceEndpoint endpoint(endpointDesc);
    hcomm::CpuRoceEndpoint::Capabilities caps;
    endpoint.Init();
    HcommResult ret = endpoint.GetCapabilities(caps);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    static constexpr uint64_t EXPECTED_MAX_MSG_SZ = 1ULL * 1024 * 1024 * 1024;
    EXPECT_EQ(caps.maxMsgSize, EXPECTED_MAX_MSG_SZ);
    EXPECT_EQ(caps.lbMax, 0);
}

// GetCapabilities 多次调用，验证缓存一致性
TEST_F(CpuRoceEndpointTest, ut_GetCapabilities_When_CalledTwice_Expect_SameResult)
{
    EndpointDesc endpointDesc{};
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    hcomm::CpuRoceEndpoint endpoint(endpointDesc);
    hcomm::CpuRoceEndpoint::Capabilities caps1;
    hcomm::CpuRoceEndpoint::Capabilities caps2;
    endpoint.Init();
    HcommResult ret = endpoint.GetCapabilities(caps1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = endpoint.GetCapabilities(caps2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(caps1.maxMsgSize, caps2.maxMsgSize);
}
TEST_F(CpuRoceEndpointTest, ut_HcommEndpointGetListenPort_When_PortIsNull_Expect_ReturnHCCL_E_PTR)
{
    HcommResult ret = HcommEndpointGetListenPort(reinterpret_cast<EndpointHandle>(0x12345678), nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(CpuRoceEndpointTest, ut_HcommEndpointGetListenPort_When_EndpointIsNull_Expect_ReturnHCCL_E_NOT_FOUND)
{
    uint32_t port = 0;
    HcommResult ret = HcommEndpointGetListenPort(nullptr, &port);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}
