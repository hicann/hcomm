#include <iostream>
#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "hcomm_c_adpt.h"
#include "server_socket/server_socket_manager.h"
#define private public
using namespace hcomm;

class ServerSocketManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ServerSocketManagerTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "ServerSocketManagerTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in ServerSocketManagerTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in ServerSocketManagerTest TearDown" << std::endl;
    }
};

TEST_F(ServerSocketManagerTest, Ut_When_Device_Socket_Listen_Expect_SUCCESS)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 60001;
    HcclResult ret = ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::DEVICE_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::DEVICE_NIC_TYPE, 60001);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ServerSocketManagerTest, Ut_When_Host_Socket_Listen_Expect_SUCCESS)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 60001;
    HcclResult ret
        = ServerSocketManager::GetInstance().ServerSocketStartListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 60001);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ServerSocketManagerTest, Ut_When_Stop_Listen_While_Not_Start_Listen_Expect_Fail)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    HcclResult ret
        = ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::DEVICE_NIC_TYPE, 60001);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(ServerSocketManagerTest, Ut_When_Stop_Listen_While_Listen_Count_Is_Zero_Listen_Expect_Fail)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 60001;
    HcclResult ret
        = ServerSocketManager::GetInstance().ServerSocketStartListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ServerSocketManager::GetInstance().hostServerSocketMap_.clear();
    ret = ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 60001);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}
TEST_F(ServerSocketManagerTest, Ut_When_ServerSocketStartListen_Device_Expect_SUCCESS)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 60001;
    HcclResult ret = ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::DEVICE_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(port, 0);
    ret = ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::DEVICE_NIC_TYPE, port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ServerSocketManagerTest, Ut_When_ServerSocketStartListen_Host_Expect_SUCCESS)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 0;
    HcclResult ret
        = ServerSocketManager::GetInstance().ServerSocketStartListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(port, 0);
    ret = ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::HOST_NIC_TYPE, port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ServerSocketManagerTest, Ut_When_ServerSocketStartListen_Illegal_NicType_Expect_Fail)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 0;
    HcclResult ret
        = ServerSocketManager::GetInstance().ServerSocketStartListen(localPort, Hccl::NicType::INVALID, 0, &port);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(ServerSocketManagerTest, Ut_When_DeInit_WithEmptyMaps_Expect_NoCrash)
{
    ServerSocketManager::GetInstance().DeInit(0);
    EXPECT_EQ(ServerSocketManager::GetInstance().deviceServerSocketMap_.size(), 0);
    EXPECT_EQ(ServerSocketManager::GetInstance().hostServerSocketMap_.size(), 0);
}

TEST_F(ServerSocketManagerTest, Ut_When_DeInit_WithDeviceSocket_Expect_Clear)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 43210;
    HcclResult ret = ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::DEVICE_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(ServerSocketManager::GetInstance().deviceServerSocketMap_.size(), 0);

    ServerSocketManager::GetInstance().DeInit(0);
    EXPECT_EQ(ServerSocketManager::GetInstance().deviceServerSocketMap_.size(), 0);
}

TEST_F(ServerSocketManagerTest, Ut_When_DeInit_WithHostSocket_Expect_Clear)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 0;
    HcclResult ret
        = ServerSocketManager::GetInstance().ServerSocketStartListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(port, 0);
    EXPECT_NE(ServerSocketManager::GetInstance().hostServerSocketMap_.size(), 0);

    ServerSocketManager::GetInstance().DeInit(0);
    EXPECT_EQ(ServerSocketManager::GetInstance().hostServerSocketMap_.size(), 0);
}

TEST_F(ServerSocketManagerTest, Ut_When_DeInitDeviceSockets_Expect_OnlyMatchingDevPhyId)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 43210;
    HcclResult ret = ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::DEVICE_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ServerSocketManager::GetInstance().DeInitDeviceSockets(0);
}

TEST_F(ServerSocketManagerTest, Ut_When_DeInitHostSockets_Expect_OnlyMatchingDevPhyId)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 0;
    HcclResult ret
        = ServerSocketManager::GetInstance().ServerSocketStartListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ServerSocketManager::GetInstance().DeInitHostSockets(0);
}

TEST_F(ServerSocketManagerTest, Ut_When_DeInitDeviceSockets_NonMatchingDevPhyId_Expect_NotClear)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 43210;
    HcclResult ret = ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::DEVICE_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(ServerSocketManager::GetInstance().deviceServerSocketMap_.size(), 0);

    ServerSocketManager::GetInstance().DeInitDeviceSockets(999);
    EXPECT_NE(ServerSocketManager::GetInstance().deviceServerSocketMap_.size(), 0);

    ServerSocketManager::GetInstance().DeInitDeviceSockets(0);
    EXPECT_EQ(ServerSocketManager::GetInstance().deviceServerSocketMap_.size(), 0);
}

TEST_F(ServerSocketManagerTest, Ut_When_DeInitHostSockets_NonMatchingDevPhyId_Expect_NotClear)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 0;
    HcclResult ret
        = ServerSocketManager::GetInstance().ServerSocketStartListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(ServerSocketManager::GetInstance().hostServerSocketMap_.size(), 0);

    ServerSocketManager::GetInstance().DeInitHostSockets(999);
    EXPECT_NE(ServerSocketManager::GetInstance().hostServerSocketMap_.size(), 0);

    ServerSocketManager::GetInstance().DeInitHostSockets(0);
    EXPECT_EQ(ServerSocketManager::GetInstance().hostServerSocketMap_.size(), 0);
}

TEST_F(ServerSocketManagerTest, Ut_When_DeviceSocketReuse_Expect_CountIncrement)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 43211;
    HcclResult ret = ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::DEVICE_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ServerSocketManager::GetInstance().deviceServerSocketMap_[localPort][43211].second, 1);

    ret = ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::DEVICE_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ServerSocketManager::GetInstance().deviceServerSocketMap_[localPort][43211].second, 2);

    ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::DEVICE_NIC_TYPE, 43211);
    ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::DEVICE_NIC_TYPE, 43211);
}

TEST_F(ServerSocketManagerTest, Ut_When_HostSocketReuse_Expect_CountIncrement)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 43212;
    HcclResult ret
        = ServerSocketManager::GetInstance().ServerSocketStartListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ServerSocketManager::GetInstance().hostServerSocketMap_[localPort][43212].second, 1);

    ret = ServerSocketManager::GetInstance().ServerSocketStartListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ServerSocketManager::GetInstance().hostServerSocketMap_[localPort][43212].second, 2);

    ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 43212);
    ServerSocketManager::GetInstance().ServerSocketStopListen(localPort, Hccl::NicType::HOST_NIC_TYPE, 43212);
}
