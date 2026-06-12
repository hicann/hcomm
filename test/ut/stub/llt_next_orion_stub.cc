/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rdma_handle_manager.h"
#include "socket/socket.h"
#include "sal.h"

#include "buffer.h"
#include "dev_buffer.h"
#include "local_ub_rma_buffer.h"
#include "socket_handle_manager.h"
#include "base_config.h"
#include "env_config.h"
#include <initializer_list>
#include "topo_info.h"
#include "rank_graph_builder.h"
#include "orion_adapter_rts.h"
#include "net_instance.h"
#include "host_socket_handle_manager.h"
#include "hccp_hdc_manager.h"
#include "rma_buffer_lite.h"
#include "ub_mem_transport.h"
#include "socket/socket.h"
#include "notify_lite.h"
#include "stream_lite.h"
#include "rtsq_base.h"
#include "ub_conn_lite_mgr.h"
#include "rmt_rma_buf_slice_lite.h"
#include "aicpu_res_package_helper.h"
#include "dev_ub_connection.h"
// #include "hccl_one_sided_service.h"
#include "ub_local_notify.h"
#include "op_base_v2.h"
#include "local_rdma_rma_buffer_manager.h"
#include "local_ub_rma_buffer_manager.h"
#include "local_rdma_rma_buffer.h"
#include "local_rma_buffer.h"
#include "local_rdma_rma_buffer_manager.h"

#include "rdma_handle_manager.h"
#include "socket/socket.h"
#include "sal.h"

#include "buffer.h"
#include "dev_buffer.h"
#include "local_ub_rma_buffer.h"
#include "socket_handle_manager.h"
#include "base_config.h"
#include "env_config.h"
#include <initializer_list>
#include "topo_info.h"
#include "rank_graph_builder.h"
#include "orion_adapter_rts.h"
#include "net_instance.h"
#include "host_socket_handle_manager.h"
#include "base_config.h"

#include "../../../legacy/unified_platform/resource/buffer/local_ipc_rma_buffer.h"

#include "../../../legacy/framework/resource_manager/socket/socket_manager.h"

#include "tp_manager.h"
#include "inner_net_dev_manager.h"
#include "hccp_hdc_manager.h"
#include "hccp_peer_manager.h"
#include "hccp_tlv_hdc_manager.h"
#include "ccu_driver_handle.h"
#include "rdma_handle_manager.h"
#include "socket_handle_manager.h"
#include "host_socket_handle_manager.h"

#include "ccu_context_mgr_imp.h"
#include "../../../legacy/unified_platform/ccu/ccu_device/ccu_res_batch_allocator.h"
#include "ccu_component.h"
#include "../../../legacy/unified_platform/ccu/ccu_device/ccu_res_specs.h"
#include "task_info.h"

#include <sstream>
#include <iostream>
#include <cstdint>
#include <iomanip>
#include <array>
#include "rt_external.h"
#include "adapter_rts_common.h"
#include "externalinput_pub.h"
#include "config_log.h"
#include "sal_pub.h"
#include "acl/error_codes/rt_error_codes.h"

#include "dispatcher_task_types.h"
#include "../../../legacy/framework/dfx/common/task_info.h"
#include "../../../legacy/framework/dfx/common/mirror_task_manager.h"
#include "../../../legacy/framework/dfx/common/global_mirror_tasks.h"
#include "../../../legacy/framework/dfx/common/circular_queue.h"
#include "../../../legacy/framework/dfx/profiling/profiling_handler.h"
#include "../../../legacy/framework/dfx/profiling/profiling_reporter.h"
#include "../../../legacy/framework/dfx/aicpu/profiling/profiling_handler_lite.h"
#include "../../../legacy/framework/dfx/aicpu/profiling/profiling_reporter_lite.h"
#include "../../../legacy/unified_platform/common/dlhal_function_v2.h"
#include "../../../legacy/framework/dfx/profiling/dlprof_function.h"
#include "../../../legacy/framework/communicator/aicpu/daemon/aicpu_daemon_service.h"
#include "../../../legacy/framework/dfx/task_exception/task_exception_handler.h"
#include "../../../legacy/unified_platform/external_system/orion_adapter_hccp.h"
#include "../../../legacy/include/hccl_communicator.h"
#include "../../../legacy/unified_platform/ccu/ccu_microcode/ccu_assist.h"
#include "acl/acl_rt.h"

#include "p2p_transport.h"
#include "dev_capability.h"
#include "p2p_connection.h"
#include "rts_cnt_notify.h"
#include "rts_1ton_cnt_notify.h"
#include "ipc_local_notify.h"
#include "host_ub_connection.h"
#include "urma_api.h"

namespace Hccl {

void *HrtMalloc(u64 size, aclrtMemType_t memType)
{
    return (void *)0x12345678;
}

void HrtMemset(void *dst, uint64_t destMax, uint64_t count)
{
    memset(dst, 0, count);
    return;
}

RdmaHandleManager::RdmaHandleManager()
{
}

RdmaHandleManager::~RdmaHandleManager()
{
}

RdmaHandle RdmaHandleManager::GetByIp(u32 devPhyId, const IpAddress &localIp)
{
    return (void *)0x12345678;
}

RdmaHandleManager &RdmaHandleManager::GetInstance()
{
    static RdmaHandleManager rdmaHandleManager;
    return rdmaHandleManager;
}
 
JfcHandle RdmaHandleManager::GetJfcHandle(RdmaHandle rdmaHandle, CqCreateInfo& cqInfo, HrtUbJfcMode jfcMode)
{
    return 0x12345678;
}

std::pair<TokenIdHandle, uint32_t> RdmaHandleManager::GetTokenIdInfo(
    RdmaHandle rdmaHandle, const BufferKey<uintptr_t, u64> &bufKey)
{
    return {0x12345678, 12345678};
}

bool RdmaHandleManager::GetRtpEnable(RdmaHandle rdmaHandle)
{
    return true;
}

HcclResult RdmaHandleManager::GetEidByIpv4Addr(const IpAddress &addr, IpAddress &eidAddr)
{
    Hccl::IpAddress ip("0000:0000:0000:0000:0000:0000:c0a8:0367", AF_INET6);
    eidAddr = ip;
    return HCCL_SUCCESS;
}

void RdmaHandleManager::UboeIpv4ToEid(const IpAddress &ipV4Address, IpAddress &eidAddress, u32 devPhyId)
{
    Hccl::IpAddress ip("0000:0000:0000:0000:0000:0000:c0a8:0367", AF_INET6);
    eidAddress = ip;
}

SocketStatus Socket::GetAsyncStatus()
{
    return SocketStatus::OK;
}

void Socket::ConnectAsync()
{
    if (role == SocketRole::SERVER) {
        std::cout << "Socket Server, connect async passed." << std::endl;
        return;
    }

    std::cout << "Socket client, connect async do." << std::endl;
    return;
}

bool Socket::ISend(void *data, u64 size, u64& compSize) const
{
    compSize = size;
    return true;
}

HcclResult Socket::ISendWithHeart(void *data, u64 size, u64& compSize) const
{ 
    compSize = size;
    return HCCL_SUCCESS;
}

HcclResult Socket::IRecvWithHeart(void *data, u64 size, u64& compSize) const
{ 
    compSize = size;
    return HCCL_SUCCESS;
}
 
void Socket::SendAsync(const u8 *sendBuf, u32 size)
{
    return;
}

void Socket::RecvAsync(u8 *recvBuf, u32 size)
{
    return;
}

void Socket::Listen()
{
    std::cout << "Socket Server, listen." << std::endl;
}

bool Socket::Listen(u32 &port)
{
    if (port == 0) {
        port = 43210;
    }
    std::cout << "Socket Server, listen on port" << port << std::endl;
    return true;
}

void Socket::Connect()
{
}

Socket::~Socket()
{
}

void Socket::Destroy()
{
}

std::size_t HashCombine(std::initializer_list<std::size_t> hashItem)
{
    std::size_t res = 17;
    std::size_t padding = 31;
    for (auto begin = hashItem.begin(); begin != hashItem.end(); ++begin) {
        res = padding * res + (*begin);
    }
    return res;
}

DevBuffer::DevBuffer(uintptr_t devAddr, std::size_t devSize) : Buffer(devSize), selfOwned(false)
{
    addr_ = devAddr;
    size_ = devSize;
}

DevBuffer::DevBuffer(std::size_t allocSize) : Buffer(allocSize), selfOwned(true)
{
    addr_ = (uintptr_t)(0x12345678);
}

std::shared_ptr<DevBuffer> DevBuffer::Create(uintptr_t devAddr, std::size_t devSize)
{
    if (devAddr == 0 || devSize == 0) {
        return nullptr;
    }
    return std::shared_ptr<DevBuffer>(new (std::nothrow) DevBuffer(devAddr, devSize));
}

DevBuffer::DevBuffer(std::size_t allocSize, std::uint32_t policy, PolicyTag /*tag*/)
    : Buffer(allocSize),
      selfOwned(true)
{
    addr_ = 0x12345678;
}

DevBuffer::~DevBuffer()
{
}

std::string DevBuffer::Describe() const
{
    return "";
}

LocalUbRmaBuffer::LocalUbRmaBuffer(std::shared_ptr<Buffer> buf) : LocalRmaBuffer(buf, RmaType::UB)
{
}

LocalUbRmaBuffer::LocalUbRmaBuffer(std::shared_ptr<Buffer> buf, RdmaHandle rdmaHandle)
    : LocalRmaBuffer(buf, RmaType::UB)
{
}

LocalUbRmaBuffer::~LocalUbRmaBuffer()
{
}

std::unique_ptr<Serializable> LocalUbRmaBuffer::GetExchangeDto()
{
    return nullptr;
}

std::string LocalUbRmaBuffer::Describe() const
{
    return "";
}

u32 LocalUbRmaBuffer::GetTokenId() const
{
    return 0;
}

u32 LocalUbRmaBuffer::GetTokenValue() const
{
    return 0;
}

TokenIdHandle LocalUbRmaBuffer::GetTokenIdHandle() const
{
    return 0x12345678;
}

u32 GetUbToken()
{
    return 0;
}

void SaluSleep(uint32_t usec)
{
    return;
}

SocketHandleManager::SocketHandleManager()
{
}

SocketHandleManager::~SocketHandleManager()
{
}

SocketHandleManager &SocketHandleManager::GetInstance()
{
    static SocketHandleManager mgr;
    return mgr;
}

SocketHandle SocketHandleManager::Create(DevId devicePhyId, const PortData &localPort)
{
    int a = 0x12345678;
    return (void *)&a;
}

std::shared_ptr<TopoInfo> RankGraphBuilder::GetTopoInfo()
{
    return nullptr;
}

// const EnvTopoFilePathConfig &EnvConfig::GetTopoFilePathConfig()
// {
//     // return topoFilePathCfg;
//     return *((EnvTopoFilePathConfig *)0x12345678);
// }

unique_ptr<RankGraph> RankGraphBuilder::Build(const string &ranktableM, const string &topoPath, RankId myRank)
{
    return nullptr;
}

std::unique_ptr<RankTableInfo> RankGraphBuilder::GetRankTableInfo()
{
    return nullptr;
}

s32 HrtGetDevice()
{
    return 1;
}
u32 HrtGetDevicePhyIdByIndex(s32 deviceLogicId)
{
    return 1U;
}

DevType HrtGetDeviceType()
{
    return DevType::DEV_TYPE_950;
}

RdmaHandle HrtRaRdmaInit(HrtNetworkMode netMode, RaInterface &in)
{
    return (RdmaHandle)0x12345678;
}

void HrtGetSocVer(std::string &socName)
{
    socName = "Ascend958B";
}

HcclResult HrtGetMainboardId(uint32_t deviceLogicId, HcclMainboardId &hcclMainboardId)
{
    (void)deviceLogicId;
    hcclMainboardId = HcclMainboardId::MAINBOARD_OTHERS;
    return HcclResult::HCCL_SUCCESS;
}

QpHandle HrtRaQpCreate(RdmaHandle rdmaHandle, int flag, int qpMode)
{
    return (QpHandle)0x12345678;
}

int HrtGetRaQpStatus(QpHandle qpHandle)
{
    return 0;
}

SocketHandle HostSocketHandleManager::Get(unsigned int, Hccl::IpAddress const &)
{
    return (void *)0x12345678;
}

SocketHandle HostSocketHandleManager::Create(unsigned int, Hccl::IpAddress const &)
{
    return (void *)0x12345678;
}

HostSocketHandleManager &HostSocketHandleManager::GetInstance()
{
    static HostSocketHandleManager hostSocketHandleManager;
    return hostSocketHandleManager;
}

HostSocketHandleManager::~HostSocketHandleManager()
{
}
HostSocketHandleManager::HostSocketHandleManager()
{
}

SocketStatus Socket::GetStatus()
{
    return SocketStatus::OK;
}
bool Socket::Send(const void *sendBuf, u32 size) const
{
    return true;
}
bool Socket::Recv(void *recvBuf, u32 size) const
{
    return true;
}

RtsqBase::RtsqBase(u32 devPhyId, u32 streamId, u32 sqId) : devPhyId_(devPhyId), streamId_(streamId), sqId_(sqId)
{
}
void RtsqBase::Reset()
{
}

StreamLite::StreamLite(u32 id, u32 sqIds, u32 phyId, u32 cqIds) : id(id), sqId(sqIds), devPhyId(phyId), cqId(cqIds)
{
    rtsq = std::make_unique<RtsqBase>(phyId, id, sqIds);
}

StreamLite::StreamLite(u32 id, u32 sqIds, u32 phyId, u32 cqIds, bool launchFlag)
    : id(id),
      sqId(sqIds),
      devPhyId(phyId),
      cqId(cqIds)
{
    (void)launchFlag;
    rtsq = std::make_unique<RtsqBase>(phyId, id, sqIds);
}

RtsqBase *StreamLite::GetRtsq() const
{
    return rtsq.get();
}

u32 GetKernelExecTimeoutFromEnvConfig()
{
    return 0;
}

RdmaHandle RdmaHandleManager::GetByAddr(
    unsigned int, Hccl::LinkProtoType const &, Hccl::IpAddress &, Hccl::PortDeploymentType)
{
    return (void *)0x12345678;
}

std::vector<ModuleData> AicpuResPackageHelper::ParsePackedData(std::vector<char, std::allocator<char>> &) const
{
    std::vector<ModuleData> result;

    return result;
}

std::vector<char> AicpuResPackageHelper::GetPackedData(
    std::vector<Hccl::ModuleData, std::allocator<Hccl::ModuleData>> &) const
{
    std::vector<char> result;

    return result;
}

DevUbConnection::DevUbConnection(const RdmaHandle rdmaHandle, const IpAddress &locAddr, const IpAddress &rmtAddr,
    const OpMode opMode, const bool devUsed, const HrtUbJfcMode jfcMode, const IpAddress &locIpv4Addr,
    const IpAddress &rmtIpv4Addr, const u8 qos)
    : RmaConnection(nullptr, RmaConnType::UB),
      rdmaHandle(rdmaHandle),
      locAddr(locAddr),
      rmtAddr(rmtAddr),
      opMode(opMode),
      jfcMode(jfcMode),
      locIpv4Addr(locIpv4Addr),
      rmtIpv4Addr(rmtIpv4Addr),
      rmtEid(rmtAddr.GetReverseEid()),
      qos_(qos)
{
}

DevUbTpConnection::DevUbTpConnection(const RdmaHandle rdmaHandle, const IpAddress &locAddr, const IpAddress &rmtAddr,
    const OpMode opMode, const bool devUsed, const HrtUbJfcMode jfcMode, const IpAddress &locIpv4Addr,
    const IpAddress &rmtIpv4Addr, const u8 qos)
    : DevUbConnection(rdmaHandle, locAddr, rmtAddr, opMode, devUsed, jfcMode, locIpv4Addr, rmtIpv4Addr, qos)
{
}

DevUbCtpConnection::DevUbCtpConnection(const RdmaHandle rdmaHandle, const IpAddress &locAddr, const IpAddress &rmtAddr,
    const OpMode opMode, const bool devUsed, const HrtUbJfcMode jfcMode, const IpAddress &locIpv4Addr,
    const IpAddress &rmtIpv4Addr, const u8 qos)
    : DevUbConnection(rdmaHandle, locAddr, rmtAddr, opMode, devUsed, jfcMode, locIpv4Addr, rmtIpv4Addr, qos)
{
}

DevUbUboeConnection::DevUbUboeConnection(const RdmaHandle rdmaHandle, const IpAddress &locAddr,
    const IpAddress &rmtAddr, const OpMode opMode, const bool devUsed, const HrtUbJfcMode jfcMode,
    const IpAddress &locIpv4Addr, const IpAddress &rmtIpv4Addr, const u8 qos)
    : DevUbConnection(rdmaHandle, locAddr, rmtAddr, opMode, devUsed, jfcMode, locIpv4Addr, rmtIpv4Addr, qos)
{
    tpProtocol = TpProtocol::UBOE;
}

std::vector<char> DevUbConnection::GetUniqueId() const
{
    std::vector<char> result;

    return result;
}

void DevUbConnection::Connect()
{
}

inline uint32_t GetRandomNum()
{
    return 3;
}

RmaConnStatus DevUbConnection::GetStatus()
{
    return RmaConnStatus::READY;
}

std::unique_ptr<Serializable> DevUbConnection::GetExchangeDto()
{
    return nullptr;
}

void DevUbConnection::ParseRmtExchangeDto(const Serializable &rmtDto)
{
}

void DevUbConnection::ImportRmtDto()
{
}

void DevUbConnection::ThrowAbnormalStatus(std::string funcName)
{
}

bool DevUbConnection::CheckRequestResult()
{
    return true;
}

void DevUbConnection::CreateJetty(const bool devUsed)
{
}

void DevUbConnection::SetJettyInfo()
{
}

bool DevUbConnection::GetTpInfo()
{
    return true;
}

void DevUbConnection::GenerateLocalPsn()
{
}

void DevUbConnection::ImportJetty()
{
}

void DevUbConnection::SetImportInfo()
{
}

void DevUbConnection::ReleaseResource()
{
}

DevUbConnection::~DevUbConnection()
{
}

// Suspend接口当前已不使用，由框架调用触发析构流程
bool DevUbConnection::Suspend()
{
    return true;
}

static void PrepareUbSendWrReqParamForWriteOrRead(HrtRaUbSendWrReqParam &sendWrReq,
    const HrtUbSendWrOpCode sendWrOpCode, const MemoryBuffer &remoteMemBuf, const MemoryBuffer &localMemBuf,
    JettyHandle remoteJettyHandle, const SqeConfig &config, u32 cqeEnable = 1)
{
}

static void PrepareUbSendWrReqParamReduceInfo(HrtRaUbSendWrReqParam &sendWrReq, DataType dataType, ReduceOp reduceOp)
{
}

static void PrepareUbSendWrReqParamNotifyInfo(
    HrtRaUbSendWrReqParam &sendWrReq, u64 data, const MemoryBuffer &remoteNotifyMemBuf)
{
}

std::unique_ptr<BaseTask> DevUbConnection::ConstructTaskUbSend(
    const HrtRaUbSendWrRespParam &sendWrResp, const SqeConfig &config)
{
    return nullptr;
}

void DevUbConnection::ProcessSlices(const MemoryBuffer &loc, const MemoryBuffer &rmt,
    std::function<void(const MemoryBuffer &, const MemoryBuffer &, u32)> processOneSlice, DataType dataType) const
{
}

void DevUbConnection::ProcessSlicesWithNotify(const MemoryBuffer &loc, const MemoryBuffer &rmt,
    std::function<void(const MemoryBuffer &, const MemoryBuffer &, u32)> processOneSlice,
    std::function<void(const MemoryBuffer &, const MemoryBuffer &)> processOneSliceWithNotify, DataType dataType) const
{
}

unique_ptr<BaseTask> DevUbConnection::PrepareRead(
    const MemoryBuffer &remoteMemBuf, const MemoryBuffer &localMemBuf, const SqeConfig &config)
{
    return nullptr;
}

unique_ptr<BaseTask> DevUbConnection::PrepareReadReduce(const MemoryBuffer &remoteMemBuf,
    const MemoryBuffer &localMemBuf, DataType dataType, ReduceOp reduceOp, const SqeConfig &config)
{
    return nullptr;
}

unique_ptr<BaseTask> DevUbConnection::PrepareWrite(
    const MemoryBuffer &remoteMemBuf, const MemoryBuffer &localMemBuf, const SqeConfig &config)
{
    return nullptr;
}

unique_ptr<BaseTask> DevUbConnection::PrepareWriteReduce(const MemoryBuffer &remoteMemBuf,
    const MemoryBuffer &localMemBuf, DataType dataType, ReduceOp reduceOp, const SqeConfig &config)
{
    return nullptr;
}

unique_ptr<BaseTask> DevUbConnection::PrepareInlineWrite(
    const MemoryBuffer &remoteMemBuf, u64 data, const SqeConfig &config)
{
    return nullptr;
}

inline HrtRaUbSendWrReqParam ConstructUbSendWrReqParamForWriteWithNotify(
    const MemoryBuffer &remoteMemBuf, const MemoryBuffer &localMemBuf, u64 data, const MemoryBuffer &remoteNotifyMemBuf)
{
    HrtRaUbSendWrReqParam sendWrReq = {};

    return sendWrReq;
}

unique_ptr<BaseTask> DevUbConnection::PrepareWriteWithNotify(const MemoryBuffer &remoteMemBuf,
    const MemoryBuffer &localMemBuf, u64 data, const MemoryBuffer &remoteNotifyMemBuf, const SqeConfig &config)
{
    return nullptr;
}

unique_ptr<BaseTask> DevUbConnection::PrepareWriteReduceWithNotify(const MemoryBuffer &remoteMemBuf,
    const MemoryBuffer &localMemBuf, DataType dataType, ReduceOp reduceOp, u64 data,
    const MemoryBuffer &remoteNotifyMemBuf, const SqeConfig &config)
{
    return nullptr;
}

string DevUbConnection::Describe() const
{
    return "heelo";
}

void DevUbConnection::AddNop(const Stream &stream)
{
}

HrtUbJfcMode DevUbConnection::GetUbJfcMode() const
{
    return jfcMode;
}

u32 DevUbConnection::GetPiVal() const
{
    return 4;
}

u32 DevUbConnection::GetCiVal() const
{
    return 3;
}

u32 DevUbConnection::GetSqDepth() const
{
    return 2;
}

void DevUbConnection::UpdateCiVal(u32 ci)
{
}

std::vector<DevUbConnection *> GetStarsPollUbConns(const std::vector<RmaConnection *> &rmaConns)
{
    std::vector<DevUbConnection *> ubConns;
    return ubConns;
}

HcclResult DevUbConnection::Describe(std::string &dfxMsg)
{
    dfxMsg = "DevUbConnectionTest";
    return HCCL_SUCCESS;
}

bool IfNeedUpdatingUbCi(const std::vector<DevUbConnection *> &ubConns)
{
    return true;
}
RmaConnection::RmaConnection(Socket *socket, const RmaConnType rmaConnType) : socket(socket), rmaConnType(rmaConnType)
{
}

RmaConnection::~RmaConnection()
{
}

void RmaConnection::Close()
{
}

RmaConnStatus RmaConnection::GetStatus()
{
    return RmaConnStatus::READY;
}

void RmaConnection::Bind(RemoteRmaBuffer *remoteRmaBuf, BufferType bufType)
{
}

RemoteRmaBuffer *RmaConnection::GetRemoteRmaBuffer(const BufferType &bufType)
{
    return nullptr;
}

unique_ptr<BaseTask> RmaConnection::PrepareRead(
    const MemoryBuffer &remoteMemBuf, const MemoryBuffer &localMemBuf, const SqeConfig &config)
{
    return nullptr;
}

unique_ptr<BaseTask> RmaConnection::PrepareReadReduce(const MemoryBuffer &remoteMemBuf, const MemoryBuffer &localMemBuf,
    DataType datatype, ReduceOp reduceOp, const SqeConfig &config)
{
    return nullptr;
}

unique_ptr<BaseTask> RmaConnection::PrepareWrite(
    const MemoryBuffer &remoteMemBuf, const MemoryBuffer &localMemBuf, const SqeConfig &config)
{
    return nullptr;
}

unique_ptr<BaseTask> RmaConnection::PrepareWriteReduce(const MemoryBuffer &remoteMemBuf,
    const MemoryBuffer &localMemBuf, DataType datatype, ReduceOp reduceOp, const SqeConfig &config)
{
    return nullptr;
}

unique_ptr<BaseTask> RmaConnection::PrepareInlineWrite(
    const MemoryBuffer &remoteMemBuf, u64 data, const SqeConfig &config)
{
    return nullptr;
}

unique_ptr<BaseTask> RmaConnection::PrepareWriteWithNotify(const MemoryBuffer &remoteMemBuf,
    const MemoryBuffer &localMemBuf, u64 data, const MemoryBuffer &remoteNotifyMemBuf, const SqeConfig &config)
{
    return nullptr;
}

unique_ptr<BaseTask> RmaConnection::PrepareWriteReduceWithNotify(const MemoryBuffer &remoteMemBuf,
    const MemoryBuffer &localMemBuf, DataType datatype, ReduceOp reduceOp, u64 data,
    const MemoryBuffer &remoteNotifyMemBuf, const SqeConfig &config)
{
    return nullptr;
}
void HrtRaSocketWhiteListAdd(void *, std::vector<Hccl::RaSocketWhitelist, std::allocator<Hccl::RaSocketWhitelist>> &)
{
}
// HcclResult DpuNotifyManager::AllocNotifyIds(uint32_t notifyNum,
//     std::vector<uint32_t> &notifyIds)  // std::unique_ptr<uint64_t[]>& handles)
// {

//     return HCCL_SUCCESS;
// }
// DpuNotifyManager &DpuNotifyManager::GetInstance()
// {
//     int numNotify = 8192;
//     static DpuNotifyManager instance{numNotify};
//     return instance;
// }
// HcclResult DpuNotifyManager::FreeNotifyIds(uint32_t notifyNum, std::vector<uint32_t> &notifyIds)
// {

//     return HCCL_SUCCESS;
// }
// DpuNotifyManager::~DpuNotifyManager()
// {

// }
// DpuNotifyManager::DpuNotifyManager(int)
// {

// }
RemoteUbRmaBuffer::RemoteUbRmaBuffer(RdmaHandle rdmaHandle) : RemoteRmaBuffer(RmaType::UB), rdmaHandle(rdmaHandle)
{
}

RemoteUbRmaBuffer::RemoteUbRmaBuffer(RdmaHandle rdmaHandle1, const Serializable &rmtDto)
    : RemoteRmaBuffer(RmaType::UB),
      rdmaHandle(rdmaHandle1)
{ // 从 DTO 取得数据，然后生成 memHandle
}

RemoteIpcRmaBuffer::RemoteIpcRmaBuffer() : RemoteRmaBuffer(RmaType::IPC), isOpened(true)
{
}

RemoteIpcRmaBuffer::RemoteIpcRmaBuffer(const Serializable &rmtDto) : RemoteRmaBuffer(RmaType::IPC), isOpened(true)
{
}

RemoteIpcRmaBuffer::RemoteIpcRmaBuffer(const Serializable &rmtDto, const string tag)
    : RemoteRmaBuffer(RmaType::IPC),
      isOpened(true)
{
}

void RemoteIpcRmaBuffer::Close() const
{
}

RemoteIpcRmaBuffer::~RemoteIpcRmaBuffer()
{
}

string RemoteIpcRmaBuffer::Describe() const
{
    return "hello";
}

RemoteRdmaRmaBuffer::RemoteRdmaRmaBuffer(RdmaHandle rdmaHandle)
    : RemoteRmaBuffer(RmaType::RDMA),
      rdmaHandle(rdmaHandle),
      keyValidLen(RDMA_MEM_KEY_LEN_ROCE)
{
}

RemoteRdmaRmaBuffer::RemoteRdmaRmaBuffer(RdmaHandle rdmaHandle, const Serializable &rmtDto)
    : RemoteRmaBuffer(RmaType::RDMA),
      rdmaHandle(rdmaHandle)
{
}

RemoteRdmaRmaBuffer::~RemoteRdmaRmaBuffer()
{
    // 待修改:  使用rdmaHandle调用 HCCP 新接口 unimport 接口，销毁key
}

string RemoteRdmaRmaBuffer::Describe() const
{
    return "hello";
}

RemoteUbRmaBuffer::~RemoteUbRmaBuffer()
{
}

string RemoteUbRmaBuffer::Describe() const
{
    return "hello";
}

SocketHandle SocketHandleManager::Get(unsigned int, Hccl::PortData const &)
{
    return (void *)0x12345678;
}
UbLocalNotify::UbLocalNotify(RdmaHandle rdmaHandle, bool devUsed)
    : BaseLocalNotify(RmaType::UB, devUsed),
      rdmaHandle(rdmaHandle)
{
}

string UbLocalNotify::Describe() const
{
    return "hello";
}

void UbLocalNotify::Wait(const Stream &stream, u32 timeout) const
{
}

void UbLocalNotify::Post(const Stream &stream) const
{
}

std::unique_ptr<Serializable> UbLocalNotify::GetExchangeDto()
{
    return nullptr;
}

void UbLocalNotify::ReleaseResource() const
{
}

UbLocalNotify::~UbLocalNotify()
{
}

// RtsNotify::RtsNotify(bool devUsed)
// {}
// RtsNotify::~RtsNotify()
// {}

UbMemTransport::UbMemTransport(CommonLocRes &commonLocRes, Attribution &attr, const LinkData &linkData,
    const Socket &socket, RdmaHandle rdmaHandle1, LocCntNotifyRes &locCntNotifyRes1, bool isRecvFirst)
    : BaseMemTransport(commonLocRes, attr, linkData, socket, TransportType::UB),
      rdmaHandle(rdmaHandle1),
      locCntNotifyRes(locCntNotifyRes1),
      isRecvFirst_(isRecvFirst)
{
}

UbMemTransport::UbMemTransport(CommonLocRes &commonLocRes, Attribution &attr, const LinkData &linkData,
    const Socket &socket, RdmaHandle rdmaHandle1, LocCntNotifyRes &locCntNotifyRes1,
    std::function<void(u32 streamId, u32 taskId, const TaskParam &taskParam)> callback)
    : BaseMemTransport(commonLocRes, attr, linkData, socket, TransportType::UB, callback),
      rdmaHandle(rdmaHandle1),
      locCntNotifyRes(locCntNotifyRes1)
{
}

HcclResult UbMemTransport::FillTagVec(
    std::vector<LocalRmaBuffer *> &bufferVec, std::vector<std::array<char, HCCL_RES_TAG_MAX_LEN>> &localUserMemTag)
{
    return HCCL_SUCCESS;
}

std::string UbMemTransport::Describe() const
{
    return "msg";
}

static void SubmitTask(const TaskUbDbSend &ubSend, const Stream &stream)
{
}

static void SubmitTask(const TaskUbDirectSend &ubDirectSend, const Stream &stream)
{
}

static void SubmitTask(const TaskWriteValue &taskWriteValue, const Stream &stream)
{
}

static void SubmitUbTask(unique_ptr<BaseTask> task, const Stream &stream)
{
}

void UbMemTransport::SubmitNotify(const MemoryBuffer &rmtNotify, u64 data, const Stream &stream)
{
}

void UbMemTransport::Post(u32 index, const Stream &stream)
{
}

void UbMemTransport::Wait(u32 index, const Stream &stream, u32 timeout)
{
}

void UbMemTransport::Read(const RmaBufferSlice &locSlice, const RmtRmaBufferSlice &rmtSlice, const Stream &stream)
{
}

void UbMemTransport::ReadReduce(
    const RmaBufferSlice &locSlice, const RmtRmaBufferSlice &rmtSlice, const ReduceIn &reduceIn, const Stream &stream)
{
}

void UbMemTransport::Write(const RmaBufferSlice &locSlice, const RmtRmaBufferSlice &rmtSlice, const Stream &stream)
{
}

void UbMemTransport::WriteReduce(
    const RmaBufferSlice &locSlice, const RmtRmaBufferSlice &rmtSlice, const ReduceIn &reduceIn, const Stream &stream)
{
}

void UbMemTransport::WriteWithNotify(const RmaBufferSlice &locSlice, const RmtRmaBufferSlice &rmtSlice,
    const WithNotifyIn &withNotify, const Stream &stream)
{
}

void UbMemTransport::WriteReduceWithNotify(const RmaBufferSlice &locSlice, const RmtRmaBufferSlice &rmtSlice,
    const ReduceIn &reduceIn, const WithNotifyIn &withNotify, const Stream &stream)
{
}

void UbMemTransport::SubmitWriteEmptyWithNotify(const WithNotifyIn &withNotify, const Stream &stream)
{
}

void UbMemTransport::SubmitWriteWithNotify(
    const MemoryBuffer &rmt, const MemoryBuffer &loc, u64 data, const MemoryBuffer &rmtNotify, const Stream &stream)
{
}

void UbMemTransport::SubmitWriteReduceWithNotify(const MemoryBuffer &rmt, const MemoryBuffer &loc,
    const ReduceIn &reduceIn, u64 data, const MemoryBuffer &rmtNotify, const Stream &stream)
{
}

bool UbMemTransport::IsResReady()
{
    return true;
}

bool UbMemTransport::IsConnsReady()
{
    return true;
}

TransportStatus UbMemTransport::GetStatus()
{
    return TransportStatus::READY;
}

void UbMemTransport::SendExchangeData()
{
}

void UbMemTransport::RecvExchangeData()
{
}

bool UbMemTransport::RecvDataProcess()
{
    BinaryStream binaryStream(recvData);

    return ConnVecUnpackProc(binaryStream);
}

void UbMemTransport::BufferVecPack(BinaryStream &binaryStream, std::vector<LocalRmaBuffer *> &bufferVec,
    std::vector<std::array<char, HCCL_RES_TAG_MAX_LEN>> &localUserMemTag)
{
}

void UbMemTransport::CntNotifyVecPack(BinaryStream &binaryStream)
{
}

void UbMemTransport::CntNotifyDescPack(BinaryStream &binaryStream)
{
}
void UbMemTransport::CntNotifyDescUnpack(BinaryStream &binaryStream)
{
}

void UbMemTransport::RmtBufferVecUnpackProc(
    u32 locNum, BinaryStream &binaryStream, RemoteBufferVec &bufferVec, UbRmtBufType type)
{
}

bool UbMemTransport::ConnVecUnpackProc(BinaryStream &binaryStream)
{
    bool result = true; // 不需要发送 finish

    return result;
}

void UbMemTransport::FillRmtRmaBufferVec(RemoteRmaBuffer *rmaBuffer, UbRmtBufType type)
{
}

void UbMemTransport::SendFinish()
{
}

void UbMemTransport::RecvFinish()
{
}

std::vector<char> UbMemTransport::GetUniqueId()
{
    std::vector<char> result;

    return result;
}

std::vector<char> UbMemTransport::GetUniqueIdV2()
{
    std::vector<char> result;

    return result;
}

std::vector<char> UbMemTransport::GetSingleRmtBufferUniqueId(u64 addr, u64 size, u32 tokenId, u32 tokenValue) const
{
    std::vector<char> result;

    return result;
}

std::vector<char> UbMemTransport::GetNotifyUniqueIds()
{
    std::vector<char> result(0);

    return result;
}

std::vector<char> UbMemTransport::GetRmtBufferUniqueIds(RemoteBufferVec &bufferVec, UbRmtBufType type) const
{
    std::vector<char> result(0);

    return result;
}

std::vector<char> UbMemTransport::GetLocBufferUniqueIds(LocalBufferVec &bufferVec, UbRmtBufType type) const
{
    std::vector<char> result(0);

    return result;
}

std::vector<char> UbMemTransport::GetConnUniqueIds()
{
    std::vector<char> result(0);

    return result;
}

void UbMemTransport::SaveDfxTaskInfo(const TaskParam &taskParam)
{
}

HcclResult UbMemTransport::GetRemoteMem(HcclMem **remoteMem, uint32_t *memNum, char **memTags)
{
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::GetUserRemoteMem(CommMem **remoteMem, char ***memTags, uint32_t *memNum)
{
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::CheckSocketStatus(std::string socketOpreator)
{
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::UpdateMemInfo(std::vector<LocalRmaBuffer *> &bufferVecTemp)
{
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::Init()
{
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::DeInit() const
{
    return HCCL_SUCCESS;
}

HcclResult UbMemTransport::Describe(std::string &dfxMsg)
{
    dfxMsg = "UbMemTransportTest";
}

HcclResult UbMemTransport::GetRemoteSeg(const void* addr, u64 len, u64 *seg)
{
    return HCCL_SUCCESS;
}

LocalRdmaRmaBufferMgr *LocalRdmaRmaBufferManager::GetInstance()
{
    LocalRdmaRmaBufferMgr instance;
    return &instance;
}

LocalUbRmaBufferMgr *LocalUbRmaBufferManager::GetInstance()
{
    LocalUbRmaBufferMgr instance;
    return &instance;
}

LocalRdmaRmaBuffer::LocalRdmaRmaBuffer(std::shared_ptr<Buffer> buf, RdmaHandle rdmaHandle)
    : LocalRmaBuffer(buf, RmaType::RDMA),
      rdmaHandle(rdmaHandle)
{
}

LocalRdmaRmaBuffer::~LocalRdmaRmaBuffer()
{
}

string LocalRdmaRmaBuffer::Describe() const
{
    return "hello";
}

std::unique_ptr<Serializable> LocalRdmaRmaBuffer::GetExchangeDto()
{
    return nullptr;
}

BaseMemTransport::BaseMemTransport(
    CommonLocRes &commonLocRes, Attribution &attr, const LinkData &linkData, const Socket &socket, TransportType type)
    : commonLocRes(commonLocRes),
      attr(attr),
      linkData(linkData),
      socket(const_cast<Socket *>(&socket)),
      transportType(type)
{
}

BaseMemTransport::BaseMemTransport(CommonLocRes &commonLocRes, Attribution &attr, const LinkData &linkData,
    const Socket &socket, TransportType type,
    std::function<void(u32 streamId, u32 taskId, TaskParam taskParam)> callback)
    : commonLocRes(commonLocRes),
      attr(attr),
      linkData(linkData),
      socket(const_cast<Socket *>(&socket)),
      transportType(type),
      callback(callback)
{
}

void BaseMemTransport::Establish()
{
}

void BaseMemTransport::SetBaseStatusReady()
{
}

bool BaseMemTransport::IsSocketReady()
{
    return true;
}

void BaseMemTransport::NotifyVecPack(BinaryStream &binaryStream)
{
}

void BaseMemTransport::ConnVecPack(BinaryStream &binaryStream)
{
}

void BaseMemTransport::HandshakeMsgPack(BinaryStream &binaryStream)
{
}

void BaseMemTransport::HandshakeMsgUnpack(BinaryStream &binaryStream)
{
}

string BaseMemTransport::GetLinkDescInfo()
{
    return "hello";
}

string BaseMemTransport::DescribeSocket() const
{
    return "hello";
}

void BaseMemTransport::CheckLocNotify(CommonLocRes &res)
{
}

void BaseMemTransport::CheckLocBuffer(CommonLocRes &res)
{
}

void BaseMemTransport::CheckLocConn(CommonLocRes &res)
{
}

void BaseMemTransport::CheckCommonLocRes(CommonLocRes &res)
{
}
HcclResult HrtRaCreateQpWithCq(
    RdmaHandle rdmaHandle, s32 sqEvent, s32 rqEvent, void *sendChannel, void *recvChannel, QpInfo &info, bool isHdcMode)
{
    return HCCL_SUCCESS;
}

HcclResult HrtRaDestroyQpWithCq(const QpInfo &info, bool isHdcMode)
{
    return HCCL_SUCCESS;
}

HccpHdcManager &HccpHdcManager::GetInstance()
{
    static HccpHdcManager hccpHdcManager;
    return hccpHdcManager;
}

void HccpHdcManager::Init(u32 deviceLogicId)
{
}

HccpHdcManager::~HccpHdcManager()
{
}

LocalUbRmaBuffer::LocalUbRmaBuffer(std::shared_ptr<Buffer> buf, void *netDevice, bool flag)
    : LocalRmaBuffer(buf, RmaType::UB)
{
}

LocalIpcRmaBuffer::LocalIpcRmaBuffer(std::shared_ptr<Buffer> buf) : LocalRmaBuffer(buf, RmaType::IPC)
{
}

LocalIpcRmaBuffer::~LocalIpcRmaBuffer()
{
}

string LocalIpcRmaBuffer::Describe() const
{
    return "";
}

std::unique_ptr<Serializable> LocalIpcRmaBuffer::GetExchangeDto()
{
    return nullptr;
}

void LocalIpcRmaBuffer::Grant(u32 pid)
{
}

SocketManager::SocketManager(u32 localRank, u32 devicePhyId, u32 deviceLogicId, const std::string &socketTag)
{
}

void SocketManager::BatchCreateSockets(const vector<LinkData> &links)
{
}

void SocketManager::ServerListen(const SocketConfig &socketConfig)
{
}

void SocketManager::ConnectSockets(const SocketConfig &socketConfig)
{
}


void SocketManager::BatchCreateSockets(const SocketConfig &socketConfig)
{
}

SocketManager::~SocketManager()
{
}

Socket *SocketManager::GetConnectedSocket(const SocketConfig &socketConfig) const
{
    return nullptr;
}

bool SocketManager::CheckServerPortListening(const PortData &portData, const uint32_t port) const
{
    return true;
}

bool SocketManager::ServerDeInit(PortData &portData) const
{
    return true;
}

void SocketManager::SetDeviceServerListenPortMap(const std::unordered_map<u32, std::unordered_map<IpAddress, u32>> &rankListenPortMap)
{
    return;
}

HccpTlvHdcManager &HccpTlvHdcManager::GetInstance()
{
    static HccpTlvHdcManager HccpTlvHdcManager;
    return HccpTlvHdcManager;
}

HccpTlvHdcManager::HccpTlvHdcManager()
{
}

HccpTlvHdcManager::~HccpTlvHdcManager()
{
}

CcuComponent &CcuComponent::GetInstance(const int32_t deviceLogicId)
{
    static CcuComponent ccuComponent;
    return ccuComponent;
}

CcuComponent::~CcuComponent()
{
}

CcuResSpecifications &CcuResSpecifications::GetInstance(const int32_t deviceLogicId)
{
    static CcuResSpecifications ccuResSpecification;
    return ccuResSpecification;
}

HcclResult CcuResSpecifications::GetGsaNum(const uint8_t dieId, uint32_t &gsaNum) const
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetInstructionNum(const uint8_t dieId, uint32_t &instrNum) const
{
    return HcclResult::HCCL_SUCCESS;
}

HccpPeerManager &HccpPeerManager::GetInstance()
{
    static HccpPeerManager hccpPeerManager;
    return hccpPeerManager;
}

HccpPeerManager::~HccpPeerManager()
{
}

void HccpPeerManager::Init(s32 deviceLogicId)
{
}

HcclResult CcuResSpecifications::GetXnNum(const uint8_t dieId, uint32_t &xnNum) const
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetLoopEngineNum(const uint8_t dieId, uint32_t &loopNum) const
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetMissionNum(const uint8_t dieId, uint32_t &missionNum) const
{
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuResSpecifications::GetMsNum(const uint8_t dieId, uint32_t &msNum) const
{
    return HcclResult::HCCL_SUCCESS;
}

CtxMgrImp &CtxMgrImp::GetInstance(s32 deviceLogicId)
{
    static CtxMgrImp contextManager;
    return contextManager;
}

CtxMgrImp::CtxMgrImp()
{
}

CtxMgrImp::~CtxMgrImp()
{
}

InnerNetDevManager &InnerNetDevManager::GetInstance()
{
    static InnerNetDevManager instance;
    return instance;
}

InnerNetDev::~InnerNetDev()
{
}

CcuResBatchAllocator &CcuResBatchAllocator::GetInstance(const int32_t deviceLogicId)
{
    static CcuResBatchAllocator ccuResBatchAllocator;
    return ccuResBatchAllocator;
}

HcclResult CcuResSpecifications::GetCkeNum(const uint8_t dieId, uint32_t &ckeNum) const
{
    return HcclResult::HCCL_SUCCESS;
}

TpManager &TpManager::GetInstance(const int32_t deviceLogicId)
{
    static TpManager tpManager;
    return tpManager;
}

void TpManager::Init()
{
}

TaskInfo::TaskInfo(
    u32 streamId, u32 taskId, u32 remoteRank, TaskParam taskParam, std::shared_ptr<DfxOpInfo> dfxOpInfo, bool isMaster)
    : streamId_(streamId),
      taskId_(taskId),
      remoteRank_(remoteRank),
      taskParam_(taskParam),
      dfxOpInfo_(dfxOpInfo),
      isMaster_(isMaster)
{
}

std::string TaskInfo::Describe() const
{
    return "";
}

std::string TaskInfo::GetBaseInfo() const
{
    return "";
}

std::string TaskInfo::GetConciseBaseInfo() const
{
    return "";
}

GlobalMirrorTasks GlobalMirrorTasks::ins_;

GlobalMirrorTasks::GlobalMirrorTasks()
{
}

GlobalMirrorTasks::~GlobalMirrorTasks()
{
}

GlobalMirrorTasks &GlobalMirrorTasks::Instance()
{
    static GlobalMirrorTasks instance;
    return instance;
}

u32 GlobalMirrorTasks::DevSize() const
{
    return 0;
}

TaskInfoQueue &GlobalMirrorTasks::CreateQueue(u32 devId, u32 streamId, QueueType type)
{
    static CircularQueue<std::shared_ptr<TaskInfo>> queue(MAX_CIRCULAR_QUEUE_LENGTH);
    return queue;
}

TaskInfoQueue *GlobalMirrorTasks::GetQueue(u32 devId, u32 streamId) const
{
    static CircularQueue<std::shared_ptr<TaskInfo>> queue(MAX_CIRCULAR_QUEUE_LENGTH);
    return &queue;
}

void GlobalMirrorTasks::DestroyQueue(u32 devId, u32 streamId)
{
}

TaskInfoQueueMap::iterator GlobalMirrorTasks::Begin(u32 devId)
{
    static TaskInfoQueueMap map;
    return map.begin();
}

TaskInfoQueueMap::iterator GlobalMirrorTasks::End(u32 devId)
{
    static TaskInfoQueueMap map;
    return map.end();
}

std::shared_ptr<TaskInfo> GlobalMirrorTasks::GetTaskInfo(u32 devId, u32 streamId, u32 taskId) const
{
    return nullptr;
}

HcclResult GlobalMirrorTasks::FindTaskInfo(
    u32 devId, u32 streamId, u32 taskId, std::shared_ptr<TaskInfo> &curTask) const
{
    return HCCL_E_NOT_FOUND;
}

MirrorTaskManager::MirrorTaskManager(u32 devId, GlobalMirrorTasks *globalMirrorTasks, bool devUsed)
    : devId_(devId),
      globalMirrorTasks_(globalMirrorTasks),
      devUsed_(devUsed)
{
}

void MirrorTaskManager::RegFullyCallBack(std::function<void(const std::string &, u32)> callBack)
{
}

void MirrorTaskManager::RegFullyCallBack(std::function<void()> callBack)
{
}

void MirrorTaskManager::AddTaskInfo(std::shared_ptr<TaskInfo> taskInfo)
{
}

bool MirrorTaskManager::IsStaticGraphMode(const CollOperator &collOperator) const
{
    return false;
}

void MirrorTaskManager::SetCurrDfxOpInfo(std::shared_ptr<DfxOpInfo> dfxOpInfo)
{
}

std::shared_ptr<DfxOpInfo> MirrorTaskManager::GetCurrDfxOpInfo() const
{
    return nullptr;
}

TaskInfoQueue *MirrorTaskManager::GetQueue(u32 streamId) const
{
}

std::unordered_map<u32, TaskInfoQueue *>::iterator MirrorTaskManager::Begin()
{
    static std::unordered_map<u32, TaskInfoQueue *> queueMap;
    return queueMap.begin();
}

std::unordered_map<u32, TaskInfoQueue *>::iterator MirrorTaskManager::End()
{
    static std::unordered_map<u32, TaskInfoQueue *> queueMap;
    return queueMap.end();
}

MirrorTaskManager::~MirrorTaskManager()
{
}

MirrorTaskManagerLite::MirrorTaskManagerLite()
{
}

void MirrorTaskManagerLite::RegFullyCallBack(std::function<void(const std::string &, u32)> callBack)
{
}

void MirrorTaskManagerLite::RegFullyCallBack(std::function<void()> callBack)
{
}

void MirrorTaskManagerLite::AddTaskInfo(std::shared_ptr<TaskInfo> taskInfo)
{
}

bool MirrorTaskManagerLite::IsStaticGraphMode(const CollOperator &collOperator) const
{
    return false;
}

void MirrorTaskManagerLite::SetCurrDfxOpInfo(std::shared_ptr<DfxOpInfo> dfxOpInfo)
{
}

std::shared_ptr<DfxOpInfo> MirrorTaskManagerLite::GetCurrDfxOpInfo() const
{
    return nullptr;
}

TaskInfoQueue *MirrorTaskManagerLite::GetQueue(u32 streamId) const
{
}

MirrorTaskManagerLite::~MirrorTaskManagerLite()
{
}
ProfilingHandler::ProfilingHandler()
{
}

ProfilingHandler::~ProfilingHandler()
{
}

std::string TaskInfo::GetParaInfo() const
{
    return "";
}

std::string TaskInfo::GetOpInfo() const
{
    return "";
}

std::string TaskInfo::GetIndopDataInfo() const
{
    return "";
}

std::string TaskInfo::GetIndopBaseInfo() const
{
    return "";
}

ProfilingHandler &ProfilingHandler::GetInstance()
{
    static ProfilingHandler instance;
    return instance;
}

void ProfilingHandler::Init()
{
}

// 回调注册
int32_t ProfilingHandler::CommandHandleWrapper(uint32_t rtType, void *data, uint32_t len)
{
    return 0;
}

// 接口预留，暂时不实现
// 函数入参，因为静态检查先删除注释：kernelType kerType, uint64_t beginTime, uint64_t endTime, bool cachedReq
void ProfilingHandler::ReportKernel() const
{
}

void ProfilingHandler::ReportHostApi(OpType opType, uint64_t beginTime, uint64_t endTime, bool cachedReq, bool isAiCpu)
{
}

void ProfilingHandler::ReportHcclOp(const DfxOpInfo &opInfo, bool cachedReq)
{
}

void ProfilingHandler::ReportHcclTaskApi(
    TaskParamType taskType, uint64_t beginTime, uint64_t endTime, bool isMasterStream, bool cachedReq, bool ignoreLevel)
{
}

void ProfilingHandler::ReportHcclTaskDetails(const TaskInfo &taskInfo, bool cachedReq)
{
}

void ProfilingHandler::CallAdditionInfo(HCCLReportData &hcclReportData, void *data, u32 len, ProfTaskType type) const
{
}

void ProfilingHandler::GetHCCLReportData(const TaskInfo &taskInfo, HCCLReportData &hcclReportData) const
{
}

void ProfilingHandler::DumpHCCLReportData(const TaskInfo &taskInfo, const HCCLReportData &hcclReportData) const
{
}

void ProfilingHandler::ReportCcuInfo(const TaskInfo &taskInfo) const
{
}

void ProfilingHandler::GetCcuTaskInfo(const TaskInfo &taskInfo, const CcuProfilingInfo &info) const
{
}

void ProfilingHandler::GetCcuGroupInfo(const TaskInfo &taskInfo, const CcuProfilingInfo &info) const
{
}

void ProfilingHandler::DumpCcuGroupInfo(const MsprofCcuGroupInfo &ccuGroupInfo) const
{
}

void ProfilingHandler::GetCcuWaitSignalInfo(const TaskInfo &taskInfo, const CcuProfilingInfo &info) const
{
}

void ProfilingHandler::ReportAclApi(
    uint32_t cmdType, uint64_t beginTime, uint64_t endTime, uint64_t cmdItemId, uint32_t threadId) const
{
}

void ProfilingHandler::ReportNodeApi(uint64_t beginTime, uint64_t endTime, uint64_t cmdItemId, uint32_t threadId, bool cachedReq)
{
    (void)cachedReq;
}

void ProfilingHandler::ReportNodeBasicInfo(uint64_t timeStamp, uint64_t cmdItemId, uint32_t threadId, bool cachedReq)
{
    (void)cachedReq;
}

void ProfilingHandler::ReportHcclOpInfo(uint64_t timeStamp, const DfxOpInfo &opInfo, uint32_t threadId, bool cachedReq)
{
    (void)cachedReq;
}

void ProfilingHandler::ReportAdditionInfo(uint32_t type, uint64_t timeStamp, void *data, uint32_t len) const
{
}

int32_t ProfilingHandler::CommandHandle(uint32_t rtType, void *data, uint32_t len) const
{
    return 0;
}

void ProfilingHandler::StartSubscribe(uint64_t profconfig)
{
}

void ProfilingHandler::StartHostApiSubscribe()
{
}

void ProfilingHandler::CallProfRegHostApi() const
{
}

void ProfilingHandler::ReportStoragedCompactInfo()
{
}

void ProfilingHandler::ReportMc2AdditionInfo()
{
}

void ProfilingHandler::StartTaskApiSubscribe()
{
}

void ProfilingHandler::CallProfRegTaskTypeApi() const
{
}

void ProfilingHandler::ReportStoragedTaskApi()
{
}

void ProfilingHandler::StartHostHcclOpSubscribe()
{
}

void ProfilingHandler::CallProfRegHcclOpApi() const
{
}

void ProfilingHandler::StartAdditionInfoSubscribe()
{
}

void ProfilingHandler::ReportStoragedAdditionInfo()
{
}

void ProfilingHandler::StartCcuSubscribe()
{
}

void ProfilingHandler::ProfilingHandler::StopSubscribe()
{
}

bool ProfilingHandler::GetHostApiState() const
{
    return false;
}
bool ProfilingHandler::GetHcclNodeState() const
{
    return false;
}
bool ProfilingHandler::GetHcclL0State() const
{
    return false;
}

bool ProfilingHandler::GetHcclL1State() const
{
    return false;
}

uint64_t ProfilingHandler::GetProfHashId(const char *name, uint32_t len) const
{
    return 0;
}

void ProfilingHandler::ReportHcclMC2CommInfo(const Stream &kfcStream, Stream &stream,
    const std::vector<Stream *> &aicpuStreams, const std::string &id, RankId myRank, u32 rankSize,
    RankId rankInParentComm)
{
}

void ProfilingHandler::ReportHcclMC2CommInfo(const u32 kfcStreamId, const std::vector<u32> &aicpuStreamsId,
    const std::string &id, RankId myRank, u32 rankSize, RankId rankInParentComm)
{
}
void ProfilingHandler::ReportMc2AdditionInfo(uint64_t timeStamp, const void *data, int len)
{
}
ProfilingHandlerLite ProfilingHandlerLite::instance_;

ProfilingHandlerLite::ProfilingHandlerLite()
{
}

ProfilingHandlerLite::~ProfilingHandlerLite()
{
}

ProfilingHandlerLite &ProfilingHandlerLite::GetInstance()
{
    static ProfilingHandlerLite instance;
    return instance;
}

void ProfilingHandlerLite::Init() const
{
}

void ProfilingHandlerLite::ReportHcclOpInfo(const DfxOpInfo &opInfo) const
{
}

void ProfilingHandlerLite::ReportHcclTaskDetails(const std::vector<TaskInfo> &taskInfo) const
{
}

void ProfilingHandlerLite::GetTaskDetailInfos(const TaskInfo &it, MsprofAicpuHcclTaskInfo &taskDetailsInfos) const
{
}

void ProfilingHandlerLite::DumpTaskDetails(
    const MsprofAicpuHcclTaskInfo &taskDetailsInfos, const TaskInfo &taskInfo) const
{
}

void ProfilingHandlerLite::ReportMainStreamTask(const FlagTaskInfo &flagTaskInfo) const
{
}

void ProfilingHandlerLite::ReportAdditionInfo(uint32_t type, uint64_t timeStamp, const void *data, int len) const
{
}

void ProfilingHandlerLite::UpdateProfSwitch()
{
}

bool ProfilingHandlerLite::IsProfOn(uint64_t feature) const
{
    return false;
}

bool ProfilingHandlerLite::IsProfSwitchOn(ProfilingLevel level)
{
    return false;
}

bool ProfilingHandlerLite::IsL1fromOffToOn()
{
    return false;
}

void ProfilingHandlerLite::SetProL1On(bool val)
{
}

void ProfilingHandlerLite::SetProL0On(bool val)
{
}

uint64_t ProfilingHandlerLite::GetProfHashId(const char *name, uint32_t len) const
{
    return 0;
}

ProfilingReporter::ProfilingReporter(MirrorTaskManager *mirrorTaskMgr, ProfilingHandler *profilingHandler)
    : mirrorTaskMgr_(mirrorTaskMgr),
      profilingHandler_(profilingHandler)
{
}

ProfilingReporter::~ProfilingReporter()
{
}

void ProfilingReporter::Init() const
{
}

void ProfilingReporter::ReportOp(uint64_t beginTime, bool cachedReq, bool opbased) const
{
}

void ProfilingReporter::ReportCallBackAllTasks(bool cachedReq)
{
}

void ProfilingReporter::ReportAllTasks(bool cachedReq) const
{
}

/* 中途打开profiling开关 */
void ProfilingReporter::UpdateProfStat()
{
}

void ProfilingReporter::CallReportMc2CommInfo(const Stream &kfcStream, Stream &stream,
    const std::vector<Stream *> &aicpuStreams, const std::string &id, RankId myRank, u32 rankSize,
    RankId rankInParentComm) const
{
}

void ProfilingReporter::CallReportMc2CommInfo(const u32 kfcStreamId, const std::vector<u32> &aicpuStreamsId,
    const std::string &id, RankId myRank, u32 rankSize, RankId rankInParentComm) const
{
}

std::array<ProfilingReporter::lastPosesMap, 65> ProfilingReporter::allLastPoses_{};

ProfilingReporterLite::ProfilingReporterLite(
    MirrorTaskManagerLite *mirrorTaskMgrLite, ProfilingHandlerLite *profilingHandlerLite, bool isIndop)
    : mirrorTaskMgrLite_(mirrorTaskMgrLite),
      profilingHandlerLite_(profilingHandlerLite)
{
}

ProfilingReporterLite::~ProfilingReporterLite()
{
}

void ProfilingReporterLite::Init() const
{
}

void ProfilingReporterLite::ReportAllTasks()
{
}

void ProfilingReporterLite::UpdateProfStat() const
{
}

DlHalFunctionV2::DlHalFunctionV2() : handle_(nullptr)
{
}

DlHalFunctionV2::~DlHalFunctionV2()
{
}

DlHalFunctionV2 &DlHalFunctionV2::GetInstance()
{
    static DlHalFunctionV2 instance;
    return instance;
}

HcclResult DlHalFunctionV2::DlHalFunctionInit()
{
    return HCCL_SUCCESS;
}

HcclResult DlHalFunctionV2::DlHalFunctionEschedInit()
{
    return HCCL_SUCCESS;
}

DlProfFunction::DlProfFunction() : handle_(nullptr)
{
    DlProfFunctionStubInit();
}

DlProfFunction::~DlProfFunction()
{
}

static uint64_t MsprofSysCycleTimeStub()
{
    HCCL_WARNING("Entry MsprofSysCycleTimeStub");
    return 0;
}

void DlProfFunction::DlProfFunctionStubInit()
{
    dlMsprofSysCycleTime = static_cast<uint64_t (*)(void)>(MsprofSysCycleTimeStub);
}
DlProfFunction &DlProfFunction::GetInstance()
{
    static DlProfFunction instance;
    return instance;
}

HcclResult DlProfFunction::DlProfFunctionInit()
{
    return HCCL_SUCCESS;
}

HcclResult DlProfFunction::DlProfFunctionInterInit()
{
    return HCCL_SUCCESS;
}

AicpuDaemonService &AicpuDaemonService::GetInstance()
{
    static AicpuDaemonService instance;
    return instance;
}

void AicpuDaemonService::ServiceRun(void *info)
{
}

void AicpuDaemonService::ServiceStop(void *info) const
{
}

void AicpuDaemonService::Register(DaemonFunc *daemonFunc)
{
}

void AicpuDaemonService::Break()
{
}

std::mutex AicpuDaemonService::mutexForFuncs_;

void TaskExceptionHandler::Process(rtExceptionInfo_t *expectionInfo)
{
}

void TaskExceptionHandler::PrintAicpuErrorMessage(rtExceptionInfo_t *expectionInfo, bool &isExistAicpuError)
{
}

std::array<TaskExceptionHandler *, 65> TaskExceptionHandlerManager::handlers_;

HcclResult RaGetAuxInfo(const RdmaHandle rdmaHandle, AuxInfoIn auxInfoIn, AuxInfoOut &auxInfoOut)
{
    return HCCL_SUCCESS;
}

void HrtRaSocketGetVnicIpInfos(u32 phyId, DeviceIdType deviceIdType, u32 deviceId, IpAddress &vnicIP)
{
}

extern "C" {
aclError aclrtSetExceptionInfoCallback(aclrtExceptionInfoCallback callback)
{
    return ACL_ERROR_NONE;
}
}

u32 Hccl::HcclCommunicator::GetRankInParentComm()
{
    return 0;
}

HcclResult CcuCleanTaskKillState(const int32_t deviceLogicId)
{
    return HCCL_SUCCESS;
}

uint16_t CcuRep::ParseRepeatNumFromParallelParam(uint64_t parallelParam)
{
    return 3;
}

HcclResult GetCcuErrorMsg(
    s32 deviceId, uint16_t status, const ParaCcu &ccuTaskParam, std::vector<CcuErrorInfo> &errorInfo)
{
    return HCCL_SUCCESS;
}

HcclResult GetCcuJettys(s32 deviceLogicId, const ParaCcu &ccuTaskParam, std::vector<CcuJetty *> &ccuJettys)
{
    return HCCL_SUCCESS;
}

HcclResult RaBatchQueryJettyStatus(
    const std::vector<JettyHandle> &jettyHandles, std::vector<JettyStatus> &jettyAttrs, u32 &num)
{
    return HCCL_SUCCESS;
}
void HrtGetTaskIdAndStreamID(u32 &taskId, u32 &streamId)
{
}

HcclResult CcuCleanDieCkes(const int32_t deviceLogicId, const uint8_t dieId)
{
    return HCCL_SUCCESS;
}

std::string CollOpToString(const BaseCollOperator &collOp)
{
    return "collOp";
}

std::shared_ptr<TaskInfo> MirrorTaskManagerLite::GetTaskInfo(u32 streamId, u32 taskId) const
{
    return nullptr;
}

HcclResult HcclCommunicator::SetAccelerator(HcclAccelerator hcclAccelerator, bool isCcuMsAvailable)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetAccelerator(int32_t accelerator, bool isCcuMsAvailable)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetRankGraphV2(void *&rankGraph)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetRankIpPortMap(RankIpPortMapPtr& rankIpPortMap)
{
    static auto emptyMap = std::make_shared<std::unordered_map<u32, std::unordered_map<IpAddress, u32>>>();
    rankIpPortMap = emptyMap;
    return HCCL_SUCCESS;
}
} // namespace Hccl

namespace Hccl {
class CommunicatorImplLite {
public:
    CommunicatorImplLite(u32 commId) : commId_(commId)
    {
    }
    ~CommunicatorImplLite()
    {
    }

private:
    u32 commId_;
};

class CommunicatorImplLiteMgr {
public:
    CommunicatorImplLiteMgr();
    ~CommunicatorImplLiteMgr();
    static CommunicatorImplLiteMgr &GetInstance();
    void DestroyComm(u32 commIdIndex);
    CommunicatorImplLite *Get(const u32 commIdIndex);
    std::vector<CommunicatorImplLite *> GetAll();
    void SetEnvConfig(const HcclDeviceEnvConfigLite &envConfig)
    {
        envConfig_ = envConfig;
    }
    const HcclDeviceEnvConfigLite &GetEnvConfig()
    {
        return envConfig_;
    }

private:
    std::unordered_map<u32, std::unique_ptr<CommunicatorImplLite>> communicatorImplLites;
    std::mutex serialMutex;
    HcclDeviceEnvConfigLite envConfig_;
};

CommunicatorImplLiteMgr::CommunicatorImplLiteMgr()
{
}
CommunicatorImplLiteMgr::~CommunicatorImplLiteMgr()
{
}

CommunicatorImplLiteMgr &CommunicatorImplLiteMgr::GetInstance()
{
    static CommunicatorImplLiteMgr instance;
    return instance;
}

void CommunicatorImplLiteMgr::DestroyComm(u32 commIdIndex)
{
    (void)commIdIndex;
}

CommunicatorImplLite *CommunicatorImplLiteMgr::Get(const u32 commIdIndex)
{
    (void)commIdIndex;
    return nullptr;
}

std::vector<CommunicatorImplLite *> CommunicatorImplLiteMgr::GetAll()
{
    return {};
}

RtNotify_t HrtIpcOpenNotifyWithFlag(const char_t *name, uint32_t flags)
{
    return nullptr;
}

u32 HrtStreamGetCqId(const aclrtStream ptr)
{
    return 0;
}

void HrtNotifyDestroy(RtNotify_t ptr)
{
    return;
}

s32 HrtGetStreamId(aclrtStream ptr)
{
    return 0;
}

aclrtStream HrtStreamCreateWithFlags(uint32_t priority, uint32_t flag)
{
    static aclrtStream stream;
    return stream;
}

void HrtStreamSetMode(HcclRtStream streamPtr, const uint64_t stmMode)
{
    return;
}

u32 HrtNotifyGetOffset(RtNotify_t ptr)
{
    return 0;
}

u32 HrtGetNotifyID(RtNotify_t notifyHandle)
{
    return 0;
}

s32 HrtDeviceGetBareTgid()
{
    return 0;
}

void HrtSetIpcNotifyPid(aclrtNotify notify, int32_t pid)
{
    return;
}

void HrtStreamDestroy(aclrtStream ptr)
{
    return;
}

void HrtIpcSetNotifyName(RtNotify_t ptr, char_t *name, uint32_t len)
{
    return;
}

aclrtNotify HrtNotifyCreateWithFlag(u32 devId, u32 flag)
{
    static aclrtNotify notify;
    return notify;
}

aclrtNotify HrtNotifyCreate(s32 deviceLogicId)
{
    static aclrtNotify notify;
    return notify;
}

u64 HrtNotifyGetAddr(RtNotify_t notifyHandle)
{
    return 0;
}

RtNotify_t HrtIpcOpenNotify(const char_t *name)
{
    return nullptr;
}

u32 HrtStreamGetSqId(const aclrtStream ptr)
{
    return 0;
}

void HrtNotifyRecord(RtNotify_t notifyPtr, aclrtStream streamPtr)
{
    return;
}

void HrtNotifyWaitWithTimeOut(RtNotify_t notifyPtr, aclrtStream streamPtr, uint32_t timeOut)
{
    return;
}

P2PTransport::P2PTransport(
    CommonLocRes &commonLocRes, Attribution &attr, const LinkData &linkData, const Socket &socket)
    : BaseMemTransport(commonLocRes, attr, linkData, socket, TransportType::P2P)
{
}

HcclResult P2PTransport::GetRemoteMem(HcclMem **remoteMem, uint32_t *memNum, char **memTags)
{
    return HCCL_SUCCESS;
}

HcclResult P2PTransport::GetUserRemoteMem(CommMem **remoteMem, char ***memTags, uint32_t *memNum)
{
    return HCCL_SUCCESS;
}

std::vector<char> P2PTransport::GetUniqueIdV2()
{
    return {};
}

std::string P2PTransport::Describe() const
{
    return "";
}

TransportStatus P2PTransport::GetStatus()
{
    return TransportStatus::READY;
}

std::vector<char> P2PTransport::GetUniqueId()
{
    return {};
}

void P2PTransport::Post(u32 index, const Stream &stream)
{
    return;
}

void P2PTransport::Read(const RmaBufferSlice &locSlice, const RmtRmaBufferSlice &rmtSlice, const Stream &stream)
{
    return;
}

void P2PTransport::ReadReduce(
    const RmaBufferSlice &locSlice, const RmtRmaBufferSlice &rmtSlice, const ReduceIn &reduceIn, const Stream &stream)
{
    return;
}

void P2PTransport::Write(const RmaBufferSlice &locSlice, const RmtRmaBufferSlice &rmtSlice, const Stream &stream)
{
    return;
}

void P2PTransport::WriteReduce(
    const RmaBufferSlice &locSlice, const RmtRmaBufferSlice &rmtSlice, const ReduceIn &reduceIn, const Stream &stream)
{
    return;
}

DevCapability::DevCapability()
{
}
DevCapability &DevCapability::GetInstance()
{
    static DevCapability devCapability;
    return devCapability;
}

P2PConnection::P2PConnection(Socket *socket, const std::string &tag) : RmaConnection(socket, RmaConnType::P2P)
{
}

void P2PConnection::Connect()
{
    return;
}
RmaConnStatus P2PConnection::GetStatus()
{
    return RmaConnStatus::READY;
}
string P2PConnection::Describe() const
{
    return "";
}

unique_ptr<BaseTask> P2PConnection::PrepareRead(
    const MemoryBuffer &remoteMemBuf, const MemoryBuffer &localMemBuf, const SqeConfig &config)
{
    return nullptr;
}

unique_ptr<BaseTask> P2PConnection::PrepareReadReduce(const MemoryBuffer &remoteMemBuf, const MemoryBuffer &localMemBuf,
    DataType datatype, ReduceOp reduceOp, const SqeConfig &config)
{
    return nullptr;
}

unique_ptr<BaseTask> P2PConnection::PrepareWrite(
    const MemoryBuffer &remoteMemBuf, const MemoryBuffer &localMemBuf, const SqeConfig &config)
{
    return nullptr;
}

unique_ptr<BaseTask> P2PConnection::PrepareWriteReduce(const MemoryBuffer &remoteMemBuf,
    const MemoryBuffer &localMemBuf, DataType datatype, ReduceOp reduceOp, const SqeConfig &config)
{
    return nullptr;
}

std::string RtsCntNotify::Describe() const
{
    return "";
}

std::string Rts1ToNCntNotify::Describe() const
{
    return "";
}

IpcLocalNotify::IpcLocalNotify(bool devUsed) : BaseLocalNotify(RmaType::IPC, devUsed)
{
}

void IpcLocalNotify::Wait(const Stream &stream, u32 timeout) const
{
    return;
}

void IpcLocalNotify::Post(const Stream &stream) const
{
    return;
}

std::unique_ptr<Serializable> IpcLocalNotify::GetExchangeDto()
{
    return nullptr;
}

string IpcLocalNotify::Describe() const
{
    return "";
}

} // namespace Hccl

HcclResult HcclCommDestroyV2(HcclComm comm)
{
    return HCCL_SUCCESS;
}

HcclResult HcommFlushV2()
{
    return HCCL_SUCCESS;
}

HcclResult HcclGetCommNameV2(HcclComm commHandle, char *commName)
{
    return HCCL_SUCCESS;
}

HcclResult HcclGetCclBuffer(
    HcclComm comm, uintptr_t &cclBufferAddr, size_t &cclBufferSize, HcclMemType &cclBufferMemType)
{
    return HCCL_SUCCESS;
}

HcclResult HcclGetRankGraphV2(HcclComm *comm, void **rankGraph)
{
    return HCCL_SUCCESS;
}

namespace Hccl {

std::pair<uint32_t, uint32_t> RdmaHandleManager::GetDieAndFuncId(RdmaHandle rdmaHandle)
{
    return {0, 0};
}

HcclResult TpManager::GetTpInfo(const RaUbGetTpInfoParam &param, TpInfo &tpInfo, bool isSync)
{
    (void)isSync;
    tpInfo.tpHandle = 1;
    tpInfo.mappedJettyPriority = static_cast<uint32_t>(param.qos & 0xFU);
    tpInfo.hasMappedJettyPriority = true;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult TpManager::ReleaseTpInfo(const RaUbGetTpInfoParam &param, const TpInfo &tpInfo)
{
    return HcclResult::HCCL_SUCCESS;
}

HrtRaUbJettyCreatedOutParam HrtRaUbCreateJetty(RdmaHandle handle, const HrtRaUbCreateJettyParam &in)
{
    return HrtRaUbJettyCreatedOutParam{};
}

void HrtRaUbDestroyJetty(JettyHandle jettyHandle)
{
}

HrtRaUbJettyImportedOutParam RaUbImportJetty(RdmaHandle handle, u8 *key, u32 keyLen, u32 tokenValue)
{
    return HrtRaUbJettyImportedOutParam{};
}

void HrtRaUbUnimportJetty(RdmaHandle handle, TargetJettyHandle targetJettyHandle)
{
}

HrtRaUbJettyImportedOutParam RaUbTpImportJetty(RdmaHandle handle, u8 *key, u32 keyLen,
    u32 tokenValue, const JettyImportCfg &jettyImportCfg)
{
    return HrtRaUbJettyImportedOutParam{};
}

ReqHandleResult HrtRaGetAsyncReqResult(RequestHandle &reqHandle)
{
    return ReqHandleResult::COMPLETED;
}

HrtRaUbSendWrRespParam HrtRaUbPostSend(JettyHandle jettyHandle, HrtRaUbSendWrReqParam &in)
{
    return HrtRaUbSendWrRespParam{};
}
}
int32_t HcommChannelRegisterDfx(ChannelHandle channel, std::function<HcclResult(unsigned int, unsigned int, const Hccl::TaskParam&, unsigned long long)> callback)
{
    return 0;
}
