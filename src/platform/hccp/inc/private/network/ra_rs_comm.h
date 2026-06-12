/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_RS_COMM_H
#define RA_RS_COMM_H
#include <stdint.h>
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include "hccp_common.h"
#include "rdma_lite_common.h"

#define HOST_LITE_RESERVED 4
#define RA_MR_MAX_NUM 8

enum OpType {
    RA_RS_SOCKET_CONN = 0,
    RA_RS_SOCKET_CLOSE = 1,
    RA_RS_SOCKET_LISTEN_START = 2,
    RA_RS_SOCKET_LISTEN_STOP = 3,
    RA_RS_GET_SOCKET = 4,
    RA_RS_SOCKET_SEND = 5,
    RA_RS_SOCKET_RECV = 6,
    RA_RS_QP_CREATE = 7,
    RA_RS_QP_DESTROY = 8,
    RA_RS_QP_CONNECT = 9,
    RA_RS_QP_STATUS = 10,
    RA_RS_MR_REG = 11,
    RA_RS_MR_DEREG = 12,
    RA_RS_SEND_WR = 13,
    RA_RS_GET_NOTIFY_BA = 14,
    RA_RS_INIT = 15,
    RA_RS_DEINIT = 16,
    RA_RS_SOCKET_INIT = 17,
    RA_RS_SOCKET_DEINIT = 18,
    RA_RS_RDEV_INIT = 19,
    RA_RS_RDEV_DEINIT = 20,
    RA_RS_WLIST_ADD = 21,
    RA_RS_WLIST_DEL = 22,
    RA_RS_GET_IFADDRS = 23,
    RA_RS_GET_INTERFACE_VERSION = 24,
    RA_RS_SEND_WRLIST = 25,
    RA_RS_SET_TSQP_DEPTH = 26,
    RA_RS_GET_TSQP_DEPTH = 27,
    RA_RS_SEND_WRLIST_EXT = 28,
    RA_RS_SET_QP_ATTR_QOS = 29,
    RA_RS_SET_QP_ATTR_TIMEOUT = 30,
    RA_RS_SET_QP_ATTR_RETRY_CNT = 31,
    RA_RS_GET_CQE_ERR_INFO = 32,
    RA_RS_GET_IFNUM = 33,
    RA_RS_GET_LITE_SUPPORT = 34,
    RA_RS_GET_LITE_RDEV_CAP = 35,
    RA_RS_GET_LITE_QP_CQ_ATTR = 36,
    RA_RS_GET_LITE_CONNECTED_INFO = 37,
    RA_RS_GET_IFADDRS_V2 = 38,
    RA_RS_QP_CREATE_WITH_ATTRS = 39,
    RA_RS_WLIST_ADD_V2 = 40,
    RA_RS_WLIST_DEL_V2 = 41,
    RA_RS_SEND_WRLIST_V2 = 42,
    RA_RS_SEND_WRLIST_EXT_V2 = 43,
    RA_RS_GET_LITE_MEM_ATTR = 44,
    RA_RS_TYPICAL_QP_CREATE = 45,
    RA_RS_TYPICAL_QP_MODIFY = 46,
    RA_RS_TYPICAL_MR_REG_V1 = 47,
    RA_RS_TYPICAL_MR_DEREG = 48,
    RA_RS_CTX_INIT = 49,
    RA_RS_CTX_DEINIT = 50,
    RA_RS_LMEM_REG = 51,
    RA_RS_LMEM_UNREG = 52,
    RA_RS_RMEM_IMPORT = 53,
    RA_RS_RMEM_UNIMPORT = 54,
    RA_RS_GET_VNIC_IP_INFOS_V1 = 55,
    RA_RS_GET_DEV_EID_INFO_NUM = 56,
    RA_RS_GET_DEV_EID_INFO_LIST = 57,
    RA_RS_CTX_CQ_CREATE = 58,
    RA_RS_CTX_CQ_DESTROY = 59,
    RA_RS_CTX_QP_CREATE = 60,
    RA_RS_CTX_QP_DESTROY = 61,
    RA_RS_CTX_QP_IMPORT = 62,
    RA_RS_CTX_QP_UNIMPORT = 63,
    RA_RS_CTX_QP_BIND = 64,
    RA_RS_CTX_QP_UNBIND = 65,
    RA_RS_CTX_BATCH_SEND_WR = 66,
    RA_RS_QP_BATCH_MODIFY = 67,
    RA_RS_AI_QP_CREATE = 68,
    RA_RS_CUSTOM_CHANNEL = 69,
    RA_RS_RDEV_GET_PORT_STATUS = 70,
    RA_RS_PING_INIT = 71,
    RA_RS_PING_ADD = 72,
    RA_RS_PING_START = 73,
    RA_RS_PING_GET_RESULTS = 74,
    RA_RS_PING_STOP = 75,
    RA_RS_PING_DEL = 76,
    RA_RS_PING_DEINIT = 77,
    RA_RS_CTX_UPDATE_CI = 78,
    RA_RS_GET_CQE_ERR_INFO_NUM = 79,
    RA_RS_GET_CQE_ERR_INFO_LIST = 80,
    RA_RS_RDEV_INIT_WITH_BACKUP = 81,
    RA_RS_QP_INFO = 82,
    RA_RS_SEND_NORMAL_WRLIST = 83,
    RA_RS_CTX_CHAN_CREATE = 84,
    RA_RS_CTX_CHAN_DESTROY = 85,
    RA_RS_AI_QP_CREATE_WITH_ATTRS = 86,
    RA_RS_TLV_INIT_V1 = 87,
    RA_RS_TLV_DEINIT = 88,
    RA_RS_TLV_REQUEST = 89,
    RA_RS_CTX_TOKEN_ID_ALLOC = 90,
    RA_RS_CTX_TOKEN_ID_FREE = 91,
    RA_RS_REMAP_MR = 92,
    RA_RS_GET_TP_INFO_LIST = 93,
    RA_RS_GET_VNIC_IP_INFOS = 94,
    RA_RS_GET_TLS_ENABLE = 95,
    RA_RS_GET_ROCE_API_VERSION = 96,
    RA_RS_SOCKET_ABORT = 97,
    RA_RS_ACCEPT_CREDIT_ADD = 98,
    RA_RS_GET_SEC_RANDOM = 99,
    RA_RS_GET_HCCN_CFG = 100,
    RA_RS_TYPICAL_MR_REG = 101,
    RA_RS_CTX_QP_DESTROY_BATCH = 102,
    RA_RS_CTX_QUERY_QP_BATCH = 103,
    RA_RS_GET_EID_BY_IP = 104,
    RA_RS_CTX_GET_AUX_INFO = 105,
    RA_RS_GET_TP_ATTR = 106,
    RA_RS_SET_TP_ATTR = 107,
    RA_RS_CTX_GET_CR_ERR_INFO_LIST = 108,
    RA_RS_CTX_GET_ASYNC_EVENTS = 109,
    RA_RS_TLV_INIT = 110,
    RA_RS_CTX_GET_UB_CONTEXT = 111,
    RA_RS_EXTER_OP_MAX_NUM,

    // 上面opcode是对部opcode,下面是内部opcode
    RA_RS_HDC_SESSION_CLOSE = 1000,
    RA_RS_GET_VNIC_IP = 1001,
    RA_RS_NOTIFY_CFG_SET = 1002,
    RA_RS_NOTIFY_CFG_GET = 1003,
    RA_RS_SET_PID = 1004,
    RA_RS_ASYNC_HDC_SESSION_CONNECT = 1005,
    RA_RS_ASYNC_HDC_SESSION_CLOSE = 1006,
    RA_RS_OP_MAX_NUM,
};

enum ModuleType {
    HCCP_INIT = 0,
    RDMA_OP = 1,
    SOCKET_OP = 2,
    OTHERS = 3,
};

enum {
    RA_RS_NOR_QP_MODE      = 0,
    RA_RS_GDR_TMPL_QP_MODE = 1,
    RA_RS_OP_QP_MODE       = 2,
    RA_RS_GDR_ASYN_QP_MODE = 3,
    RA_RS_OP_QP_MODE_EXT   = 4,
    RA_RS_ERR_QP_MODE      = 5,
};

struct OpcodeInterfaceInfo {
    enum OpType opcode;
    unsigned int version;
};

struct WrlistSendCompleteNum {
    unsigned int sendNum;
    unsigned int *completeNum;
};

struct RdmaMrRegAttr {
    void *addr;
    unsigned long long len;
    int access;
    unsigned int resv;
};

struct RdmaMrRegInfo {
    void *addr;
    unsigned long long len;
    int access;
    unsigned int lkey;
    unsigned int rkey;
};

struct SocketConnectInfo {
    unsigned int phyId;
    int family;
    union HccpIpAddr localIp;
    union HccpIpAddr remoteIp;
    unsigned int port;
    char tag[SOCK_CONN_TAG_SIZE];
};

struct SocketListenInfo {
    unsigned int phyId;
    int family;
    union HccpIpAddr localIp;
    unsigned int port;
    unsigned int phase;
    unsigned int err;
};

struct SocketFdData {
    int fd;
    unsigned int phyId;
    int family; // AF_INET(ipv4) or AF_INET6(ipv6)
    union HccpIpAddr localIp;
    union HccpIpAddr remoteIp;
    int status;
    char tag[SOCK_CONN_TAG_SIZE];
};

struct SocketPeerInfo {
    int phyId;
    int fd;
    void *socketHandle;
    uint32_t sslEnable;
};

struct QpAttrDscpInfo {
    unsigned char tc;
    unsigned char sl;
};

struct LiteMrInfo {
    uint32_t key;
    uint64_t addr;
    uint64_t len;
};

struct LiteRdevCapResp {
    struct dev_cap_info cap;
    unsigned char reserved[HOST_LITE_RESERVED];
};

struct LiteQpCqAttrResp {
    struct rdma_lite_device_qp_attr qpData;
    struct rdma_lite_device_cq_attr sendCqData;
    struct rdma_lite_device_cq_attr recvCqData;
    unsigned char reserved[HOST_LITE_RESERVED];
};

struct LiteConnectedInfoResp {
    unsigned int state;
    struct LiteMrInfo localMr[RA_MR_MAX_NUM];
    struct LiteMrInfo remMr[RA_MR_MAX_NUM];
    struct QosAttr qosAttr;
    unsigned char reserved[HOST_LITE_RESERVED];
};

struct LiteMemAttrResp {
    struct rdma_lite_device_mem_attr memData;
    unsigned char reserved[HOST_LITE_RESERVED];
};

struct RsQpNormWithAttrs {
    int isExp;
    int isExt;
    struct QpExtAttrs extAttrs;
    unsigned int aiOpSupport;
};

struct RsQpRespWithAttrs {
    unsigned int qpn;
    unsigned long long aiQpAddr; // refer to struct ibv_qp *
    unsigned int sqIndex; // index of sq
    unsigned int dbIndex; // index of db
    unsigned int psn;
    unsigned int gidIdx;

    // below cq related info valid when data_plane_flag.bs.cq_cstm was 1
    unsigned long long aiScqAddr; // refer to struct ibv_cq *scq
    unsigned long long aiRcqAddr; // refer to struct ibv_cq *rcq
    struct AiDataPlaneInfo dataPlaneInfo;
};

struct RsQpResp {
    unsigned int qpn;
    unsigned int psn;
    unsigned int gidIdx;
    union ibv_gid gid;
    int directFlag;
};

struct RaRsDevInfo {
    unsigned int phyId;
    unsigned int devIndex;
};

enum {
    THREAD_HALT  = 0,
    THREAD_RUNNING  = 1,
    THREAD_DESTROYING  = 2,
};

enum {
    HDC_UNCONNECTED  = 0,
    HDC_CONNECTED  = 1,
};

struct TlvRequestMsgHead {
    unsigned int phyId;
    unsigned int type;
    unsigned int moduleType;
    unsigned int totalBytes;
    unsigned int sendBytes;
    unsigned int offset;
};

enum {
    CONTEXT_TYPE_JETTY = 0,
    CONTEXT_TYPE_JFC = 1,
};

#define RA_THREAD_TRY_TIME 500
#define RA_THREAD_SLEEP_TIME 2000
#define RA_CONNECT_TRY_TIME (500 * 120)
#define HDC_TRY_TIME 200
#define HDC_USLEEP_TIME 20000
#define THREAD_SLEEP_TIME 100 /* 100us */
#define MAX_POOL_QUEUE_SIZE 2048U
#define MAX_POOL_THREAD_NUM 2U
#define RA_POOL_THREAD_NUM 1U

#define MAX_WR_NUM_V1 1024
#define MAX_WR_NUM 64
#define MAX_PORT_NUM 65535
#define MAX_IP_INFO_NUM 128
#define MAX_IP_INFO_NUM_V1 256
#define MAX_SGE_NUM 16
#define RA_RS_4K_PAGE_SIZE 0x1000
#define HCCN_CFG_MSG_DATA_LEN 2048U
#define MAX_TLV_MSG_DATA_LEN 2048U

#define RA_RS_GET_IFNUM_VERSION 1
#define RA_RS_WLIST_ADD_V2_VERSION 1
#define RA_RS_WLIST_DEL_V2_VERSION 1
#define RA_RS_SEND_WRLIST_V2_VERSION 1
#define RA_RS_SEND_WRLIST_EXT_V2_VERSION 1
#define RA_RS_SOCKET_CONN_VERSION 2
#define RA_RS_SOCKET_LISTEN_VERSION 2
#define RA_RS_GET_SOCKET_VERSION 2
#define RA_RS_GET_VNIC_IP_INFOS_VERSION 1
#define RA_RS_GET_NOTIFY_BA_VERSION 1
#define RA_RS_OPCODE_BASE_VERSION 1

#define LITE_VERSION_V2                  2
#define LITE_SUPPORT_DEV_MEM_REGISTER    1
#define LITE_SUPPORT_PCIE_BAR_HUGE_MEM   (1 << 1)
#define LITE_NOT_SUPPORT                 0
#define LITE_ALIGN_4KB                   1
#define LITE_ALIGN_2MB                   (1 << 1)

#define RA_RS_GET_ALL_IP_BIT_MASK (1U << 31)

#define CQ_DEFAULT_MIN_SEND_DEPTH         64
#define CQ_DEFAULT_MIN_RECV_DEPTH         64
#define QP_DEFAULT_MIN_CAP_SEND_WR        CQ_DEFAULT_MIN_SEND_DEPTH
#define QP_DEFAULT_MIN_CAP_RECV_WR        CQ_DEFAULT_MIN_RECV_DEPTH
#define QP_DEFAULT_MAX_CAP_INLINE_DATA    32
#define QP_DEFAULT_MIN_CAP_SEND_SGE       1
#define QP_DEFAULT_MIN_CAP_RECV_SGE       1
#define QP_DEFAULT_MAX_ATTR_TIMEOUT       20
#define QP_DEFAULT_MAX_ATTR_RETRY_CNT     7

int ConverReturnCode(enum ModuleType module, int erroCode);

static inline void RaRsSetDevInfo(struct RaRsDevInfo *devInfo, unsigned int phyId, unsigned int devIndex)
{
    devInfo->phyId = phyId;
    devInfo->devIndex = devIndex;
}

#endif

