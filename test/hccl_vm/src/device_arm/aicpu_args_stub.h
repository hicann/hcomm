/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_ARGS_STUB_H
#define AICPU_ARGS_STUB_H

#include <cstdint>
constexpr uint32_t UNIQUEID_HEADER_OFFSET = 152;
constexpr uint32_t UNIQUEID_HEADER_SIZE = 20;
constexpr uint32_t COMMON_DATA_SIZE = 8;
constexpr uint32_t NOTIFY_ID_SIZE = 8;
constexpr uint32_t NOTIFY_BUFFER_SIZE = 28;
constexpr uint32_t LOCAL_BUFFER_SIZE = 28;
constexpr uint32_t REMOTE_BUFFER_SIZE = 28;
constexpr uint32_t HCOMID_MAX_SIZE = 128;

constexpr uint32_t P2P_MAX_ARG_SIZE_STUB = 8192;

// HcclP2pKernelParam
struct HcclP2pKernelParamStub {
    uint64_t sendRecvThread;
    uint8_t opParams[P2P_MAX_ARG_SIZE_STUB];
};

struct ThreadNotifyRecordParam {
    uint64_t thread;
    uint64_t dstThread;
    uint32_t dstNotifyIdx;
};

struct ThreadNotifyWaitParam {
    uint64_t thread;
    uint32_t notifyIdx;
};

struct HDCommunicateParams {
    uint64_t hostAddr{ 0 };
    uint64_t deviceAddr{ 0 };
    uint64_t readCacheAddr{ 0 };
    uint32_t devMemSize{ 0 };
    uint32_t buffLen{ 0 };
    uint32_t flag{ 0};
};

struct DevAicpuCommConfig {
    bool taskExceptionEnable{true};
    // 如要新增配置类字段，在此处添加
};
struct CommAicpuParam {
    char hcomId[HCOMID_MAX_SIZE];
    int32_t deviceLogicId;
    uint32_t devicePhyId;
    uint32_t deviceType;
    uint32_t userRankSize;
    uint32_t userRank;
    HDCommunicateParams kfcControlTransferH2DParams;
    HDCommunicateParams kfcStatusTransferD2HParams;
    DevAicpuCommConfig commConfig; // 收编通信域配置类变量
};

struct DevAicpuThreadConfig {
    // 如要新增配置类字段，在此处添加
};

struct ThreadMgrAicpuParam {
    uint32_t threadNum;
    char hcomId[128];
    char threadParam[200][6000]; // 含序列化后thread信息，约40KB
    void* deviceHandle;
    uint32_t rsv1;
    int32_t deviceLogicId{-1}; // 基础通信使用
    uint32_t deviceType{0}; // 基础通信使用
    DevAicpuThreadConfig threadConfig; // 收编thread配置类变量
};

struct InitTask {
    uint64_t context;
    bool isCustom;
};

struct DevAicpuChannelConfig {
    // 如要新增配置类字段，在此处添加
};

struct HcclChannelUrmaRes {
    char  hcomId[256];                    // 通信域ID 最大长度待修改
    void* channelList;                    // 反序列后返回给host侧的device侧handle地址
    uint32_t   listNum = 0;               // 建链channel的总数量
    void* uniqueIdAddr;                   // 序列化后device侧地址
    uint32_t   uniqueIdSize{0};           // 序列化后总地址长度
    void*   channelSizeAddr{nullptr};           // 存放序列化后device channel size的指针
    uint32_t*  remoteRankList;            // 序列化后返回给host侧的device侧rankList地址
    uint32_t*  remoteRankId;              // 记录每个channel的对端rank
    int32_t    deviceLogicId{0};          // 基础通信使用
    uint32_t   deviceType{0};             // 基础通信使用
    DevAicpuChannelConfig channelConfig;  // 收编channel配置类变量
};

#pragma pack(push, 1)

struct UniqueIdV2Header {
    uint32_t type;
    uint32_t notifyNum;
    uint32_t bufferNum;
    uint32_t rmtBufferNum;
    uint32_t connNum;
};

struct ConnUniqueIds {
    uint32_t dieId;
    uint32_t funcId;
    uint32_t jettyId;
    uint32_t jfcPollMode;
    uint8_t dwqeCacheLocked;
    uint64_t dbAddr;
    uint64_t sqCiAddr;
    uint64_t sqBuffVa;
    uint32_t sqDepth;
    uint32_t tpn;
    uint8_t rmtEidRaw[16U];
    uint8_t locEidRaw[16U];
};

struct ConnUniqueBlock {
    uint64_t size;
    ConnUniqueIds conn[0];
};

struct AicpuTsThread {
    uint32_t streamType;
    uint32_t notifyLoadType;
    uint32_t devId;
};

#pragma pack(pop)

#endif
