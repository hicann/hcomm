/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef COLL_COMM_MGR_H
#define COLL_COMM_MGR_H

#include <unordered_set>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <array>
#include "coll_comm.h"
#include "cluster_monitor.h"

namespace hccl {
/**
 * @note 职责：实现多个集合通信通信域上下文的创建、销毁管理，及多通信域资源、信息的共享等。
 */
class CollCommMgr {
public:
    static CollCommMgr *GetInstance();
    void RegisteCollComm(CollComm* collComm);
    void UnRegisteCollComm(CollComm* collComm);
    std::unordered_map<std::string, CollComm*> GetAllCollComms();
    hcomm::ClusterMonitor &GetClusterMonitor(s32 deviceLogicId);
    HcclResult TryReserveCcuMsComm(s32 deviceLogicId, const std::string &commId, bool &reserved);
    void ReleaseCcuMsComm(s32 deviceLogicId, const std::string &commId);

private:
    static CollCommMgr* instance_;
    std::unordered_map<std::string, CollComm*> allCollComms_;
    std::array<hcomm::ClusterMonitor, MAX_MODULE_DEVICE_NUM> clusterMonitor_;
    std::array<std::string, MAX_MODULE_DEVICE_NUM> ccuMsCommIds_{};

    std::mutex mutex_;
    std::mutex ccuMsCommMutex_;
};
}
#endif // COLL_COMM_MGR_H
