/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_TS_UBG_CHANNEL_H
#define AICPU_TS_UBG_CHANNEL_H

#include "aicpu_ts_uboe_ubg_channel_helper.h"

namespace hcomm {

constexpr char_t UBG_FINISH_MSG[FINISH_MSG_SIZE] = "Ubg Comm Pipe ready!";

class AicpuTsUbgChannel : public AicpuTsUboeUbgChannelHelper {
public:
    AicpuTsUbgChannel(EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc)
        : AicpuTsUboeUbgChannelHelper(endpointHandle, channelDesc)
    {}

    ~AicpuTsUbgChannel() = default;

    HcommChannelKind GetChannelKind() const override { return HcommChannelKind::AICPU_TS_UBG; }

    HcclResult Init() override;
    ChannelStatus GetStatus() override;
    HcclResult Clean() override;
    HcclResult Resume() override;

protected:
    HcclResult BuildConnection() override;
    void SendFinish() override;
    void RecvFinish() override;

private:
    void ProcessUbgState();

    MAKE_ENUM(
        UbgStatus, INIT, BUILD_CONN, SEND_SIZE, RECV_SIZE, SEND_DATA, RECV_DATA, SEND_FIN, RECV_FIN, PROCESS_DATA,
        SET_READY, READY)
    UbgStatus ubgStatus{UbgStatus::INIT};
};

} // namespace hcomm

#endif // AICPU_TS_UBG_CHANNEL_H
