# HcclTeamCreateChannelsDesc

## 功能说明

HCCL层Team通道创建描述符，用于[HcclTeamChannelsCreate](../control_plane_api/comms_domain_resource_mgmt/HcclTeamChannelsCreate.md)接口描述通信引擎、notify数量、通信协议及channel个数等信息。

## 定义原型

```c
typedef struct {
    CommAbiHeader   header;
    CommEngine      engine;          /* COMM_ENGINE_AICPU_TS / CCU 等 */
    uint32_t        notifyNum;       /* channel所需的notify数量 */
    CommProtocol    protocol;        /* 例如 COMM_PROTOCOL_HCCS / COMM_PROTOCOL_UB_RTP / COMM_PROTOCOL_UBOE */
    uint32_t        channelCnt;      /*  用户可以自行指定channel个数 */

    uint32_t        reserved[4];
} HcclTeamCreateChannelsDesc;
```

## 字段说明

| 字段名 | 描述 |
| --- | --- |
| header | ABI头部，由[HcclTeamCreateChannelsDescInit](../control_plane_api/comms_domain_resource_mgmt/HcclTeamCreateChannelsDescInit.md)初始化。CommAbiHeader类型的定义可参见[CommAbiHeader](CommAbiHeader.md)。 |
| engine | 通信引擎类型，例如COMM_ENGINE_AICPU_TS、COMM_ENGINE_CCU。CommEngine类型的定义可参见[CommEngine](CommEngine.md)。 |
| notifyNum | 每个channel所需的notify数量，取值范围为\[0,64\]。 |
| protocol | 通信协议，例如COMM_PROTOCOL_HCCS、COMM_PROTOCOL_UB_RTP、COMM_PROTOCOL_UBOE。CommProtocol类型的定义可参见[CommProtocol](CommProtocol.md)。 |
| channelCnt | 每个对端成员的channel个数，不可为0。 |
| reserved[4] | 预留字段。 |
