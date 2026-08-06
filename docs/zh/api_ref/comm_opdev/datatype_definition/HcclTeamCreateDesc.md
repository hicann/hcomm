# HcclTeamCreateDesc

## 功能说明

HCCL层Team创建描述符，用于创建world team或sub team时描述成员、网络层、通信协议及syncMem需求等信息。

## 定义原型

```c
typedef struct {
    CommAbiHeader header;
    const uint32_t *rankIds;     /* 用户希望填写的rankids */
    uint32_t  rankNum;
    uint32_t  selfRankId;
    uint32_t  netLayer;     /* 用户希望使用的网络层，0表示默认选择 */
    uint32_t  channelCnt;   /* 用户可以自行指定channel个数 */
    CommProtocol protocol;  /* 用户希望使用的通信协议，-1表示保留协议类型, 1个team仅支持一个协议 */
    HcommTeamSyncMemRequirement requirement;

    uint32_t reserved[4];
} HcclTeamCreateDesc;
```

## 字段说明

| 字段名 | 描述 |
| --- | --- |
| header | ABI头部，由[HcclTeamCreateDescInit](../control_plane_api/comms_domain_resource_mgmt/HcclTeamCreateDescInit.md)初始化。CommAbiHeader类型的定义可参见[CommAbiHeader](CommAbiHeader.md)。 |
| rankIds | rankId数组，长度为rankNum。world team与sub team均必填，不可为NULL。 |
| rankNum | 成员数量。不可为0或1，且不可大于通信域的rankSize。 |
| selfRankId | 本rank的实际rankId，必须存在于rankIds中。 |
| netLayer | 希望使用的网络层，只能为0、1或2，0表示默认选择。 |
| channelCnt | 每个对端成员的channel个数，不可为0。 |
| protocol | 希望使用的通信协议，-1或COMM_PROTOCOL_RESERVED表示保留协议类型，1个team仅支持一个协议。CommProtocol类型的定义可参见[CommProtocol](CommProtocol.md)。 |
| requirement | syncMem需求，包含signal/counter/barrier数量。HcommTeamSyncMemRequirement类型的定义可参见[HcommTeamSyncMemRequirement](HcommTeamSyncMemRequirement.md)。 |
| reserved[4] | 预留字段。 |
