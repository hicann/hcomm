# HcommChannelDesc

## 功能说明

定义组件间通道参数。

## 定义原型

```c
typedef struct {
    CommAbiHeader header;             /* ABI头部，包含版本等信息 */
    EndpointDesc remoteEndpoint;      /* 远端网络设备端侧描述 */
    uint32_t notifyNum;               /* channel上使用的同步信号数量 */

    // exchangeAllMems = True时不需要配置memHandle
    bool exchangeAllMems;             /* 表示是否交换本地网络设备端注册的内存信息 */
    HcommMemHandle *memHandles;       /* 注册到通信域的待交换内存句柄，exchangeAllMems为True时无效 */
    uint32_t memHandleNum;            /* 注册到通信域的待交换内存句柄数量，exchangeAllMems为True时无效 */
    HcommSocket socket;               /* Socket句柄 */
    HcommSocketRole role;             /* 本端角色(SERVER或CLIENT) */
    uint16_t port;                    /* Socket监听指定端口*/
    union {
        uint8_t raws[128];            /* 通用缓存 */
        struct {
            uint32_t queueNum;        /* QP数量，当前仅支持一个QP */
            uint32_t retryCnt;        /* 最大重传次数，范围为0~7，默认为7 */
            uint32_t retryInterval;   /* 重传间隔，范围为5~24，默认为20 (对应时间4.096*2^20us) */
            uint8_t tc;               /* 流量类别(QoS)，范围为0~255，默认为132 */
            uint8_t sl;               /* 服务等级(QoS)，范围为0~7，默认为4 */
            uint32_t qpThreshold;     /* 多QP场景下，每个QP最小数据量(B) */
        } roceAttr;
        struct {
            uint32_t qos;             /* HCCS QoS */
        } hccsAttr;
        struct {
            uint32_t sqDepth;         /* UB队列深度，0表示使用默认值 */
        } ubAttr;
    };
    uint32_t qos;             /* 通信域QoS与协议解耦 */
    const char *channelName;  /* channel业务匹配标识，两端需相同；NULL表示匿名channel */
} HcommChannelDesc;
```
