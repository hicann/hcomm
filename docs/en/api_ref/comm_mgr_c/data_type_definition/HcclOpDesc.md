# HcclOpDesc

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T08:24:21.690Z pushedAt=2026-08-14T08:44:43.978Z -->

## Description

Describes the operator information when the AI CPU runs a kernel function, including the operator type, name, and communication-related parameters. This structure is used in custom communication operator scenarios and supports the description of Send and Receive communication tasks.

## Prototype

```c
typedef struct {
    CommAbiHeader header;
    uint32_t opDescType;
    char opName[HCCL_OP_DESC_OP_NAME_MAX_LEN];
    union {
        uint8_t raws[76];
        HcclOpP2pDesc p2p;
    };
} HcclOpDesc;
```

## Parameters

- **header**: ABI header, which contains information such as the version. For the type definition, see [CommAbiHeader](../../comm_opdev/datatype_definition/CommAbiHeader.md).
- **opDescType**: Operator description type.
- **opName**: Operator name, with a maximum length of 256 bytes (HCCL_OP_DESC_OP_NAME_MAX_LEN).
- **raws**: Raw data used in general scenarios, with a length of 76 bytes.
- **p2p**: Description parameters of Send and Receive communication tasks, of the [HcclOpP2pDesc](#hcclopp2pdesc) type. As a union member, it shares the 76-byte space with raws.

## HcclOpP2pDesc

Describes the detailed information of Send and Receive communication tasks, including parameters such as the data buffer address, communication command type, data type, data count, peer rank ID, and unfold stream. This structure is used as a union member of HcclOpDesc.

### Prototype

```c
typedef struct {
    void *buffer;
    uint8_t reserved[8];
    HcclCMDType cmdType;
    HcclDataType dataType;
    uint64_t count;
    uint32_t remoteRank;
    void *unfoldStream;
} HcclOpP2pDesc;
```

### Parameters

- **buffer**: Data buffer address, which is the memory address used to send or receive data. Ensure that the memory is correctly allocated and accessible.
- **reserved**: Reserved field, 8 bytes in length, for future extension.
- **cmdType**: Communication command type, which specifies the communication operation type (for example, HCCL_CMD_SEND and HCCL_CMD_RECEIVE).
- **dataType**: Data type. For the type definition, see [HcclDataType](./HcclDataType.md).
- **count**: Number of data elements to be transferred. It must match the actual data size and type of the buffer corresponding to dataType.
- **remoteRank**: Rank ID of the peer node for communication. It must be within the valid rank ID range of the communicator.
- **unfoldStream**: Unfold stream, used for stream control of AI CPU communication tasks.

## Related Constants

```c
const uint32_t HCCL_OP_DESC_OP_NAME_MAX_LEN = 256;  // Maximum length of the operator name
```
