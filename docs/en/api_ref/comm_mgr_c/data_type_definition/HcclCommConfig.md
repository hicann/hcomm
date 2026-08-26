# HcclCommConfig

<!-- md-trans-meta sourceCommit=5d2b6a1c1d3ecb2aedf23e593d18044c239c2ddc translatedAt=2026-08-14T08:23:08.583Z pushedAt=2026-08-15T03:42:30.039Z -->

## Description

When initializing a communicator with specific configurations, this data type is used to define the communicator configuration information, including multiple configuration items such as the buffer size, deterministic computation switch, communicator name, communication operator expansion mode, communication algorithm, and symmetric memory reservation.

## Prototype

```c
const uint32_t HCCL_COMM_CONFIG_INFO_BYTES = 24;
const uint32_t COMM_NAME_MAX_LENGTH = 128;
const uint32_t BUFFER_NAME_MAX_LENGTH = 128;
const uint32_t UDI_MAX_LENGTH = 128;
const uint32_t HCCL_COMM_ALGO_MAX_LENGTH = 1600;
const uint32_t HCCL_COMM_RETRY_ENABLE_MAX_LENGTH = 50;
const uint32_t HCCL_COMM_RETRY_PARAMS_MAX_LENGTH = 128;
typedef struct HcclCommConfigDef {
    char reserved[HCCL_COMM_CONFIG_INFO_BYTES];    /* Reserved field. Do not modify. */
    uint32_t hcclBufferSize;
    uint32_t hcclDeterministic;
    char hcclCommName[COMM_NAME_MAX_LENGTH];
    char hcclUdi[UDI_MAX_LENGTH];
    uint32_t hcclOpExpansionMode;
    uint32_t hcclRdmaTrafficClass;
    uint32_t hcclRdmaServiceLevel;
    uint32_t hcclWorldRankID;
    uint64_t hcclJobID;
    uint8_t aclGraphZeroCopyEnable;
    int32_t hcclExecTimeOut;
    char hcclAlgo[HCCL_COMM_ALGO_MAX_LENGTH];
    char hcclRetryEnable[HCCL_COMM_RETRY_ENABLE_MAX_LENGTH];
    char hcclRetryParams[HCCL_COMM_RETRY_PARAMS_MAX_LENGTH];
    char hcclBufferName[BUFFER_NAME_MAX_LENGTH];
    uint32_t hcclQos;
    uint64_t hcclSymWinMaxMemSizePerRank;
} HcclCommConfig;
```

## Parameters

- **hcclBufferSize**: buffer size of the shared data. It must be configured as an integer greater than or equal to 1, in MB. For the value range and usage constraints for different product types, see the environment variable [HCCL_BUFFSIZE](https://gitcode.com/cann/hccl/blob/9.1.0/docs/en/user_guide/hccl_env/HCCL_BUFFSIZE.md).

  Note the following:

  - This configuration item has a higher priority than the HCCL_BUFFSIZE environment variable.
  - The memory requested by this configuration item is exclusively used by HCCL and cannot be reused by other service memory.
  - A communicator created based on this configuration item exclusively occupies memory of "2\*hcclBufferSize", ensuring that concurrent operators of multiple communicators do not affect each other.
  - For collective communication operators, performance may degrade when the data size exceeds this configured value. It is recommended that the hcclBufferSize value be greater than the data size.

- **hcclDeterministic**: deterministic computation switch, supported on the following models:

  The following lists the supported values and their meanings for different AI processors. Values not listed are not supported.

  <!-- npu="950" id1 -->
  - Ascend 950PR/Ascend 950DT: This configuration is not supported. You can configure the global deterministic computation switch through the HCCL_DETERMINISTIC environment variable.
  <!-- end id1 -->
  <!-- npu="A3" id2 -->
  - Atlas A3 Training Series/Atlas A3 Inference Series: The supported values and their meanings are as follows:
    - 0 (default): Disables deterministic computation.
    - 1: Enables deterministic computation for reduction communication operators, supporting the AllReduce and ReduceScatter communication operators.
    - 2: In single-operator mode, configuring this parameter to "2" has the same effect as configuring it to "1". In static graph mode, configuring it to "2" is not supported.
  <!-- end id2 -->

  <!-- npu="910b" id3 -->
  - Atlas A2 training products/Atlas A2 inference products: The supported values and their meanings are as follows:
    - 0 (default): Disables deterministic computation.
    - 1: Enables deterministic computation for reduction communication operators, including AllReduce, ReduceScatter, Reduce, and ReduceScatterV.
    - 2: Enables strict deterministic computation for reduction communication operators, that is, the order preservation function (ensuring that the reduction order of all bits is consistent on the basis of determinism). The supported communication operators are AllReduce, ReduceScatter, and ReduceScatterV. When this parameter is set to this value, the following conditions must be met:
      - Only multi-server symmetric distribution scenarios are supported. Asymmetric distribution scenarios are not supported.
      - When order preservation is enabled, the saturation mode is not supported. Only the INF/NaN mode is supported.
      - Compared with deterministic computation, enabling order preservation causes a certain degree of performance degradation. It is recommended to use this function in inference scenarios.
  <!-- end id3 -->

    > [!NOTE] Note
    > When deterministic computation is not enabled, the results of multiple executions may differ. This difference generally comes from asynchronous multi-thread execution in the operator implementation, which causes the accumulation order of floating-point numbers to change. When deterministic computation is enabled, the operator produces the same output across multiple executions under the same hardware and input.
    > By default, deterministic computation does not need to be enabled. However, when you find that a model produces different results across multiple executions or during precision tuning, you can enable deterministic computation to assist debugging and tuning. Note that after it is enabled, the operator execution time increases, causing performance degradation.

- **hcclCommName**: communicator name, with a maximum length of 128.

  The specified communicator name must be unique among all communicators. If it is not specified, HCCL automatically generates one.

- **hcclUdi**: User-defined information, with a maximum length of 128. The default value is empty.
- **hcclOpExpansionMode**: Configures the expansion mode of communication operators. This is a communicator-level configuration.

  The following lists the supported values and their meanings for different AI processors. Values not listed are not supported for configuration.

  <!-- npu="950" id4 -->
  **For Ascend 950PR/Ascend 950DT, the supported values and their meanings are as follows:**

  - 0: Uses the default operator expansion mode. For **Ascend 950PR/Ascend 950DT**, communication operators are expanded on the AI CPU compute unit by default.
  - 2: Communication operators are expanded on the AI CPU compute unit and scheduled by the STARS scheduler.

    This configuration item supports the Broadcast, Reduce, AllReduce, Scatter, ReduceScatter, ReduceScatterV, AllGather, AllGatherV, AlltoAll, AlltoAllV, AlltoAllVC, Send, Recv, and BatchSendRecv operators.

    In graph mode (Ascend IR) or graph capture (aclgraph) scenarios, when the communication algorithm uses the AI CPU mode, the number of concurrent graphs on a single card cannot exceed 6. Otherwise, communication may be blocked because the AI CPU cores are fully occupied.

  - 3: Communication operators are expanded on the Vector Core compute unit on the Device side.
    - This configuration supports only symmetric networking and the inference feature.
    - In this configuration, if the data size does not meet the running requirements on Vector Core, some operators automatically switch to the default mode.
    - This configuration supports only the Broadcast, Reduce, AllReduce, ReduceScatter, Scatter, AllGather, AlltoAll, and AlltoAllV operators, and currently supports only the single-server scenario.
      - For the Broadcast, Scatter, AllGather, AlltoAll, and AlltoAllV operators, the supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, and bfp16.
      - For the Reduce, AllReduce, and ReduceScatter operators, the supported data types are int8, int16, int32, float16, float32, and bfp16.

    - With this configuration, the AllReduce, ReduceScatter, AllGather, and AlltoAll operators support the core control capability. You are advised to configure the number of Vector Cores based on the concurrency between compute operators and communication operators in actual service scenarios.

  - 4: The communication operator is expanded on the Vector Core compute unit on the Device side, but mode switching does not occur as the data size changes. Vector Core is always used for computation. If the running conditions of Vector Core are not met, an error is reported and the process exits.
    - This configuration supports only symmetric networking and the inference feature.
    - For the operators supported by this configuration and the constraints, see configuration "3".

  - 5: The communication operator is expanded on the CCU (Collective Communication Unit) and uses CcuBuffer for memory read/write. Ascend 950PR does not support this configuration.

    In this mode, when the CCU communicates with multiple remote ends, CcuBuffer is used as a relay to save memory read/write bandwidth. CcuBuffer is characterized by its small size but high speed. When CCU resources are insufficient, the system automatically switches to "2: AI CPU mode".

    - This configuration supports only the Broadcast, Reduce, AllReduce, ReduceScatter, AllGather, AllGatherV, and ReduceScatterV operators, and currently supports only the single-server scenario.
      - For the Broadcast, AllGather, and AllGatherV operators, the supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, and bfp16.
      - For the Reduce, AllReduce, ReduceScatter, and ReduceScatterV operators, the supported data types are int16, int32, float16, float32, and bfp16.

  - 6: The communication operator is expanded on the CCU and uses the scheduling mode.

    The scheduling mode uses the CCU as a scheduler to schedule UB WQE tasks to the UB engine. In the scheduling mode, CcuBuffer is not used, and data is directly transferred between two ranks from on-chip memory to on-chip memory.

    For the AllReduce, ReduceScatter, and Reduce operators in single-server communication scenarios, when the data size exceeds a certain value, the system automatically switches to "2: AI_CPU mode" to prevent performance degradation (this threshold is not fixed and is adjusted based on factors such as the operator running mode and network scale).

    In this mode, the ReduceScatterV and AllGatherV operators support only the single-server scenario.

    When CCU resources are insufficient, the system automatically switches to "2: AI CPU mode".
  <!-- end id4 -->

  <!-- npu="A3" id5 -->
  **For Atlas A3 training products/Atlas A3 inference products, the supported values and their meanings are as follows:**

  - 0: Uses the default operator expansion mode. By default, Atlas A3 training products/Atlas A3 inference products use the AI CPU compute unit on the device side.
  - 2: The communication operator is expanded on the AI CPU compute unit.
  - 3: The communication operator is expanded on the Vector Core compute unit on the device side.
    - This configuration supports only symmetric networking and inference features.
    - In this configuration, if the data size does not meet the running requirements on "Vector Core", some operators automatically switch to the default mode.
    - This configuration item supports only the Broadcast, AllReduce, ReduceScatter, AllGather, AlltoAll, AlltoAllV, and AlltoAllVC operators.
      - For the Broadcast operator, the supported data types are int8, uint8, int16, uint16, int32, uint32, float16, float32, and bfp16. Only single-server communication within a SuperPoD is supported, and only the single-operator mode and Ascend IR graph mode are supported. Multi-server communication and cross-SuperPoD communication are not supported.
      - For the AllReduce operator, the supported data types are int8, int16, int32, float16, float32, and bfp16. The reduce operation types support only sum, max, and min. Only single-server and multi-server communication within a SuperPoD is supported. Cross-SuperPoD communication is not supported.
      - For the ReduceScatter operator, the supported data types are int8, int16, int32, float16, float32, and bfp16. The reduce operation types support only sum, max, and min. Only single-server and multi-server communication within a SuperPoD is supported. Cross-SuperPoD communication is not supported.
      - For the AllGather, AlltoAll, AlltoAllV, and AlltoAllVC operators, the supported data types are int8, uint8, int16, uint16, int32, uint32, float16, float32, and bfp16. Only single-server and multi-server communication within a SuperPoD is supported. Cross-SuperPoD communication is not supported.

    - For the Broadcast, AllReduce, ReduceScatter, AllGather, and AlltoAll (single-server communication scenario) operators, when the data size exceeds a certain value, the system automatically switches to "2: AI CPU mode" to prevent performance degradation (this threshold is not fixed and is adjusted based on factors such as the operator running mode, whether deterministic computation is enabled, and the network scale). For the AlltoAllV, AlltoAllVC, and AlltoAll (multi-server communication scenario) operators, the system does not automatically switch to "2: AI CPU" mode. To avoid performance deterioration, when the maximum communication data size between any two ranks does not exceed 1 MB, it is recommended to configure "3: AIV mode"; otherwise, use "2: AI CPU mode".
    - Under this configuration item, collective communication supports the core control capability. It is recommended that you configure the number of Vector Cores based on the concurrency of computation operators and communication operators in the actual usage scenario.

      - For the Broadcast operator, it is recommended to allocate at least ranksize Vector Cores.
      - For the AllGather and non-deterministic ReduceScatter operators, it is recommended to allocate at least max\(2, ceil\(ranksize/20\)\) Vector Cores.
      - For the AllReduce, deterministic ReduceScatter, AlltoAll, AlltoAllV, and AlltoAllVC operators, it is recommended to allocate at least max\(2, ceil\(ranksize/20\)\) Vector Cores, and the number of cores must be an even number (if the calculated result is an odd number, round it up to the next even number).
      If the number of Vector Cores allocated during service compilation cannot meet the algorithm orchestration requirements, HCCL reports an error and prompts the minimum number of Vector Cores required.

  - 4: The communication operator is expanded on the Vector Core compute units on the device side, but no mode switch occurs as the data size changes. Vector Cores are always used for computation. If the Vector Core running conditions are not met, an error is reported and the process exits.
    - This configuration supports only symmetric networking and inference features.
    - This configuration supports the AllReduce, ReduceScatter, AllGather, AlltoAll, AlltoAllV, and AlltoAllVC operators. For the supported data types and scenario restrictions of the related operators, see configuration "3".
    - Under this configuration, collective communication supports the core control capability. The Vector Core quantity requirements of different operators are the same as those of configuration "3".
  <!-- end id5 -->

  <!-- npu="910b" id6 -->
   **For Atlas A2 training products/Atlas A2 inference products, the supported values and their meanings are as follows:**

  - 0: Uses the default operator expansion mode. Atlas A2 training products/Atlas A2 inference products use the host CPU by default.
  - 1: Communication operators are expanded on the host CPU.
  - 2: Communication operators are expanded on the AI CPU compute unit.

    This configuration item supports only the AllGather, AlltoAll, AlltoAllV, and AlltoAllVC operators.

    In graph mode (Ascend IR) or graph capture (aclgraph) scenarios, when the communication algorithm uses AI CPU mode, the number of concurrent graphs on a single device cannot exceed 6. Otherwise, communication may be blocked because the AI CPU cores are fully occupied.

  - 3: The communication operator is expanded on the Vector Core compute unit on the device side.
    - This configuration supports only symmetric networking and inference features.
    - In this configuration, if the data size does not meet the running requirements on the Vector Core, some operators automatically switch to the default mode.
    - This configuration item supports only the Broadcast, AllReduce, AlltoAll, AlltoAllV, AlltoAllVC, AllGather, ReduceScatter, AllGatherV, and ReduceScatterV operators.
      - For the Broadcast operator, the supported data types are int8, uint8, int16, uint16, int32, uint32, float16, float32, and bfp16. Only the single-operator mode with 8 or fewer devices in a single-server scenario is supported.
      - For the AllReduce operator, the supported data types are int8, int16, int32, float16, float32, and bfp16. The reduce operation types support only sum, max, and min.
      - For the AlltoAll, AlltoAllV, and AlltoAllVC operators, the supported data types are int8, uint8, int16, uint16, int32, uint32, float16, float32, and bfp16. For the AlltoAllV and AlltoAllVC operators, only the single-server scenario is supported. For the AlltoAll operator in graph mode, only the single-server scenario is supported.
      - For the AllGather operator, the supported data types are int8, uint8, int16, uint16, int32, uint32, float16, float32, and bfp16. For this operator in graph mode, only the single-server scenario is supported.
      - For the ReduceScatter operator, the supported data types are int8, int16, int32, float16, float32, and bfp16. The reduce operation types support only sum, max, and min. For this operator in graph mode, only the single-server scenario is supported.
      - For the AllGatherV operator, the supported data types are int8, uint8, int16, uint16, int32, uint32, float16, float32, and bfp16, and only the single-operator mode is supported.
      - For the ReduceScatterV operator, the supported data types are int8, int16, int32, float16, float32, and bfp16, and the reduce operation types support only sum, max, and min.

    - Under this configuration item, collective communication supports the core control capability. It is recommended that you configure the Vector Core quantity based on the concurrency of compute operators and communication operators in the actual usage scenario.

      - For the AllReduce, ReduceScatter, and ReduceScatterV operators, it is recommended to allocate at least 24 cores.
      - For the Broadcast, AlltoAll, AlltoAllV, AlltoAllVC, AllGather, and AllGatherV operators, it is recommended to allocate at least 16 cores.

      If the number of Vector Cores allocated during service compilation cannot meet the algorithm orchestration requirements, HCCL reports an error and prompts the minimum number of Vector Cores required.

  - 4: indicates that the communication operator is expanded on the Vector Core compute units on the device side, but does not switch modes as the data size changes. Vector Cores are always used for computation. If the Vector Core running conditions are not met, an error is reported and the process exits.
    - This configuration supports only symmetric networking and inference features.
    - This configuration item supports only the AllReduce, AlltoAll, AlltoAllV, AlltoAllVC, AllGather, and ReduceScatter operators. For the data types and scenario restrictions supported by the related operators, see configuration "3".
    - Under this configuration item, collective communication supports the core control capability. The Vector Core quantity requirements for different operators are the same as those for configuration "3".
  <!-- end id6 -->

    > [!NOTE] Note
    > - In multi-communicator parallel scenarios, multiple communicators cannot be configured as "3" or "4" (AIV Only mode) at the same time.
    > <!-- npu="910b" id7 -->
    > - For Atlas A2 training products/Atlas A2 inference products, when the communication operator expansion mode is set to "3" or "4" and hcclDeterministic is set to "1" (to enable deterministic computation), in single-server single-operator and graph mode scenarios, when the data size is less than or equal to 8 MB, deterministic computation takes effect only for the AllReduce and ReduceScatter operators. In other scenarios and for other operators, the hcclDeterministic configuration prevails.
    > - For Atlas A2 training products/Atlas A2 inference products, if hcclDeterministic is set to "2" (order preservation enabled), hcclOpExpansionMode cannot be set to "3" or "4", and the order preservation function prevails.
    > <!-- end id7 -->
    > <!-- npu="A3" id8 -->
    > - For Atlas A3 training products/Atlas A3 inference products, when the communication operator expansion mode is set to "3" or "4" and hcclDeterministic is set to "1" (to enable deterministic computation) or "2" (to enable order preservation), when the data size is less than 8 MB, deterministic computation takes effect only for the AllReduce and ReduceScatter operators. In other scenarios and for other operators, the hcclDeterministic configuration prevails.
    > <!-- end id8 -->

- **hcclRdmaTrafficClass**: Configures the traffic class of the RDMA NIC. The value range is \[0,255\], and the value must be an integer multiple of 4.

  In the RoCE V2 protocol, this value corresponds to the Type of Service (ToS) field in the IP packet header. It has 8 bits in total, where bit\[0,1\] is fixed to 0 and bit\[2,7\] is the DSCP. Therefore, dividing this value by 4 yields the DSCP value.

  **Note:**

  - 0xFFFFFFFF is used as the priority determination flag. When this value is configured as 0xFFFFFFFF, the configuration of this communicator is invalid, and the environment variable configuration or the default value 132 is used according to the priority.

- **hcclRdmaServiceLevel**: Configures the service level of the RDMA NIC. The value must be consistent with the PFC priority configured on the NIC. Inconsistent configuration may cause performance deterioration.

  The value must be an unsigned integer in the range \[0,7\].

  **Note:**

  - 0xFFFFFFFF is used as the priority determination flag. When the value is configured as 0xFFFFFFFF, this communicator configuration is invalid, and the environment variable configuration or the default value 4 is used according to the priority.

- **hcclWorldRankID**: Field used in Network Scale Load Balance-Data Plane (NSLB-DP) scenarios. It represents the global rank ID of the current process in the AI framework (such as PyTorch).

  **Note:**

  - This configuration is not supported for Ascend 950PR/Ascend 950DT.

- **hcclJobID**: A field used in NSLB-DP scenarios, representing the unique identifier of the current distributed service, generated by the AI framework.

  **Note:**

  - This configuration is not supported for Ascend 950PR/Ascend 950DT.

- **aclGraphZeroCopyEnable**: This parameter takes effect only for Reduce-type operators in graph capture mode (aclgraph) and controls whether to enable zero-copy.
  - 0 (default): Disables zero-copy.
  - 1: Enables zero-copy.

  **Note**

  - This configuration is not supported for Ascend 950PR/Ascend 950DT.

- **hcclExecTimeOut**: During distributed training or inference, different device processes may execute inconsistent tasks across devices (for example, only specific processes save checkpoint data). This parameter controls the synchronization wait time between devices during execution. Within the configured time, each device process waits for other devices to execute communication synchronization. The unit is s. For the value range and usage constraints for different product types, see the environment variable [HCCL_EXEC_TIMEOUT](https://gitcode.com/cann/hccl/blob/9.1.0/docs/en/user_guide/hccl_env/HCCL_EXEC_TIMEOUT.md).

  **Configuration example:**

    ```text
    # Integer-second configuration, configured as 1800s
    hcclExecTimeOut = 1800
    # Ten-millisecond precision configuration, configured as 50ms
    hcclExecTimeOut = 0.05
    ```

  **Note:**

  - 0xFFFFFFFF is used as the priority determination flag. When it is configured as 0xFFFFFFFF, this communicator configuration is invalid, and the environment variable configuration or the default value 1836 is used according to the priority.
  - Ascend 950PR/Ascend 950DT do not support this configuration. You can configure the global timeout through the HCCL_EXEC_TIMEOUT environment variable.

- **hcclAlgo**: used to configure the inter-server communication algorithm and the inter-SuperPoD communication algorithm for collective communication. It supports two configuration modes: configuring the algorithm type globally and configuring the algorithm type by operator. Note that HCCL provides an adaptive algorithm selection function, which selects an appropriate algorithm by default based on the product form, data size, and number of servers. Generally, you do not need to specify an algorithm manually. If you specify an inter-server communication algorithm through this parameter, the adaptive algorithm selection function no longer takes effect.

  For the parameter information of the configuration modes and the algorithm types supported by different product types, see the environment variable [HCCL_ALGO](https://gitcode.com/cann/hccl/blob/9.1.0/docs/en/user_guide/hccl_env/HCCL_ALGO.md). The configuration modes are as follows:

  - Configure the algorithm type for the communicator: `hcclAlgo = "level0:NA;level1:<algo>;level2:<algo>"`. Example:

    ```text
    hcclAlgo = "level0:NA;level1:H-D_R"
    ```

  - Configure the algorithm type by operator: `hcclAlgo = "<op0>=level0:NA;level1:<algo0>;level2:<algo1>/<op1>=level0:NA;level1:<algo3>;level2:<algo4>"`. Example:

    ```text
    # The AllReduce operator uses the Ring algorithm, the AllGather operator uses the RHD algorithm, and other operators automatically select the communication algorithm based on the product form, number of nodes, and data size.
    hcclAlgo = "allreduce=level0:NA;level1:ring/allgather=level0:NA;level1:H-D_R"
    ```

  **Note:**

  - Ascend 950PR/Ascend 950DT do not support this configuration. You can configure the global communication algorithm through the HCCL_ALGO environment variable.

- **hcclRetryEnable**: Used to configure whether to enable the re-execution feature of HCCL operators. Re-execution means that when a communication operator reports an SDMA or RDMA CQE error during execution, HCCL attempts to re-execute this communication operator. **Supported only on Atlas A3 training products/Atlas A3 inference products.**

  Through this parameter, developers can configure whether to enable the re-execution feature in communicators at two physical levels: inter-server and inter-SuperPoD. Each level supports two states: enabled or disabled. For usage constraints, see the environment variable [HCCL_OP_RETRY_ENABLE](https://gitcode.com/cann/hccl/blob/9.1.0/docs/en/user_guide/hccl_env/HCCL_OP_RETRY_ENABLE.md). The configuration format is `hcclRetryEnable = "L1:1, L2:0"`. The parameter values are as follows:

  - L1 indicates that the physical scope of the communicator is the inter-server communicator. A value of 0 indicates that re-execution is not enabled for inter-server communication tasks within the communicator, and a value of 1 indicates that re-execution is enabled for inter-server communication tasks within the communicator. The default value is 0.
  - L2 indicates that the physical scope of the communicator is the inter-SuperPoD communicator. A value of 0 indicates that re-execution is not enabled for inter-SuperPoD communication tasks within the communicator, and a value of 1 indicates that re-execution is enabled for inter-SuperPoD communication tasks within the communicator. The default value is 0.

- **hcclRetryParams**: Only when developers have enabled the HCCL operator re-execution feature through the **hcclRetryEnable** parameter can this parameter be used to configure the wait time before the first re-execution, the maximum number of re-executions, and the interval between two re-executions. **Supported only on Atlas A3 training products/Atlas A3 inference products.**

  For usage constraints, see the environment variable [HCCL_OP_RETRY_PARAMS](https://gitcode.com/cann/hccl/blob/9.1.0/docs/en/user_guide/hccl_env/HCCL_OP_RETRY_PARAMS.md). The configuration format is `hcclRetryParams = "MaxCnt:3, HoldTime:5000, IntervalTime:1000"`. The parameter values are as follows:

  - MaxCnt: maximum number of retransmissions, of the uint32 type, with a value range of \[1,10\], a default value of 1, and the unit of times.
  - HoldTime: waiting time from the detection of a communication operator execution failure to the start of the first re-execution, of the uint32 type, with a value range of \[0,60000\], a default value of 5000, and the unit of ms.
  - IntervalTime: interval between two re-executions of the same communication operator, of the uint32 type, with a value range of \[0,60000\], a default value of 1000, and the unit of ms.

- **hcclBufferName**: CCLBuffer name. Multiple communicators that use the same buffer name share the same CCLBuffer. If this parameter is not specified, the buffer is not shared by default. The maximum length is 128. Note that communicators that pass in the same CCLBuffer name must deliver operators to the same stream.

  **Note:**

  - Ascend 950PR/Ascend 950DT does not support this configuration.

- **hcclQos**: Used to configure the hyperplane QoS level. Value range: \[0, 7\].

   For Ascend 950PR/Ascend 950DT, the default value is 4.
   For Atlas A3 training products/Atlas A3 inference products and Atlas A2 training products/Atlas A2 inference products, the default value is 6.

- **hcclSymWinMaxMemSizePerRank**: In the HCCS scenario of Atlas A3 training products/Atlas A3 inference products, this parameter indicates the size of symmetric memory reserved for each rank in the current communicator, in GB. Value range: \[1, maximum physical memory allowed to be allocated in the current environment\]. Default value: 16. This parameter takes effect only in the HCCS scenario of Atlas A3 training products/Atlas A3 inference products. In the URMA scenario of Ascend 950PR/Ascend 950DT, the allocated device memory is used to register the symmetric memory window, and the reserved symmetric memory size configured by this parameter is not relied upon.

## Configuration Priority

The preceding configurations are at the communicator level. For some parameters, HCCL also provides global environment variable configuration. The priority is as follows:

- The communicator-level configuration (HcclCommConfig) takes precedence over environment variables.

  If a parameter is configured in HcclCommConfig, the configured value takes effect.

- Environment variables have the next highest priority.

  If the corresponding parameter is not configured in HcclCommConfig but the environment variable is set, the value of the environment variable is used.

- The default value takes effect last.

  If neither HcclCommConfig nor the environment variable is configured, the default value listed in the following table is used.

**Table 1** Configuration priority details

| Configuration Item | Configuration Priority |
| --- | --- |
| hcclBufferSize | Configuration item hcclBufferSize (communicator-level configuration) > environment variable [HCCL_BUFFSIZE](https://gitcode.com/cann/hccl/blob/9.1.0/docs/en/user_guide/hccl_env/HCCL_BUFFSIZE.md) (global configuration) > default value 200. |
| hcclDeterministic | Configuration item hcclDeterministic (communicator-level configuration) > environment variable [HCCL_DETERMINISTIC](https://gitcode.com/cann/hccl/blob/9.1.0/docs/en/user_guide/hccl_env/HCCL_DETERMINISTIC.md) (global configuration) > default deterministic switch (same as the default value of the environment variable HCCL_DETERMINISTIC). |
| hcclOpExpansionMode | Configuration item hcclOpExpansionMode (communicator-level configuration) > environment variable [HCCL_OP_EXPANSION_MODE](https://gitcode.com/cann/hccl/blob/9.1.0/docs/en/user_guide/hccl_env/HCCL_OP_EXPANSION_MODE.md) (global configuration) > default operator expansion mode (same as the default value of the environment variable HCCL_OP_EXPANSION_MODE).|
| hcclRdmaTrafficClass | Configuration item hcclRdmaTrafficClass (communicator-level configuration) > environment variable [HCCL_RDMA_TC](https://gitcode.com/cann/hccl/blob/9.1.0/docs/en/user_guide/hccl_env/HCCL_RDMA_TC.md) (global configuration) > default value 132. |
| hcclRdmaServiceLevel | Configuration item hcclRdmaServiceLevel (communicator-level configuration) > environment variable [HCCL_RDMA_SL](https://gitcode.com/cann/hccl/blob/9.1.0/docs/en/user_guide/hccl_env/HCCL_RDMA_SL.md) (global configuration) > default value 4. |
| hcclExecTimeOut | Configuration item hcclExecTimeOut (communicator-level configuration) > environment variable [HCCL_EXEC_TIMEOUT](https://gitcode.com/cann/hccl/blob/9.1.0/docs/en/user_guide/hccl_env/HCCL_EXEC_TIMEOUT.md) (global configuration) > default timeout (same as the default value of the environment variable HCCL_EXEC_TIMEOUT). |
| hcclAlgo | Configuration item hcclAlgo (communicator-level configuration) > environment variable [HCCL_ALGO](https://gitcode.com/cann/hccl/blob/9.1.0/docs/en/user_guide/hccl_env/HCCL_ALGO.md) (global configuration) > adaptive algorithm selection. |
| hcclRetryEnable | Configuration item hcclRetryEnable (communicator-level configuration) > environment variable [HCCL_OP_RETRY_ENABLE](https://gitcode.com/cann/hccl/blob/9.1.0/docs/en/user_guide/hccl_env/HCCL_OP_RETRY_ENABLE.md) (global configuration) > default value 0. |
| hcclRetryParams | Configuration item hcclRetryParams (communicator-level configuration) > environment variable [HCCL_OP_RETRY_PARAMS](https://gitcode.com/cann/hccl/blob/9.1.0/docs/en/user_guide/hccl_env/HCCL_OP_RETRY_PARAMS.md) (global configuration) > default configuration (MaxCnt: 1, HoldTime: 5000, IntervalTime: 1000). |
